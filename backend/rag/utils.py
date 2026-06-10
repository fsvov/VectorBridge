from collections import defaultdict
from typing import List, Tuple, Dict, Any, Optional
import hashlib
import os
import json
import threading
import time
import requests

from backend.indexing.milvus_client import get_milvus_store
from backend.indexing.embedding import embedding_service as _embedding_service
from backend.indexing.parent_chunk_store import ParentChunkStore
from langchain.chat_models import init_chat_model

# 检索结果 TTL 缓存（避免同一 query 短期内重复检索）
_RETRIEVAL_CACHE_TTL = float(os.getenv("RETRIEVAL_CACHE_TTL", "300"))  # 默认 5 分钟
_retrieval_cache: Dict[str, tuple[float, Dict[str, Any]]] = {}
_retrieval_cache_lock = threading.Lock()
_last_query_image_ocr_error = ""
CTX_ENRICH_ENABLED = os.getenv("CTX_ENRICH_ENABLED", "true").lower() != "false"
CTX_ENRICH_WINDOW = int(os.getenv("CTX_ENRICH_WINDOW", "1"))  # 取前后各 N 个 chunk
DOC_PERMISSION_ENABLED = os.getenv("DOC_PERMISSION_ENABLED", "false").lower() != "false"

LLM_API_KEY = os.getenv("LLM_API_KEY")
MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL")
RERANK_MODEL = os.getenv("RERANK_MODEL")
RERANK_BINDING_HOST = os.getenv("RERANK_BINDING_HOST")
RERANK_API_KEY = os.getenv("RERANK_API_KEY")
AUTO_MERGE_ENABLED = os.getenv("AUTO_MERGE_ENABLED", "true").lower() != "false"
AUTO_MERGE_THRESHOLD = int(os.getenv("AUTO_MERGE_THRESHOLD", "2"))
LEAF_RETRIEVE_LEVEL = int(os.getenv("LEAF_RETRIEVE_LEVEL", "3"))
def _read_positive_int_env(name: str, default: int) -> int:
    try:
        return max(int(os.getenv(name, str(default))), 1)
    except ValueError:
        return default


RETRIEVAL_CANDIDATE_MULTIPLIER = _read_positive_int_env("RETRIEVAL_CANDIDATE_MULTIPLIER", 3)
_RETRIEVAL_CANDIDATE_K_RAW = os.getenv("RETRIEVAL_CANDIDATE_K", "").strip()
CATALOG_RETRIEVAL_CANDIDATE_K = _read_positive_int_env("CATALOG_RETRIEVAL_CANDIDATE_K", 120)


def _read_float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


RERANK_MIN_SCORE = _read_float_env("RERANK_MIN_SCORE", 0.0)
RETRIEVAL_FUSION_METHOD = os.getenv("RETRIEVAL_FUSION_METHOD", "ubg")
IMAGE_RETRIEVAL_MIN_SCORE = _read_float_env("IMAGE_RETRIEVAL_MIN_SCORE", 0.18)
IMAGE_RETRIEVAL_FALLBACK_MIN_SCORE = _read_float_env("IMAGE_RETRIEVAL_FALLBACK_MIN_SCORE", 0.10)
IMAGE_RETRIEVAL_DEBUG_TOP_K = _read_positive_int_env("IMAGE_RETRIEVAL_DEBUG_TOP_K", 3)

RETRIEVAL_TRACE_FIELDS = (
    "retrieval_pipeline",
    "retrieval_mode",
    "candidate_k",
    "candidate_k_source",
    "candidate_k_config_error",
    "retrieval_candidate_multiplier",
    "retrieval_top_k",
    "leaf_retrieve_level",
    "recall_count",
    "post_merge_candidate_count",
    "candidate_count",
    "auto_merge_enabled",
    "auto_merge_applied",
    "auto_merge_threshold",
    "auto_merge_replaced_chunks",
    "auto_merge_steps",
    "rerank_enabled",
    "rerank_applied",
    "rerank_model",
    "rerank_endpoint",
    "rerank_error",
    "rerank_min_score",
    "post_rerank_count",
    "post_threshold_count",
    "retrieval_empty",
    "has_query_image",
    "image_retrieval_enabled",
    "image_retrieval_error",
    "image_retrieval_min_score",
    "image_retrieval_fallback_min_score",
    "image_context_fallback",
    "image_matches",
    "query_image_ocr_enabled",
    "query_image_ocr_text",
    "query_image_ocr_error",
)

# 全局初始化检索依赖（与 api 共用 embedding_service，保证 BM25 状态一致）
_milvus_manager = get_milvus_store()
_parent_chunk_store = ParentChunkStore()

_stepback_model = None


def resolve_candidate_k(top_k: int) -> Tuple[int, Dict[str, Any]]:
    """解析 Milvus 候选池大小；RETRIEVAL_CANDIDATE_K 优先，否则 top_k × multiplier。"""
    if _RETRIEVAL_CANDIDATE_K_RAW:
        try:
            candidate_k = max(int(_RETRIEVAL_CANDIDATE_K_RAW), top_k)
        except ValueError:
            candidate_k = max(top_k * RETRIEVAL_CANDIDATE_MULTIPLIER, top_k)
            return candidate_k, {
                "candidate_k_source": "multiplier",
                "retrieval_candidate_multiplier": RETRIEVAL_CANDIDATE_MULTIPLIER,
                "candidate_k_config_error": "invalid RETRIEVAL_CANDIDATE_K",
            }
        return candidate_k, {
            "candidate_k_source": "env",
            "retrieval_candidate_multiplier": RETRIEVAL_CANDIDATE_MULTIPLIER,
        }
    candidate_k = max(top_k * RETRIEVAL_CANDIDATE_MULTIPLIER, top_k)
    return candidate_k, {
        "candidate_k_source": "multiplier",
        "retrieval_candidate_multiplier": RETRIEVAL_CANDIDATE_MULTIPLIER,
    }


def retrieval_trace_fields(meta: Dict[str, Any]) -> Dict[str, Any]:
    """从 retrieve meta 提取应写入 rag_trace 的检索字段。"""
    return {key: meta[key] for key in RETRIEVAL_TRACE_FIELDS if key in meta and meta[key] is not None}


def merge_retrieval_trace(accumulated: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """合并多路检索 trace（扩展召回）；计数类字段累加，配置类字段保留首次。"""
    incoming = retrieval_trace_fields(meta)
    if not accumulated:
        return incoming
    additive = {
        "recall_count",
        "post_merge_candidate_count",
        "auto_merge_replaced_chunks",
        "auto_merge_steps",
    }
    merged = dict(accumulated)
    for key, value in incoming.items():
        if key in additive:
            merged[key] = int(merged.get(key) or 0) + int(value or 0)
        elif key == "auto_merge_applied":
            merged[key] = bool(merged.get(key)) or bool(value)
        else:
            merged.setdefault(key, value)
    return merged


def _get_rerank_endpoint() -> str:
    if not RERANK_BINDING_HOST:
        return ""
    host = RERANK_BINDING_HOST.strip().rstrip("/")
    if "/rerank" in host:
        return host
    return f"{host}/v1/rerank"


def _effective_score(doc: dict) -> Optional[float]:
    """精排分优先，否则用召回分；用于合并聚合与合并后重排。"""
    rerank_score = doc.get("rerank_score")
    if rerank_score is not None:
        return float(rerank_score)
    score = doc.get("score")
    if score is not None:
        return float(score)
    return None


def _meets_rerank_min_score(doc: dict) -> bool:
    score = _effective_score(doc)
    if score is None:
        return RERANK_MIN_SCORE <= 0
    return score >= RERANK_MIN_SCORE


def _merge_rank_score_into(target: dict, source: dict) -> None:
    incoming = _effective_score(source)
    if incoming is None:
        return
    uses_rerank = source.get("rerank_score") is not None or target.get("rerank_score") is not None
    if uses_rerank:
        existing = target.get("rerank_score")
        if existing is None:
            target["rerank_score"] = incoming
        else:
            target["rerank_score"] = max(float(existing), incoming)
        return
    existing = target.get("score")
    if existing is None:
        target["score"] = incoming
    else:
        target["score"] = max(float(existing), incoming)


def _merge_to_parent_level(docs: List[dict], threshold: int = 2) -> Tuple[List[dict], int]:
    groups: Dict[str, List[dict]] = defaultdict(list)
    for doc in docs:
        parent_id = (doc.get("parent_chunk_id") or "").strip()
        if parent_id:
            groups[parent_id].append(doc)

    merge_parent_ids = [parent_id for parent_id, children in groups.items() if len(children) >= threshold]
    if not merge_parent_ids:
        return docs, 0

    parent_docs = _parent_chunk_store.get_documents_by_ids(merge_parent_ids)
    parent_map = {item.get("chunk_id", ""): item for item in parent_docs if item.get("chunk_id")}

    merged_docs: List[dict] = []
    parent_slot: Dict[str, int] = {}
    merged_count = 0
    for doc in docs:
        parent_id = (doc.get("parent_chunk_id") or "").strip()
        if not parent_id or parent_id not in parent_map:
            merged_docs.append(doc)
            continue

        if parent_id in parent_slot:
            existing = merged_docs[parent_slot[parent_id]]
            _merge_rank_score_into(existing, doc)
            merged_count += 1
            continue

        parent_doc = dict(parent_map[parent_id])
        _merge_rank_score_into(parent_doc, doc)
        parent_doc["merged_from_children"] = True
        parent_doc["merged_child_count"] = len(groups[parent_id])
        parent_slot[parent_id] = len(merged_docs)
        merged_docs.append(parent_doc)
        merged_count += 1

    return merged_docs, merged_count


def _empty_merge_meta() -> Dict[str, Any]:
    return {
        "auto_merge_enabled": AUTO_MERGE_ENABLED,
        "auto_merge_applied": False,
        "auto_merge_threshold": AUTO_MERGE_THRESHOLD,
        "auto_merge_replaced_chunks": 0,
        "auto_merge_steps": 0,
        "post_merge_candidate_count": 0,
    }


def _auto_merge_candidates(docs: List[dict]) -> Tuple[List[dict], Dict[str, Any]]:
    """在完整召回候选上执行 L3→L2→L1 合并；不改变顺序，精排由后续步骤负责。"""
    meta = _empty_merge_meta()
    meta["post_merge_candidate_count"] = len(docs)
    if not AUTO_MERGE_ENABLED or not docs:
        return docs, meta

    merged_docs, merged_count_l3_l2 = _merge_to_parent_level(docs, threshold=AUTO_MERGE_THRESHOLD)
    merged_docs, merged_count_l2_l1 = _merge_to_parent_level(merged_docs, threshold=AUTO_MERGE_THRESHOLD)

    replaced_count = merged_count_l3_l2 + merged_count_l2_l1
    meta.update({
        "auto_merge_applied": replaced_count > 0,
        "auto_merge_replaced_chunks": replaced_count,
        "auto_merge_steps": int(merged_count_l3_l2 > 0) + int(merged_count_l2_l1 > 0),
        "post_merge_candidate_count": len(merged_docs),
    })
    return merged_docs, meta


def _sort_by_rank_score(docs: List[dict]) -> List[dict]:
    return sorted(docs, key=lambda item: _effective_score(item) or 0.0, reverse=True)


def _is_catalog_query(query: str) -> bool:
    q = (query or "").lower()
    has_class = any(token in q for token in ("a类", "a 类", "b类", "b 类", "c类", "c 类"))
    has_kind = "期刊" in q or "会议" in q
    has_domain = any(token in q for token in ("计算机", "网络", "存储", "并行", "分布", "安全", "数据库", "软件", "人工智能"))
    return has_class and has_kind and has_domain


def _domain_query_terms(query: str) -> list[tuple[str, float]]:
    q = (query or "").lower()
    terms: list[tuple[str, float]] = []
    if "期刊" in q:
        terms.extend([("期刊", 20.0), ("journal", 25.0), ("transactions", 20.0)])
    if "会议" in q:
        terms.extend([("会议", 20.0), ("conference", 25.0), ("symposium", 20.0)])
    if "a类" in q or "a 类" in q:
        terms.extend([("一、a 类", 180.0), ("a类", 100.0), ("a 类", 100.0)])
    if "b类" in q or "b 类" in q:
        terms.extend([("二、b 类", 180.0), ("b类", 100.0), ("b 类", 100.0)])
    if "c类" in q or "c 类" in q:
        terms.extend([("三、c 类", 180.0), ("c类", 100.0), ("c 类", 100.0)])
    if "存储" in q:
        terms.extend([
            ("存储系统", 100.0),
            ("storage", 100.0),
            ("systems", 25.0),
            ("cloud computing", 40.0),
            ("transactions on storage", 80.0),
        ])
    if "网络" in q:
        terms.extend([
            ("计算机网络", 120.0),
            ("network", 70.0),
            ("networking", 90.0),
            ("communication", 40.0),
            ("communications", 40.0),
            ("wireless", 40.0),
            ("internet", 35.0),
        ])
    if "并行" in q and "分布" in q:
        terms.extend([
            ("并行与分布计算", 80.0),
            ("parallel and distributed", 100.0),
            ("parallel", 45.0),
            ("distributed", 45.0),
            ("tpds", 80.0),
            ("jpdc", 80.0),
            ("journal of parallel and distributed computing", 120.0),
            ("ieee transactions on parallel and distributed systems", 120.0),
            ("parallel computing", 80.0),
        ])
    return terms


def _apply_domain_lexical_boost(query: str, docs: List[dict]) -> List[dict]:
    terms = _domain_query_terms(query)
    if not terms:
        return docs
    q = (query or "").lower()
    wants_journal = "期刊" in q
    wants_conference = "会议" in q
    wanted_class = None
    if "a类" in q or "a 类" in q:
        wanted_class = "a"
    elif "b类" in q or "b 类" in q:
        wanted_class = "b"
    elif "c类" in q or "c 类" in q:
        wanted_class = "c"

    boosted = []
    for doc in docs:
        item = dict(doc)
        text = (item.get("text") or "").lower()
        boost = sum(weight for term, weight in terms if term in text)
        if wanted_class == "a":
            if "一、a 类" in text or "a 类" in text or "a类" in text:
                boost += 400.0
            elif "二、b 类" in text or "三、c 类" in text:
                boost -= 250.0
        elif wanted_class == "b":
            if "二、b 类" in text or "b 类" in text or "b类" in text:
                boost += 400.0
            elif "一、a 类" in text or "三、c 类" in text:
                boost -= 250.0
        elif wanted_class == "c":
            if "三、c 类" in text or "c 类" in text or "c类" in text:
                boost += 400.0
            elif "一、a 类" in text or "二、b 类" in text:
                boost -= 250.0

        if wants_conference:
            if "会议简称" in text or "推荐国际学术会议" in text or "/conf/" in text:
                boost += 180.0
            if "期刊简称" in text or "推荐国际学术期刊" in text or "/journals/" in text:
                boost -= 220.0
        elif wants_journal:
            if "期刊简称" in text or "推荐国际学术期刊" in text or "/journals/" in text:
                boost += 180.0
            if "会议简称" in text or "推荐国际学术会议" in text or "/conf/" in text:
                boost -= 220.0

        if boost:
            base_score = float(item.get("rerank_score", item.get("score", 0.0)) or 0.0)
            item["score"] = float(item.get("score", 0.0) or 0.0) + boost
            item["domain_lexical_boost"] = round(boost, 3)
            item["domain_boosted_score"] = round(base_score + boost, 3)
        boosted.append(item)
    return _sort_by_rank_score(boosted)


def dedupe_documents(docs: List[dict]) -> List[dict]:
    """按 chunk_id 去重；重复项保留更高 rank 分（rerank_score 优先）。"""
    by_key: Dict[str, dict] = {}
    order: List[str] = []
    for item in docs:
        chunk_id = (item.get("chunk_id") or "").strip()
        key = chunk_id or f"{item.get('filename')}|{item.get('page_number')}|{item.get('text')}"
        if key not in by_key:
            by_key[key] = item
            order.append(key)
            continue
        _merge_rank_score_into(by_key[key], item)
    return [by_key[key] for key in order]


def _rerank_documents(query: str, docs: List[dict], top_k: int) -> Tuple[List[dict], Dict[str, Any]]:
    docs_with_rank = [{**doc, "rrf_rank": i} for i, doc in enumerate(docs, 1)]
    meta: Dict[str, Any] = {
        "rerank_enabled": bool(RERANK_MODEL and RERANK_API_KEY and RERANK_BINDING_HOST),
        "rerank_applied": False,
        "rerank_model": RERANK_MODEL,
        "rerank_endpoint": _get_rerank_endpoint(),
        "rerank_error": None,
        "candidate_count": len(docs_with_rank),
    }
    if not docs_with_rank or not meta["rerank_enabled"]:
        return _sort_by_rank_score(docs_with_rank)[:top_k], meta

    payload = {
        "model": RERANK_MODEL,
        "query": query,
        "documents": [doc.get("text", "") for doc in docs_with_rank],
        "top_n": min(top_k, len(docs_with_rank)),
        "return_documents": False,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RERANK_API_KEY}",
    }
    try:
        meta["rerank_applied"] = True
        response = requests.post(
            meta["rerank_endpoint"],
            headers=headers,
            json=payload,
            timeout=15,
        )
        if response.status_code >= 400:
            meta["rerank_error"] = f"HTTP {response.status_code}: {response.text[:100]}"
            if LLM_API_KEY and MODEL:
                print(f"  [rerank] HTTP {response.status_code}，降级到 LLM Chat 打分...", flush=True)
                return _llm_rerank_fallback(query, docs_with_rank, top_k, meta)
            return _sort_by_rank_score(docs_with_rank)[:top_k], meta

        items = response.json().get("results", [])
        reranked = []
        for item in items:
            idx = item.get("index")
            if isinstance(idx, int) and 0 <= idx < len(docs_with_rank):
                doc = dict(docs_with_rank[idx])
                score = item.get("relevance_score")
                if score is not None:
                    doc["rerank_score"] = score
                reranked.append(doc)

        if reranked:
            return reranked[:top_k], meta

        meta["rerank_error"] = "empty_rerank_results"
        return _sort_by_rank_score(docs_with_rank)[:top_k], meta
    except (requests.RequestException, json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        meta["rerank_error"] = str(e)
        from backend.infra.event_logger import log_rerank_fallback
        log_rerank_fallback(query, str(e))
        if LLM_API_KEY and MODEL:
            return _llm_rerank_fallback(query, docs_with_rank, top_k, meta)
        return _sort_by_rank_score(docs_with_rank)[:top_k], meta


def _llm_rerank_fallback(
    query: str,
    docs: List[dict],
    top_k: int,
    meta: Dict[str, Any],
) -> Tuple[List[dict], Dict[str, Any]]:
    """LLM Chat 降级 rerank：用 chat 模型对文档做 0-10 相关性打分。"""
    if not docs:
        return _sort_by_rank_score(docs)[:top_k], meta

    # 截断文档文本以控制 token 消耗
    truncated_docs = []
    for i, doc in enumerate(docs[:top_k * 3]):  # 最多打分 candidate_k*3 个文档
        text = (doc.get("text", "") or "")[:300]
        truncated_docs.append({"index": i, "text": text})

    if not truncated_docs:
        return _sort_by_rank_score(docs)[:top_k], meta

    doc_lines = "\n".join(
        f"[{d['index']}] {d['text']}" for d in truncated_docs
    )
    prompt = (
        "评估以下文档与用户问题的相关性，为每篇文档打 0-10 的整数分。\n"
        "10=完全相关，5=部分相关，0=无关。\n\n"
        f"用户问题：{query}\n\n"
        f"文档列表：\n{doc_lines}\n\n"
        "输出 JSON 数组（不要其他文字）：[{\"index\":0,\"score\":8},{\"index\":1,\"score\":3},...]"
    )

    try:
        from langchain.chat_models import init_chat_model

        fast_model = os.getenv("FAST_MODEL") or MODEL
        llm = init_chat_model(
            model=fast_model,
            model_provider="openai",
            api_key=LLM_API_KEY,
            base_url=BASE_URL,
            temperature=0,
            request_timeout=30,
            max_retries=1,
        )
        response = (llm.invoke(prompt).content or "").strip()
        # 提取 JSON 数组
        import re as _re

        match = _re.search(r"\[[\s\S]*\]", response)
        if not match:
            meta["rerank_error"] = "llm_fallback_parse_failed"
            meta["rerank_applied"] = False
            return _sort_by_rank_score(docs)[:top_k], meta

        scores = json.loads(match.group())
        for item in scores:
            idx = item.get("index")
            score = item.get("score", 0)
            if isinstance(idx, int) and 0 <= idx < len(docs):
                docs[idx]["rerank_score"] = float(score) / 10.0  # 归一化到 0-1

        meta["rerank_applied"] = True
        meta["rerank_model"] = f"{MODEL} (LLM Chat fallback)"
        meta["rerank_error"] = None
        reranked = _sort_by_rank_score(docs)
        return reranked[:top_k], meta

    except Exception as e:
        meta["rerank_error"] = f"llm_fallback_error: {e}"
        meta["rerank_applied"] = False
        return _sort_by_rank_score(docs)[:top_k], meta


def _get_stepback_model():
    global _stepback_model
    if not LLM_API_KEY or not MODEL:
        return None
    if _stepback_model is None:
        _stepback_model = init_chat_model(
            model=MODEL,
            model_provider="openai",
            api_key=LLM_API_KEY,
            base_url=BASE_URL,
            temperature=0.2,
            request_timeout=60,
            max_retries=1,
        )
    return _stepback_model


def _generate_step_back_question(query: str) -> str:
    model = _get_stepback_model()
    if not model:
        return ""
    prompt = (
        "请将用户的具体问题抽象成更高层次、更概括的‘退步问题’，"
        "用于探寻背后的通用原理或核心概念。只输出退步问题一句话，不要解释。\n"
        f"用户问题：{query}"
    )
    try:
        return (model.invoke(prompt).content or "").strip()
    except Exception:
        return ""


def _answer_step_back_question(step_back_question: str) -> str:
    model = _get_stepback_model()
    if not model or not step_back_question:
        return ""
    prompt = (
        "请简要回答以下退步问题，提供通用原理/背景知识，"
        "控制在120字以内。只输出答案，不要列出推理过程。\n"
        f"退步问题：{step_back_question}"
    )
    try:
        return (model.invoke(prompt).content or "").strip()
    except Exception:
        return ""


def generate_hypothetical_document(query: str) -> str:
    model = _get_stepback_model()
    if not model:
        return ""
    prompt = (
        "请基于用户问题生成一段‘假设性文档’，内容应像真实资料片段，"
        "用于帮助检索相关信息。文档可以包含合理推测，但需与问题语义相关。"
        "只输出文档正文，不要标题或解释。\n"
        f"用户问题：{query}"
    )
    try:
        return (model.invoke(prompt).content or "").strip()
    except Exception:
        return ""


def step_back_expand(query: str) -> dict:
    step_back_question = _generate_step_back_question(query)
    step_back_answer = _answer_step_back_question(step_back_question)
    if step_back_question or step_back_answer:
        expanded_query = (
            f"{query}\n\n"
            f"退步问题：{step_back_question}\n"
            f"退步问题答案：{step_back_answer}"
        )
    else:
        expanded_query = query
    return {
        "step_back_question": step_back_question,
        "step_back_answer": step_back_answer,
        "expanded_query": expanded_query,
    }


def _finalize_retrieval(
    query: str,
    retrieved: List[dict],
    top_k: int,
    retrieval_mode: str,
    candidate_k: int,
    candidate_config: Dict[str, Any],
) -> Dict[str, Any]:
    """生产流水线：召回候选 → Rerank（先精排原始 chunk）→ Auto-merge → top_k → 阈值过滤。"""
    # 先 Rerank 原始 L3 chunk（精排模型看到的是真实检索内容，而非合并后的父块文本）
    reranked_docs, rerank_meta = _rerank_documents(query=query, docs=retrieved, top_k=candidate_k)
    reranked_docs = _apply_domain_lexical_boost(query, reranked_docs)
    # 再 Auto-merge（合并后父块继承子块最高精排分）
    merged_docs, merge_meta = _auto_merge_candidates(reranked_docs)
    # 取 top_k
    final_docs = _sort_by_rank_score(merged_docs)[:top_k]
    # 阈值过滤
    final_docs = [d for d in final_docs if _meets_rerank_min_score(d)]
    post_rerank_count = len(reranked_docs)
    meta = {
        **rerank_meta,
        **merge_meta,
        **candidate_config,
        "retrieval_mode": retrieval_mode,
        "retrieval_pipeline": "recall_rerank_merge",
        "candidate_k": candidate_k,
        "retrieval_top_k": top_k,
        "leaf_retrieve_level": LEAF_RETRIEVE_LEVEL,
        "recall_count": len(retrieved),
        "rerank_min_score": RERANK_MIN_SCORE,
        "post_rerank_count": post_rerank_count,
        "post_threshold_count": len(final_docs),
        "retrieval_empty": len(final_docs) == 0,
    }
    return {"docs": final_docs, "meta": meta}


def _cache_key_for_query(query: str) -> str:
    return hashlib.sha256(query.strip().lower().encode()).hexdigest()


def _get_cached_retrieval(query: str) -> Optional[Dict[str, Any]]:
    key = _cache_key_for_query(query)
    with _retrieval_cache_lock:
        entry = _retrieval_cache.get(key)
        if entry and time.time() - entry[0] < _RETRIEVAL_CACHE_TTL:
            return entry[1]
        if entry:
            del _retrieval_cache[key]
    return None


def _set_cached_retrieval(query: str, result: Dict[str, Any]) -> None:
    key = _cache_key_for_query(query)
    with _retrieval_cache_lock:
        _retrieval_cache[key] = (time.time(), result)
        # 限制缓存大小
        if len(_retrieval_cache) > 200:
            oldest = min(_retrieval_cache.items(), key=lambda x: x[1][0])
            del _retrieval_cache[oldest[0]]


def _clear_retrieval_cache() -> None:
    with _retrieval_cache_lock:
        _retrieval_cache.clear()


def _extract_query_image_ocr_text(query_image_path: str | None) -> str:
    global _last_query_image_ocr_error
    _last_query_image_ocr_error = ""
    if not query_image_path:
        return ""
    try:
        from backend.rag.ocr import extract_text_from_image
        result = extract_text_from_image(query_image_path)
        _last_query_image_ocr_error = result.get("error") or ""
        return (result.get("text") or "").strip()
    except Exception as exc:
        _last_query_image_ocr_error = str(exc)[:300]
        return ""


def _build_ocr_augmented_query(query: str, ocr_text: str) -> str:
    ocr_text = (ocr_text or "").strip()
    if not ocr_text:
        return query
    return f"{query}\n\n图片OCR文字：{ocr_text}"


# ─── RSE + Contextual Compression ─────────────────────────────────────────────

RSE_ENABLED = os.getenv("RSE_ENABLED", "false").lower() != "false"
COMPRESSION_ENABLED = os.getenv("COMPRESSION_ENABLED", "false").lower() != "false"
RSE_PROMPT = (
    "从以下文档片段中，只提取与用户问题直接相关的句子。不相关的句子直接丢弃。保留原文表述，不要改写。如果没有相关句子，输出空。\n\n"
    "用户问题：{query}\n\n文档片段：\n{chunks_text}\n\n"
    "输出格式（只输出提取结果，不要解释）：\n[片段1相关句子]\n---\n[片段2相关句子]\n---"
)
COMPRESSION_PROMPT = (
    "将以下检索结果压缩为精简的上下文。删除重复信息，保留关键事实（数字、日期、名称、事件），输出控制在一半长度以内。\n\n"
    "用户问题：{query}\n\n检索结果：\n{context}\n\n压缩后的上下文（直接输出，不要解释）："
)


def _call_flash_llm(prompt: str) -> str:
    try:
        from langchain.chat_models import init_chat_model
        fm = os.getenv("FAST_MODEL") or MODEL
        llm = init_chat_model(model=fm, model_provider="openai", api_key=LLM_API_KEY, base_url=BASE_URL, temperature=0, request_timeout=60, max_retries=1)
        return (llm.invoke(prompt).content or "").strip()
    except Exception:
        return ""


def extract_relevant_segments(query: str, docs: List[dict]) -> List[dict]:
    if not RSE_ENABLED or not docs: return docs
    texts = [f"[{i}] {d.get('text','')[:500]}" for i, d in enumerate(docs)]
    result = _call_flash_llm(RSE_PROMPT.format(query=query, chunks_text="\n\n".join(texts)))
    if not result: return docs
    parts = result.split("---")
    for i, part in enumerate(parts):
        if i < len(docs) and part.strip(): docs[i]["text"] = part.strip()
    return docs


def compress_context(query: str, docs: List[dict]) -> str:
    if not COMPRESSION_ENABLED or not docs: return ""
    ctx = "\n\n".join(f"[{i}] {d.get('text','')[:600]}" for i, d in enumerate(docs[:10]))
    return _call_flash_llm(COMPRESSION_PROMPT.format(query=query, context=ctx[:5000]))


# ─── CRAG ─────────────────────────────────────────────────────────────────────

CRAG_ENABLED = os.getenv("CRAG_ENABLED", "true").lower() != "false"


def crag_retrieve(query: str, top_k: int) -> tuple[List[dict], dict]:
    result = retrieve_documents(query, top_k=top_k)
    docs = result.get("docs", [])
    meta = result.get("meta", {})
    if len(docs) >= 2: return docs, meta
    try:
        from backend.indexing.milvus_client import get_milvus_store
        ms = get_milvus_store()
        emb = _embedding_service.get_embeddings([query])[0]
        sparse = _embedding_service.get_sparse_embedding(query)
        l2 = ms.hybrid_retrieve(emb, sparse, top_k=top_k * 2, filter_expr="chunk_level >= 2")
        if l2: docs = l2[:top_k]; meta["retrieval_mode"] = "crag_l2"; meta["crag_applied"] = True
    except Exception: pass
    if len(docs) < 2:
        try:
            ms = get_milvus_store()
            emb = _embedding_service.get_embeddings([query])[0]
            sparse = _embedding_service.get_sparse_embedding(query)
            wide = ms.hybrid_retrieve(emb, sparse, top_k=top_k * 5, filter_expr="chunk_level >= 2")
            if wide: docs = wide[:top_k]; meta["retrieval_mode"] = "crag_wide"; meta["crag_applied"] = True
        except Exception: pass
    return docs, meta


def _enrich_with_adjacent_chunks(docs: List[dict], filter_expr: str) -> List[dict]:
    if not docs or CTX_ENRICH_WINDOW < 1:
        return docs
    to_fetch_by_file: dict[str, set[int]] = {}
    for doc in docs:
        idx = doc.get("chunk_idx")
        if idx is None: continue
        filename = doc.get("filename", "")
        if not filename:
            continue
        to_fetch = to_fetch_by_file.setdefault(filename, set())
        for offset in range(1, CTX_ENRICH_WINDOW + 1):
            to_fetch.add(idx - offset)
            to_fetch.add(idx + offset)
    if not to_fetch_by_file: return docs
    try:
        extra = []
        for filename, to_fetch in to_fetch_by_file.items():
            quoted = ", ".join(str(i) for i in to_fetch if i >= 0)
            if not quoted:
                continue
            file_filter = _join_filter(filter_expr, f'filename == "{filename.replace(chr(34), chr(92) + chr(34))}"')
            extra.extend(_milvus_manager.query(
                filter_expr=f"chunk_idx in [{quoted}] && {file_filter}",
                output_fields=["chunk_id", "text", "filename", "page_number", "chunk_idx", "chunk_level"],
                limit=len(to_fetch) * 2,
            ))
        seen = {d.get("chunk_id") for d in docs}
        for e in extra:
            if e.get("chunk_id", "") not in seen:
                e["score"] = 0.0
                docs.append(e)
    except Exception: pass
    return docs


def retrieve_documents(
    query: str, top_k: int = 5,
    query_image_path: Optional[str] = None,
    use_image: bool = True,
) -> Dict[str, Any]:
    cache_key = f"{query}__img={query_image_path or 'none'}"
    cached = _get_cached_retrieval(cache_key)
    if cached is not None:
        return cached

    candidate_k, candidate_config = resolve_candidate_k(top_k)
    if _is_catalog_query(query) and candidate_k < CATALOG_RETRIEVAL_CANDIDATE_K:
        candidate_k = CATALOG_RETRIEVAL_CANDIDATE_K
        candidate_config = {
            **candidate_config,
            "candidate_k_source": "catalog_query",
            "catalog_candidate_k": CATALOG_RETRIEVAL_CANDIDATE_K,
        }
    filter_expr = f"chunk_level == {LEAF_RETRIEVE_LEVEL}"
    if DOC_PERMISSION_ENABLED:
        filter_expr += ' && file_visibility == "public"'

    use_ubg = RETRIEVAL_FUSION_METHOD == "ubg"
    query_image_ocr_text = ""
    query_for_text_retrieval = query
    if query_image_path:
        query_image_ocr_text = _extract_query_image_ocr_text(query_image_path)
        query_for_text_retrieval = _build_ocr_augmented_query(query, query_image_ocr_text)

    try:
        dense_embeddings = _embedding_service.get_embeddings([query_for_text_retrieval])
        dense_embedding = dense_embeddings[0]
        sparse_embedding = _embedding_service.get_sparse_embedding(query_for_text_retrieval)

        # 图片向量（如有）。不要先检查 available；embed_* 会按需懒加载 CLIP。
        image_embedding = None
        image_retrieval_error = None
        if use_image:
            try:
                from backend.indexing.multimodal_embedding import get_multimodal_embedding_service
                mm_svc = get_multimodal_embedding_service()
                if query_image_path:
                    image_embedding = mm_svc.embed_image(query_image_path)
                elif mm_svc.available:
                    image_embedding = mm_svc.embed_text(query_for_text_retrieval)
            except Exception as e:
                image_retrieval_error = str(e)

        if query_image_path and _vector_has_values(image_embedding):
            retrieved, image_match_meta = _image_first_search(
                image_embedding, candidate_k, filter_expr,
                dense_embedding=dense_embedding,
                sparse_embedding=sparse_embedding,
            )
            retrieval_mode = "image_first"
        elif use_ubg:
            retrieved = _ubg_weighted_search(
                dense_embedding, sparse_embedding, image_embedding,
                query_for_text_retrieval, candidate_k, filter_expr,
            )
            retrieval_mode = "ubg_fusion"
        else:
            retrieved = _milvus_manager.hybrid_retrieve(
                dense_embedding=dense_embedding,
                sparse_embedding=sparse_embedding,
                top_k=candidate_k,
                filter_expr=filter_expr,
            )
            retrieval_mode = "hybrid"

        result = _finalize_retrieval(
            query=query_for_text_retrieval,
            retrieved=retrieved,
            top_k=top_k,
            retrieval_mode=retrieval_mode,
            candidate_k=candidate_k,
            candidate_config=candidate_config,
        )
        result["meta"]["has_query_image"] = bool(query_image_path)
        result["meta"]["image_retrieval_enabled"] = _vector_has_values(image_embedding)
        result["meta"]["image_retrieval_error"] = image_retrieval_error
        result["meta"]["image_retrieval_min_score"] = IMAGE_RETRIEVAL_MIN_SCORE
        result["meta"]["image_retrieval_fallback_min_score"] = IMAGE_RETRIEVAL_FALLBACK_MIN_SCORE
        result["meta"]["query_image_ocr_enabled"] = bool(query_image_path)
        result["meta"]["query_image_ocr_text"] = query_image_ocr_text
        result["meta"]["query_image_ocr_error"] = _last_query_image_ocr_error
        if query_image_path and _vector_has_values(image_embedding):
            result["meta"].update(image_match_meta)
        if CTX_ENRICH_ENABLED:
            result["docs"] = _enrich_with_adjacent_chunks(result["docs"], filter_expr)
        _set_cached_retrieval(cache_key, result)
        return result
    except Exception as e:
        from backend.infra.event_logger import log_hybrid_fallback
        log_hybrid_fallback(query, str(e))
        try:
            dense_embeddings = _embedding_service.get_embeddings([query_for_text_retrieval])
            dense_embedding = dense_embeddings[0]
            retrieved = _milvus_manager.dense_retrieve(
                dense_embedding=dense_embedding,
                top_k=candidate_k,
                filter_expr=filter_expr,
            )
            result = _finalize_retrieval(
                query=query_for_text_retrieval,
                retrieved=retrieved,
                top_k=top_k,
                retrieval_mode="dense_fallback",
                candidate_k=candidate_k,
                candidate_config=candidate_config,
            )
            if CTX_ENRICH_ENABLED:
                result["docs"] = _enrich_with_adjacent_chunks(result["docs"], filter_expr)
            _set_cached_retrieval(cache_key, result)
            return result
        except Exception:
            return _empty_retrieval_result(top_k, candidate_k, candidate_config)


def _vector_has_values(vector) -> bool:
    if vector is None:
        return False
    try:
        return bool(vector.any())
    except AttributeError:
        return any(abs(float(x)) > 1e-8 for x in vector)
    except Exception:
        return False


def _as_list_vector(vector) -> list[float]:
    if hasattr(vector, "tolist"):
        return vector.tolist()
    return list(vector)


def _join_filter(base: str, extra: str) -> str:
    if not base:
        return extra
    if not extra:
        return base
    return f"{base} && {extra}"


def _image_first_search(
    image_emb,
    top_k: int,
    filter_expr: str,
    dense_embedding: list | None = None,
    sparse_embedding: dict | None = None,
) -> tuple[list[dict], dict]:
    """带图查询优先使用图片向量召回，避免泛文本把结果带偏。"""
    image_filter_expr = _join_filter(filter_expr, 'image_kind != ""')
    image_results = _milvus_manager.image_retrieve(
        _as_list_vector(image_emb),
        top_k=top_k,
        filter_expr=image_filter_expr,
    )
    debug_matches = [
        {
            "filename": doc.get("filename", ""),
            "page_number": doc.get("page_number", 0),
            "score": float(doc.get("score", 0.0) or 0.0),
            "image_score": float(doc.get("score", 0.0) or 0.0),
            "image_kind": doc.get("image_kind", ""),
            "image_page": doc.get("image_page", doc.get("page_number", 0)),
            "image_sort_order": doc.get("image_sort_order", -999),
            "chunk_id": doc.get("chunk_id", ""),
            "text": (doc.get("text", "") or "")[:500],
        }
        for doc in image_results[:IMAGE_RETRIEVAL_DEBUG_TOP_K]
    ]
    filtered = [
        {**doc, "_image_score": doc.get("score", 0.0)}
        for doc in image_results
        if (doc.get("score", 0.0) or 0.0) >= IMAGE_RETRIEVAL_MIN_SCORE
    ]
    fallback = False
    if not filtered:
        filtered = [
            {**doc, "_image_score": doc.get("score", 0.0), "_image_context_fallback": True}
            for doc in image_results[:IMAGE_RETRIEVAL_DEBUG_TOP_K]
            if (doc.get("score", 0.0) or 0.0) >= IMAGE_RETRIEVAL_FALLBACK_MIN_SCORE
        ]
        fallback = bool(filtered)
    meta = {
        "image_matches": debug_matches,
        "image_context_fallback": fallback,
    }
    if not filtered:
        return [], meta

    # 只在图片召回候选内部做轻量文本加分，避免通用查询文本引入无关文档。
    score_by_id = {
        doc.get("chunk_id", doc.get("id", "")): float(doc.get("_image_score", 0.0) or 0.0)
        for doc in filtered
    }
    try:
        if dense_embedding is not None:
            dense_results = _milvus_manager.dense_retrieve(dense_embedding, top_k=top_k, filter_expr=filter_expr)
            for doc in dense_results:
                cid = doc.get("chunk_id", doc.get("id", ""))
                if cid in score_by_id:
                    score_by_id[cid] += 0.05 * float(doc.get("score", 0.0) or 0.0)
        if sparse_embedding is not None:
            sparse_results = _milvus_manager.sparse_retrieve(sparse_embedding, top_k=top_k, filter_expr=filter_expr)
            for doc in sparse_results:
                cid = doc.get("chunk_id", doc.get("id", ""))
                if cid in score_by_id:
                    score_by_id[cid] += 0.05 * float(doc.get("score", 0.0) or 0.0)
    except Exception:
        pass

    for doc in filtered:
        cid = doc.get("chunk_id", doc.get("id", ""))
        doc["score"] = score_by_id.get(cid, float(doc.get("score", 0.0) or 0.0))
    return sorted(filtered, key=lambda d: d.get("score", 0.0) or 0.0, reverse=True)[:top_k], meta


def _ubg_weighted_search(
    dense_emb: list, sparse_emb: dict, image_emb,
    query: str, top_k: int, filter_expr: str,
) -> list[dict]:
    """UBG 加权三路检索合并。"""
    # 分别检索
    dense_results = _milvus_manager.dense_retrieve(dense_emb, top_k=top_k * 2, filter_expr=filter_expr)
    sparse_results = _milvus_manager.sparse_retrieve(sparse_emb, top_k=top_k * 2, filter_expr=filter_expr)
    image_results = []
    if _vector_has_values(image_emb):
        try:
            image_results = _milvus_manager.image_retrieve(
                _as_list_vector(image_emb),
                top_k=top_k,
                filter_expr=_join_filter(filter_expr, 'image_kind != ""'),
            )
        except Exception as _img_e:
            import logging
            logging.getLogger(__name__).warning(f"Image retrieval failed: {_img_e}")

    # 计算每路分数统计
    from backend.rag.ubg_fusion import (
        compute_score_stats, load_ubg_fusion, get_ubg_heuristic,
    )
    dense_scores = [d.get("score", 0.0) for d in dense_results]
    sparse_scores = [s.get("score", 0.0) for s in sparse_results]
    image_scores = [i.get("score", 0.0) for i in image_results]

    d_stats = compute_score_stats(dense_scores)
    s_stats = compute_score_stats(sparse_scores)
    i_stats = compute_score_stats(image_scores)

    # 获取权重
    has_image = len(image_results) > 0
    ubg_model = load_ubg_fusion()
    if ubg_model is not None:
        import numpy as np
        import torch
        device = next(ubg_model.parameters()).device
        query_vec = torch.tensor(dense_emb, device=device).unsqueeze(0)
        stats_vec = torch.tensor([[
            d_stats["mean"], d_stats["std"], d_stats["min"], d_stats["max"],
            s_stats["mean"], s_stats["std"], s_stats["min"], s_stats["max"],
            i_stats["mean"], i_stats["std"], i_stats["min"], i_stats["max"],
        ]], device=device)
        with torch.no_grad():
            u_weights = ubg_model(query_vec, stats_vec)[0]
        weights = {
            "dense": float(u_weights[0]),
            "sparse": float(u_weights[1]),
            "image": float(u_weights[2]) if has_image else 0.0,
        }
    else:
        heuristic = get_ubg_heuristic()
        import numpy as np
        query_vec = np.array(dense_emb)
        weights = heuristic.compute_weights(query_vec, None, has_image)

    # 加权合并：按 chunk_id 去重，分数 = 各路分数 × 权重 的加权和
    merged: dict[str, dict] = {}
    for doc in dense_results:
        cid = doc.get("chunk_id", doc.get("id", ""))
        merged[cid] = {**doc, "_ubg_score": doc.get("score", 0.0) * weights["dense"]}

    for doc in sparse_results:
        cid = doc.get("chunk_id", doc.get("id", ""))
        s = doc.get("score", 0.0) * weights["sparse"]
        if cid in merged:
            merged[cid]["_ubg_score"] += s
        else:
            merged[cid] = {**doc, "_ubg_score": s}

    for doc in image_results:
        cid = doc.get("chunk_id", doc.get("id", ""))
        s = doc.get("score", 0.0) * weights["image"]
        if cid in merged:
            merged[cid]["_ubg_score"] += s
        else:
            merged[cid] = {**doc, "_ubg_score": s}

    # 按加权分数排序
    sorted_docs = sorted(merged.values(), key=lambda d: d["_ubg_score"], reverse=True)
    for d in sorted_docs:
        d["score"] = d.pop("_ubg_score", d.get("score", 0.0))
    return sorted_docs[:top_k]


def _empty_retrieval_result(top_k, candidate_k, candidate_config) -> Dict[str, Any]:
    return {
        "docs": [],
        "meta": {
            "rerank_enabled": bool(RERANK_MODEL and RERANK_API_KEY and RERANK_BINDING_HOST),
            "rerank_applied": False,
            "rerank_model": RERANK_MODEL,
            "rerank_endpoint": _get_rerank_endpoint(),
            "rerank_error": "retrieve_failed",
            "retrieval_mode": "failed",
            "retrieval_pipeline": "recall_merge_rerank",
            "candidate_k": candidate_k,
            **candidate_config,
            "retrieval_top_k": top_k,
            "leaf_retrieve_level": LEAF_RETRIEVE_LEVEL,
            "recall_count": 0,
            **_empty_merge_meta(),
            "candidate_count": 0,
            "rerank_min_score": RERANK_MIN_SCORE,
            "post_rerank_count": 0,
            "post_threshold_count": 0,
            "retrieval_empty": True,
        },
    }
