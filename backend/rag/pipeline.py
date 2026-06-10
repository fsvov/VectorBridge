from typing import Annotated, Literal, TypedDict, List, Optional
import json
import operator
import os
import time as _time_module
import importlib
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, END
from langgraph.types import Send
from pydantic import BaseModel, Field

from backend.rag.utils import (
    retrieve_documents,
    step_back_expand,
    generate_hypothetical_document,
    dedupe_documents,
    retrieval_trace_fields,
    merge_retrieval_trace,
    crag_retrieve,
    CRAG_ENABLED,
    extract_relevant_segments,
    compress_context,
    RSE_ENABLED,
    COMPRESSION_ENABLED,
)
# 延迟导入避免循环引用 (pipeline → chat.streaming → chat.__init__ → chat.service → chat.runtime → tools → pipeline)
_streaming_mod = None


def _get_streaming():
    global _streaming_mod
    if _streaming_mod is None:
        _streaming_mod = importlib.import_module("backend.chat.streaming")
    return _streaming_mod


def _emit_rag_step(icon: str, label: str, detail: str = ""):
    _get_streaming().emit_rag_step(icon, label, detail)


def _set_sub_agent_group(group: str):
    _get_streaming().set_sub_agent_group(group)


def _clear_sub_agent_group():
    _get_streaming().clear_sub_agent_group()

API_KEY = os.getenv("LLM_API_KEY")
MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL")
GRADE_MODEL = os.getenv("GRADE_MODEL", "")
FAST_MODEL = os.getenv("FAST_MODEL") or MODEL
AUX_MODEL = os.getenv("AUX_MODEL") or FAST_MODEL
RAG_FUSION_ENABLED = os.getenv("RAG_FUSION_ENABLED", "true").lower() != "false"
RAG_FUSION_VARIANTS = int(os.getenv("RAG_FUSION_VARIANTS", "3"))
INTENT_ROUTER_ENABLED = os.getenv("INTENT_ROUTER_ENABLED", "true").lower() != "false"

_grader_model = None
_router_model = None
_complexity_model = None
_aux_model = None
_fusion_model = None
_intent_model = None


def _get_grader_model():
    global _grader_model
    if not API_KEY or not GRADE_MODEL:
        return None
    if _grader_model is None:
        _grader_model = init_chat_model(
            model=GRADE_MODEL,
            model_provider="openai",
            api_key=API_KEY,
            base_url=BASE_URL,
            temperature=0,
            stream_usage=True,
            request_timeout=30,
            max_retries=1,
        )
    return _grader_model


def _get_router_model():
    global _router_model
    if not API_KEY or not MODEL:
        return None
    if _router_model is None:
        _router_model = init_chat_model(
            model=MODEL,
            model_provider="openai",
            api_key=API_KEY,
            base_url=BASE_URL,
            temperature=0,
            stream_usage=True,
            request_timeout=60,
            max_retries=1,
        )
    return _router_model


def _get_complexity_model():
    """FAST_MODEL 用于问题复杂度分类和子问题分解。"""
    global _complexity_model
    if not API_KEY or not FAST_MODEL:
        return None
    if _complexity_model is None:
        _complexity_model = init_chat_model(
            model=FAST_MODEL,
            model_provider="openai",
            api_key=API_KEY,
            base_url=BASE_URL,
            temperature=0,
            stream_usage=True,
            request_timeout=30,
            max_retries=1,
        )
    return _complexity_model


def _get_aux_model():
    """AUX_MODEL 用于需要更强推理的任务（冲突检测、幻觉验证、笔记压缩）。"""
    global _aux_model
    if not API_KEY or not AUX_MODEL:
        return _get_complexity_model()
    if _aux_model is None:
        _aux_model = init_chat_model(
            model=AUX_MODEL,
            model_provider="openai",
            api_key=API_KEY,
            base_url=BASE_URL,
            temperature=0,
            stream_usage=True,
        )
    return _aux_model


GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question. \n "
    "Here is the retrieved document: \n\n {context} \n\n"
    "Here is the user question: {question} \n"
    "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
    "Return only one lowercase word: yes or no. Do not return JSON, markdown, or explanations."
)


class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""

    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


def _parse_grade_score(raw: str) -> str:
    text = (raw or "").strip().lower()
    if not text:
        return "unknown"
    first_token = text.replace("：", ":").replace("，", ",").split()[0].strip(".,:;\"'`[]{}()")
    if first_token in ("yes", "y", "true", "relevant"):
        return "yes"
    if first_token in ("no", "n", "false", "irrelevant", "unrelated"):
        return "no"
    if "not relevant" in text or "irrelevant" in text or "unrelated" in text:
        return "no"
    if "relevant" in text or "yes" in text:
        return "yes"
    if "no" in text:
        return "no"
    return "unknown"


class RewriteStrategy(BaseModel):
    """Choose a query expansion strategy."""

    strategy: Literal["step_back", "hyde", "complex"]


class ComplexityResult(BaseModel):
    """问题复杂度分类结果。"""

    complexity: Literal["simple", "complex"] = Field(
        description="问题复杂度：'simple' 为简单问题，'complex' 为复杂问题"
    )
    reason: str = Field(default="", description="分类理由")


class SubQuestions(BaseModel):
    """复杂问题分解后的子问题列表。"""

    sub_questions: List[str] = Field(
        description="2-4 个独立子问题，每个聚焦原问题的一个方面",
        min_length=1,
        max_length=4,
    )


class CoverageResult(BaseModel):
    """子问题覆盖度验证结果。"""

    coverage: Literal["adequate", "insufficient"] = Field(
        description="子问题是否充分覆盖了原问题的核心维度"
    )
    missing: str = Field(default="", description="缺失的维度描述（覆盖不足时填写）")
    suggestion: str = Field(default="", description="建议补充的子问题（覆盖不足时填写）")


class ConflictResult(BaseModel):
    """多文档冲突检测结果。"""

    has_conflict: bool = Field(default=False, description="是否存在文档间明显冲突")
    conflict_detail: str = Field(default="", description="冲突描述")


class HallucinationCheck(BaseModel):
    """答案幻觉检测结果。"""

    verdict: Literal["supported", "partially_unsupported"] = Field(
        description="答案是否完全被提供的文档支撑"
    )
    unsupported_claims: str = Field(default="", description="未被文档支撑的具体声明")


class RAGState(TypedDict):
    question: str
    query_image_path: Optional[str]
    query: str
    context: str
    docs: List[dict]
    route: Optional[str]
    expansion_type: Optional[str]
    expanded_query: Optional[str]
    step_back_question: Optional[str]
    step_back_answer: Optional[str]
    hypothetical_doc: Optional[str]
    rag_trace: Optional[dict]
    # 复杂度路由新增字段
    complexity: Optional[str]
    complexity_reason: Optional[str]
    sub_questions: Optional[List[str]]
    is_sub_agent: bool
    sub_results: Annotated[List[dict], operator.add]


# ─── RAG-Fusion ─────────────────────────────────────────────────────────────

FUSION_PROMPT = (
    "将以下用户问题改写为 {n} 个不同角度的搜索查询变体，以提高检索命中率。\n"
    "每个变体应聚焦原问题的一个不同方面或使用不同的表述方式。\n"
    "输出 JSON 数组（不要其他文字）：[\"变体1\", \"变体2\", ...]\n\n"
    "用户问题：{question}"
)


def _get_fusion_model():
    global _fusion_model
    if not API_KEY or not FAST_MODEL:
        return None
    if _fusion_model is None:
        _fusion_model = init_chat_model(
            model=FAST_MODEL, model_provider="openai",
            api_key=API_KEY, base_url=BASE_URL, temperature=0.3, stream_usage=True,
            request_timeout=30, max_retries=1,
        )
    return _fusion_model


def _generate_query_variants(question: str, n: int = 3) -> List[str]:
    model = _get_fusion_model()
    if not model or n <= 1:
        return [question]
    try:
        prompt = FUSION_PROMPT.format(n=n, question=question)
        result = (model.invoke(prompt).content or "").strip()
        import re as _re
        match = _re.search(r"\[[\s\S]*\]", result)
        if match:
            variants = json.loads(match.group())
            if isinstance(variants, list) and len(variants) > 0:
                return [str(v) for v in variants[:n]]
    except Exception:
        pass
    return [question]


def _fusion_retrieve(question: str, top_k: int) -> tuple[List[dict], dict]:
    """RAG-Fusion: 生成 N 个查询变体 → 独立检索 → RRF 融合去重。"""
    variants = _generate_query_variants(question, n=RAG_FUSION_VARIANTS)
    _emit_rag_step("🔀", f"RAG-Fusion", f"生成 {len(variants)} 个查询变体")
    from backend.infra.event_logger import log_event
    log_event("fusion_generated", {"original": question[:80], "variants": len(variants)})

    all_results: List[dict] = []
    all_meta: dict = {}

    for vi, variant in enumerate(variants):
        # 每个变体检索 top_k/2+1，汇总后去重
        k = max(3, top_k // 2 + 1)
        result = retrieve_documents(variant, top_k=k)
        docs = result.get("docs", [])
        # 标记来源变体
        for d in docs:
            d["_fusion_variant"] = vi
        all_results.extend(docs)
        if vi == 0:
            all_meta = result.get("meta", {})

    # 按 chunk_id 去重，保留最高分
    seen: dict[str, dict] = {}
    for doc in all_results:
        cid = doc.get("chunk_id", "")
        key = cid or f"{doc.get('filename')}|{doc.get('page_number')}|{doc.get('text', '')[:50]}"
        if key not in seen:
            seen[key] = doc
        else:
            existing_score = seen[key].get("score", 0) or 0
            new_score = doc.get("score", 0) or 0
            if new_score > existing_score:
                seen[key] = doc

    merged = sorted(seen.values(), key=lambda d: d.get("score", 0) or 0, reverse=True)
    _emit_rag_step("✅", f"Fusion 完成", f"{len(variants)} 变体 → {len(all_results)} 原始 → {len(merged)} 去重")

    all_meta["fusion_variants"] = len(variants)
    all_meta["fusion_raw_count"] = len(all_results)
    all_meta["fusion_merged_count"] = len(merged)
    return merged[:top_k], all_meta


# ─── Intent Router ───────────────────────────────────────────────────────────

class IntentResult(BaseModel):
    intent: Literal["factual", "conceptual", "comparative", "out_of_scope"] = Field(
        description="查询意图类型"
    )

INTENT_PROMPT = (
    "分类以下用户查询的意图类型：\n"
    "- factual: 查询具体事实、数字、日期、名称、属性值（如'营收多少''成立于哪年'）\n"
    "- conceptual: 概念解释、原理说明、背景知识（如'什么是''如何理解'）\n"
    "- comparative: 多对象对比、跨文档分析（如'对比A和B''哪个更好'）\n"
    "- out_of_scope: 与知识库无关的问题、闲聊、天气等\n\n"
    "用户查询：{question}\n\n"
    "只输出意图类型（一个单词）。"
)


def _get_intent_model():
    global _intent_model
    if not API_KEY or not FAST_MODEL:
        return None
    if _intent_model is None:
        _intent_model = init_chat_model(
            model=FAST_MODEL, model_provider="openai",
            api_key=API_KEY, base_url=BASE_URL, temperature=0, stream_usage=True,
            request_timeout=30, max_retries=1,
        )
    return _intent_model


def classify_intent(state: RAGState) -> RAGState:
    """查询意图分类路由。"""
    question = state["question"]
    intent = "factual"  # default

    if INTENT_ROUTER_ENABLED:
        model = _get_intent_model()
        if model:
            try:
                resp = (model.invoke(INTENT_PROMPT.format(question=question)).content or "").strip().lower()
                for i in ["factual", "conceptual", "comparative", "out_of_scope"]:
                    if i in resp:
                        intent = i
                        break
            except Exception:
                intent = "factual"

    _emit_rag_step("🧭", f"查询意图: {intent}")

    rag_trace = state.get("rag_trace", {}) or {}
    rag_trace["intent"] = intent
    return {
        "rag_trace": rag_trace,
        "complexity": intent if intent == "comparative" else "simple",
        "query": question,
    }


def _route_by_intent(state: RAGState) -> str:
    intent = (state.get("rag_trace") or {}).get("intent", "factual")
    if intent == "out_of_scope":
        return "refuse_answer"
    if intent == "comparative":
        return "decompose_question"
    return "retrieve_initial"


def refuse_answer(state: RAGState) -> RAGState:
    """知识库外问题直接拒答。"""
    _emit_rag_step("🚫", "知识库外问题，建议用户重新提问")
    return {
        "docs": [],
        "context": "",
        "rag_trace": {
            "tool_used": False,
            "tool_name": "search_knowledge_base",
            "retrieved_chunks": [],
            "intent": "out_of_scope",
            "refused": True,
        },
    }


def _format_docs(docs: List[dict]) -> str:
    if not docs:
        return ""
    chunks = []
    for i, doc in enumerate(docs, 1):
        source = doc.get("filename", "Unknown")
        page = doc.get("page_number", "N/A")
        text = doc.get("text", "")
        chunks.append(f"[{i}] {source} (Page {page}):\n{text}")
    return "\n\n---\n\n".join(chunks)


def retrieve_initial(state: RAGState) -> RAGState:
    t0 = _time_module.time()
    query = state["question"]
    query_image_path = state.get("query_image_path")
    top_k = int(os.getenv("RETRIEVAL_TOP_K", "8"))
    final_k = int(os.getenv("FINAL_TOP_K", "3"))  # 最终送入 LLM 的 chunk 数
    retrieve_meta: dict = {}
    timings: dict = {}

    if query_image_path:
        _emit_rag_step("🖼", "图片查询检索中...", "已接收用户上传图片")
        retrieved = retrieve_documents(query, top_k=top_k, query_image_path=query_image_path)
        results = retrieved.get("docs", [])
        retrieve_meta = retrieved.get("meta", {})
        if not results:
            _emit_rag_step("⚠️", "图片检索无结果，降级纯文本检索", "")
            retrieved = retrieve_documents(query, top_k=top_k, use_image=False)
            results = retrieved.get("docs", [])
            retrieve_meta = retrieved.get("meta", {})
            retrieve_meta["retrieval_mode"] = "image_fallback_text"
    elif RAG_FUSION_ENABLED and RAG_FUSION_VARIANTS > 1:
        _emit_rag_step("🔍", "RAG-Fusion 检索中...", f"查询: {query[:50]}")
        results, retrieve_meta = _fusion_retrieve(query, top_k)
    else:
        _emit_rag_step("🔍", "正在检索知识库...", f"查询: {query[:50]}")
        if CRAG_ENABLED:
            results, retrieve_meta = crag_retrieve(query, top_k)
        else:
            retrieved = retrieve_documents(query, top_k=top_k)
            results = retrieved.get("docs", [])
            retrieve_meta = retrieved.get("meta", {})
    # Rerank 后只取 top-N 送入生成，减少噪声
    if len(results) > final_k:
        results = results[:final_k]
    context = _format_docs(results)
    _emit_rag_step(
        "🧱",
        "三级分块检索",
        (
            f"叶子层 L{retrieve_meta.get('leaf_retrieve_level', 3)} 召回，"
            f"候选 {retrieve_meta.get('candidate_k', 0)}"
        ),
    )
    _emit_rag_step(
        "🧩",
        "Auto-merging 合并",
        (
            f"启用: {bool(retrieve_meta.get('auto_merge_enabled'))}，"
            f"应用: {bool(retrieve_meta.get('auto_merge_applied'))}，"
            f"替换片段: {retrieve_meta.get('auto_merge_replaced_chunks', 0)}"
        ),
    )
    # 盲区检测 — Conformal 校准（如有）或固定阈值
    scores = [d.get("score", 0) or 0 for d in results]
    max_score = max(scores) if scores else 0
    blindspot_threshold = float(os.getenv("BLINDSPOT_MIN_SCORE", "0.3"))
    retrieval_confidence = None

    # 尝试 Conformal 校准
    try:
        from backend.rag.conformal_retrieval import load_calibrator
        calibrator = load_calibrator()
        if calibrator.calibrated:
            conf_result = calibrator.predict_confidence(max_score)
            retrieval_confidence = conf_result["confidence"]
            blindspot_threshold = conf_result["threshold"]
    except Exception:
        pass

    is_blindspot = len(results) == 0 or max_score < blindspot_threshold

    if is_blindspot:
        conf_str = f", 置信度: {retrieval_confidence:.2f}" if retrieval_confidence else ""
        _emit_rag_step("🚫", "知识盲区", f"最高相关分 {max_score:.3f} < 阈值 {blindspot_threshold:.4f}{conf_str}")
        from backend.infra.event_logger import log_blindspot
        log_blindspot(query, max_score)
    else:
        conf_str = f", 置信度: {retrieval_confidence:.2f}" if retrieval_confidence else ""
        _emit_rag_step("✅", f"检索完成，找到 {len(results)} 个片段",
                       f"模式: {retrieve_meta.get('retrieval_mode', 'hybrid')}, 最高分 {max_score:.3f}{conf_str}")
    if not results:
        _emit_rag_step("⚠️", "无可用片段，跳过评估并强制 step-back 扩展检索")
    # RSE: 提取相关句子
    if RSE_ENABLED and results:
        results = extract_relevant_segments(query, results)
        _emit_rag_step("✂️", "RSE 提取完成", f"{len(results)} 个片段去噪")
    # Compression: 压缩上下文
    compressed_ctx = ""
    if COMPRESSION_ENABLED and results:
        compressed_ctx = compress_context(query, results)
        _emit_rag_step("📦", "上下文压缩完成", f"原 {sum(len(d.get('text','')) for d in results)} 字符 → {len(compressed_ctx)} 字符")

    t_total = _time_module.time() - t0
    timings["retrieve_initial"] = round(t_total, 3)
    _emit_rag_step("⏱", f"检索耗时 {t_total:.2f}s")

    rag_trace = {
        "tool_used": True,
        "tool_name": "search_knowledge_base",
        "query": query,
        "expanded_query": query,
        "retrieved_chunks": results,
        "initial_retrieved_chunks": results,
        "retrieval_stage": "initial",
        "is_blindspot": is_blindspot,
        "blindspot_max_score": max_score,
        "blindspot_threshold": blindspot_threshold,
        "retrieval_confidence": retrieval_confidence,
        "timings": timings,
        "compressed_context": compressed_ctx if compressed_ctx else None,
        "rse_applied": RSE_ENABLED,
        "compression_applied": COMPRESSION_ENABLED,
        **retrieval_trace_fields(retrieve_meta),
    }
    return {
        "query": query,
        "docs": results,
        "context": context,
        "rag_trace": rag_trace,
    }


def _route_after_initial(state: RAGState) -> Literal["grade_documents", "rewrite_question"]:
    if not state.get("docs"):
        return "rewrite_question"
    # HyDE 门槛：检索最高分低于阈值时强制重写
    hyde_threshold = float(os.getenv("HYDE_TRIGGER_SCORE", "0.4"))
    docs = state.get("docs", [])
    if docs:
        max_score = max((d.get("score", 0) or 0) for d in docs)
        if max_score < hyde_threshold:
            return "rewrite_question"
    return "grade_documents"


def grade_documents_node(state: RAGState) -> RAGState:
    grader = _get_grader_model()
    _emit_rag_step("📊", "正在评估文档相关性...")
    if not grader:
        grade_update = {
            "grade_score": "unknown",
            "grade_route": "rewrite_question",
            "rewrite_needed": True,
        }
        rag_trace = state.get("rag_trace", {}) or {}
        rag_trace.update(grade_update)
        return {"route": "rewrite_question", "rag_trace": rag_trace}
    question = state["question"]
    context = state.get("context", "")
    prompt = GRADE_PROMPT.format(question=question, context=context)
    try:
        response = grader.invoke([{"role": "user", "content": prompt}])
        score = _parse_grade_score(getattr(response, "content", str(response)))
        grade_error = ""
    except Exception as e:
        score = "unknown"
        grade_error = str(e)
    route = "generate_answer" if score == "yes" else "rewrite_question"
    if score == "unknown" and state.get("docs"):
        route = "generate_answer"
    if score == "unknown":
        _emit_rag_step("⚠️", "相关性评分不可用，使用检索结果继续", "评分: unknown")
    elif route == "generate_answer":
        _emit_rag_step("✅", "文档相关性评估通过", f"评分: {score}")
    else:
        _emit_rag_step("⚠️", "文档相关性不足，将重写查询", f"评分: {score}")
    grade_update = {
        "grade_score": score,
        "grade_route": route,
        "rewrite_needed": route == "rewrite_question",
    }
    if grade_error:
        grade_update["grade_error"] = grade_error[:200]
    rag_trace = state.get("rag_trace", {}) or {}
    rag_trace.update(grade_update)
    return {"route": route, "rag_trace": rag_trace}


def rewrite_question_node(state: RAGState) -> RAGState:
    question = state["question"]
    force_step_back = not state.get("docs")
    _emit_rag_step("✏️", "正在重写查询...")

    if force_step_back:
        strategy = "step_back"
    else:
        router = _get_router_model()
        strategy = "step_back"
        if router:
            prompt = (
                "请根据用户问题选择最合适的查询扩展策略，仅输出策略名。\n"
                "- step_back：包含具体名称、日期、代码等细节，需要先理解通用概念的问题。\n"
                "- hyde：模糊、概念性、需要解释或定义的问题。\n"
                "- complex：多步骤、需要分解或综合多种信息的复杂问题。\n"
                f"用户问题：{question}"
            )
            try:
                decision = router.with_structured_output(RewriteStrategy).invoke(
                    [{"role": "user", "content": prompt}]
                )
                strategy = decision.strategy
            except Exception:
                strategy = "step_back"

    expanded_query = question
    step_back_question = ""
    step_back_answer = ""
    hypothetical_doc = ""

    if strategy in ("step_back", "complex"):
        _emit_rag_step("🧠", f"使用策略: {strategy}", "生成退步问题")
        step_back = step_back_expand(question)
        step_back_question = step_back.get("step_back_question", "")
        step_back_answer = step_back.get("step_back_answer", "")
        expanded_query = step_back.get("expanded_query", question)

    if not force_step_back and strategy in ("hyde", "complex"):
        _emit_rag_step("📝", "HyDE 假设性文档生成中...")
        hypothetical_doc = generate_hypothetical_document(question)

    rag_trace = state.get("rag_trace", {}) or {}
    rag_trace.update({
        "rewrite_strategy": strategy,
        "rewrite_query": expanded_query,
        "grade_skipped": force_step_back,
    })

    return {
        "expansion_type": strategy,
        "expanded_query": expanded_query,
        "step_back_question": step_back_question,
        "step_back_answer": step_back_answer,
        "hypothetical_doc": hypothetical_doc,
        "rag_trace": rag_trace,
    }


def retrieve_expanded(state: RAGState) -> RAGState:
    strategy = state.get("expansion_type") or "step_back"
    _emit_rag_step("🔄", "使用扩展查询重新检索...", f"策略: {strategy}")
    results: List[dict] = []
    rerank_errors = []
    retrieval_trace: dict = {}
    _top_k = int(os.getenv("RETRIEVAL_TOP_K", "8"))
    query_image_path = state.get("query_image_path")

    if strategy in ("hyde", "complex"):
        hypothetical_doc = state.get("hypothetical_doc") or generate_hypothetical_document(state["question"])
        retrieved_hyde = retrieve_documents(hypothetical_doc, top_k=_top_k, query_image_path=query_image_path)
        results.extend(retrieved_hyde.get("docs", []))
        hyde_meta = retrieved_hyde.get("meta", {})
        _emit_rag_step(
            "🧱",
            "HyDE 三级检索",
            (
                f"L{hyde_meta.get('leaf_retrieve_level', 3)} 召回，"
                f"候选 {hyde_meta.get('candidate_k', 0)}，"
                f"合并替换 {hyde_meta.get('auto_merge_replaced_chunks', 0)}"
            ),
        )
        if hyde_meta.get("rerank_error"):
            rerank_errors.append(f"hyde:{hyde_meta.get('rerank_error')}")
        retrieval_trace = merge_retrieval_trace(retrieval_trace, hyde_meta)

    if strategy in ("step_back", "complex"):
        expanded_query = state.get("expanded_query") or state["question"]
        retrieved_stepback = retrieve_documents(expanded_query, top_k=_top_k, query_image_path=query_image_path)
        results.extend(retrieved_stepback.get("docs", []))
        step_meta = retrieved_stepback.get("meta", {})
        _emit_rag_step(
            "🧱",
            "Step-back 三级检索",
            (
                f"L{step_meta.get('leaf_retrieve_level', 3)} 召回，"
                f"候选 {step_meta.get('candidate_k', 0)}，"
                f"合并替换 {step_meta.get('auto_merge_replaced_chunks', 0)}"
            ),
        )
        if step_meta.get("rerank_error"):
            rerank_errors.append(f"step_back:{step_meta.get('rerank_error')}")
        retrieval_trace = merge_retrieval_trace(retrieval_trace, step_meta)

    deduped = dedupe_documents(results)

    # 扩展阶段可能合并了多路召回（如 hyde + step_back），
    # 这里统一重排展示名次，避免出现 1,2,3,4,5,4,5 这类重复名次。
    for idx, item in enumerate(deduped, 1):
        item["rrf_rank"] = idx

    context = _format_docs(deduped)
    _emit_rag_step("✅", f"扩展检索完成，共 {len(deduped)} 个片段")
    rag_trace = state.get("rag_trace", {}) or {}
    rag_trace.update({
        "expanded_query": state.get("expanded_query") or state["question"],
        "step_back_question": state.get("step_back_question", ""),
        "step_back_answer": state.get("step_back_answer", ""),
        "hypothetical_doc": state.get("hypothetical_doc", ""),
        "expansion_type": strategy,
        "retrieved_chunks": deduped,
        "expanded_retrieved_chunks": deduped,
        "retrieval_stage": "expanded",
        "rerank_error": "; ".join(rerank_errors) if rerank_errors else retrieval_trace.get("rerank_error"),
        **retrieval_trace,
    })
    return {"docs": deduped, "context": context, "rag_trace": rag_trace}


# ---------------------------------------------------------------------------
# 复杂度分类 & 子问题分解
# ---------------------------------------------------------------------------

COMPLEXITY_PROMPT = (
    "你是一个问题复杂度分类器。请判断用户问题的复杂度。\n\n"
    "【简单问题】：事实查询、定义查询、单一信息点查询、明确的 yes/no 问题、"
    "某个具体属性/参数/规格的查询。\n"
    "【复杂问题】：需要跨文档综合、多角度分析、比较对比、多步骤推理、"
    "需要综合多个信息源才能完整回答的问题。\n\n"
    "用户问题：{question}\n\n"
    "请输出分类结果。"
)

COVERAGE_PROMPT = (
    "你是一个问题分解质量审核员。评估给定的子问题列表是否充分覆盖了原问题的所有核心维度。\n\n"
    "原问题：{question}\n\n"
    "子问题列表：\n{sub_questions}\n\n"
    "判断：子问题是否从不同角度、不同维度覆盖了原问题？是否遗漏了重要方面？\n"
    "如果覆盖充分，输出 adequate；如果有明显缺失，输出 insufficient 并说明缺少什么。"
)

CONFLICT_PROMPT = (
    "你是一个信息一致性审核员。检查以下检索到的文档片段之间是否存在事实冲突。\n\n"
    "文档片段：\n{documents}\n\n"
    "判断标准：如果两个片段对同一事实给出明显矛盾的陈述（如数字不同、结论相反），标记为冲突。\n"
    "如果冲突存在，输出 has_conflict=true 并描述冲突内容。如果无冲突，has_conflict=false。"
)

HALLUCINATION_PROMPT = (
    "你是一个答案准确性审核员。逐句检查 AI 回答中的每个事实声明，是否被提供的文档片段支撑。\n\n"
    "文档片段：\n{documents}\n\n"
    "AI 回答：\n{answer}\n\n"
    "输出：如果所有事实声明都能在文档中找到依据，输出 supported。"
    "如果有声明无法在文档中验证，输出 partially_unsupported 并列出未被支撑的声明。\n"
    "注意：通用知识、推理过程不需要逐字匹配；只检查具体的数字、名称、事件等事实性声明。"
)

DECOMPOSE_PROMPT = (
    "请将以下复杂问题分解为 2-4 个独立的子问题。\n"
    "每个子问题应聚焦于原问题的一个明确方面，能独立通过知识库检索获得答案。\n"
    "子问题之间应覆盖原问题的核心维度，避免重叠。\n\n"
    "原问题：{question}\n\n"
    "请输出子问题列表。"
)


def classify_complexity(state: RAGState) -> RAGState:
    """使用 FAST_MODEL 判断问题复杂度。"""
    question = state["question"]
    _emit_rag_step("🧭", "正在分析问题复杂度...")

    model = _get_complexity_model()
    if not model:
        _emit_rag_step("⚠️", "复杂度模型不可用，默认简单问题")
        return {"complexity": "simple", "complexity_reason": "model_unavailable"}

    prompt = COMPLEXITY_PROMPT.format(question=question)
    try:
        result = model.with_structured_output(ComplexityResult).invoke(
            [{"role": "user", "content": prompt}]
        )
        complexity = (result.complexity or "simple").strip().lower()
        reason = (result.reason or "").strip()
        if complexity not in ("simple", "complex"):
            complexity = "simple"
    except Exception:
        complexity = "simple"
        reason = "classification_error"

    if complexity == "simple":
        _emit_rag_step("✅", "简单问题 → 走标准 RAG 流程", f"理由: {reason[:60]}")
    else:
        _emit_rag_step("🔀", "复杂问题 → 将分解为子问题并行检索", f"理由: {reason[:60]}")

    return {"complexity": complexity, "complexity_reason": reason}


def decompose_question(state: RAGState) -> RAGState:
    """将复杂问题分解为 2-4 个独立子问题。"""
    question = state["question"]
    _emit_rag_step("🧩", "正在分解复杂问题...")

    model = _get_complexity_model()
    if not model:
        _emit_rag_step("⚠️", "分解模型不可用，使用原始问题")
        return {"sub_questions": [question]}

    prompt = DECOMPOSE_PROMPT.format(question=question)
    try:
        result = model.with_structured_output(SubQuestions).invoke(
            [{"role": "user", "content": prompt}]
        )
        sub_qs = [sq.strip() for sq in (result.sub_questions or []) if sq.strip()]
        if not sub_qs:
            sub_qs = [question]
    except Exception:
        sub_qs = [question]

    for i, sq in enumerate(sub_qs, 1):
        _emit_rag_step("📌", f"子问题 {i}", sq[:80])

    return {"sub_questions": sub_qs}


def validate_sub_questions(state: RAGState) -> RAGState:
    """验证子问题覆盖度，不足则尝试补充。"""
    question = state["question"]
    sub_qs = state.get("sub_questions") or [question]
    _emit_rag_step("🔬", "正在验证子问题覆盖度...")

    model = _get_complexity_model()
    if not model or len(sub_qs) <= 1:
        _emit_rag_step("✅", "覆盖度验证跳过", "模型不可用或仅1个子问题")
        return {}

    prompt = COVERAGE_PROMPT.format(
        question=question,
        sub_questions="\n".join(f"- {sq}" for sq in sub_qs),
    )
    try:
        result = model.with_structured_output(CoverageResult).invoke(
            [{"role": "user", "content": prompt}]
        )
        if result.coverage == "insufficient" and result.suggestion:
            suggestion = result.suggestion.strip()
            if suggestion and len(sub_qs) < 4:
                sub_qs = list(sub_qs) + [suggestion]
                _emit_rag_step("📌", f"补充子问题 {len(sub_qs)}", suggestion[:80])
                return {"sub_questions": sub_qs}
        _emit_rag_step("✅", "子问题覆盖度验证通过")
    except Exception:
        _emit_rag_step("⚠️", "覆盖度验证失败，使用原始分解")
    return {}


def _route_after_complexity(state: RAGState):
    """复杂度路由：simple 走原有流程，complex 先分解再并行检索。"""
    if state.get("complexity") == "complex":
        return "decompose_question"
    return "retrieve_initial"


def _fanout_sub_questions(state: RAGState):
    """将分解后的子问题通过 Send API 并行分发到 rag_sub_agent 子图。"""
    sub_qs = state.get("sub_questions") or []
    if not sub_qs:
        # 分解失败，回退到原有流程
        return [Send("retrieve_initial", {
            "question": state["question"],
            "query_image_path": state.get("query_image_path"),
            "query": state["question"],
            "context": "",
            "docs": [],
            "route": None,
            "expansion_type": None,
            "expanded_query": None,
            "step_back_question": None,
            "step_back_answer": None,
            "hypothetical_doc": None,
            "rag_trace": None,
            "complexity": None,
            "complexity_reason": None,
            "sub_questions": None,
            "is_sub_agent": False,
            "sub_results": [],
        })]
    return [
        Send("rag_sub_agent", {
            "question": sq,
            "query_image_path": state.get("query_image_path"),
            "query": sq,
            "context": "",
            "docs": [],
            "route": None,
            "expansion_type": None,
            "expanded_query": None,
            "step_back_question": None,
            "step_back_answer": None,
            "hypothetical_doc": None,
            "rag_trace": None,
            "complexity": None,
            "complexity_reason": None,
            "sub_questions": None,
            "is_sub_agent": True,
            "sub_results": [],
        })
        for sq in sub_qs
    ]


def synthesis(state: RAGState) -> RAGState:
    """合并所有子 Agent 检索到的文档，去重排序后输出最终上下文。"""
    sub_results = state.get("sub_results", [])
    _emit_rag_step("🔬", f"正在合成 {len(sub_results)} 个子问题的检索结果...")

    all_docs: List[dict] = []
    for result in sub_results:
        docs = result.get("docs", [])
        all_docs.extend(docs)

    deduped = dedupe_documents(all_docs)
    for idx, item in enumerate(deduped, 1):
        item["rrf_rank"] = idx

    context = _format_docs(deduped)
    # 多文档冲突检测
    context_text = _format_docs(deduped)
    conflict_info = _detect_conflicts_sync(context_text)

    _emit_rag_step("✅", f"合成完成，共 {len(deduped)} 个去重片段")
    if conflict_info.get("has_conflict"):
        _emit_rag_step("⚠️", "检测到文档间冲突", conflict_info.get("conflict_detail", "")[:80])

    # 合并所有子 Agent 的 rag_trace
    sub_traces = []
    for result in sub_results:
        trace = result.get("rag_trace")
        if trace:
            sub_traces.append(trace)

    original_trace = state.get("rag_trace") or {}
    rag_trace = {
        **original_trace,
        "tool_used": True,
        "tool_name": "search_knowledge_base",
        "query": state["question"],
        "query_image_path": state.get("query_image_path"),
        "expanded_query": state["question"],
        "retrieved_chunks": deduped,
        "retrieval_stage": "synthesis",
        "complexity": "complex",
        "complexity_reason": state.get("complexity_reason", ""),
        "sub_questions": state.get("sub_questions", []),
        "sub_agent_count": len(sub_results),
        "synthesis_merged_count": len(all_docs),
        "sub_traces": sub_traces,
        "conflict_info": conflict_info,
    }

    return {"docs": deduped, "context": context, "rag_trace": rag_trace}


# ---------------------------------------------------------------------------
# 子 Agent RAG 子图（每个子问题独立运行完整 RAG 流程）
# ---------------------------------------------------------------------------

def build_rag_sub_agent_graph():
    """构建子 Agent RAG 子图：retrieve → grade → rewrite → retrieve_expanded。"""
    sub_graph = StateGraph(RAGState)
    sub_graph.add_node("retrieve_initial", retrieve_initial)
    sub_graph.add_node("grade_documents", grade_documents_node)
    sub_graph.add_node("rewrite_question", rewrite_question_node)
    sub_graph.add_node("retrieve_expanded", retrieve_expanded)

    sub_graph.set_entry_point("retrieve_initial")
    sub_graph.add_conditional_edges(
        "retrieve_initial",
        _route_after_initial,
        {
            "grade_documents": "grade_documents",
            "rewrite_question": "rewrite_question",
        },
    )
    sub_graph.add_conditional_edges(
        "grade_documents",
        lambda state: state.get("route"),
        {
            "generate_answer": END,
            "rewrite_question": "rewrite_question",
        },
    )
    sub_graph.add_edge("rewrite_question", "retrieve_expanded")
    sub_graph.add_edge("retrieve_expanded", END)
    return sub_graph.compile()


# 子 Agent 子图实例（模块级单例）
_rag_sub_agent_graph = build_rag_sub_agent_graph()


def rag_sub_agent(state: RAGState) -> RAGState:
    """包装子图，将子图结果封装为 sub_results 以便主图通过 reducer 合并。"""
    question = state.get("question", "")
    # 短标签：截取前 40 字符作为前端分组标识
    short_label = question[:40] + "..." if len(question) > 40 else question
    _set_sub_agent_group(short_label)
    try:
        result = _rag_sub_agent_graph.invoke(state)
    finally:
        _clear_sub_agent_group()
    return {
        "sub_results": [{
            "question": question,
            "docs": result.get("docs", []),
            "rag_trace": result.get("rag_trace"),
        }],
    }


# ---------------------------------------------------------------------------
# 主 RAG 图
# ---------------------------------------------------------------------------

def _detect_conflicts_sync(docs_text: str) -> dict:
    """检测检索文档间是否存在事实冲突（使用 AUX_MODEL 更强推理）。"""
    if not docs_text or len(docs_text) < 100:
        return {"has_conflict": False, "conflict_detail": ""}
    model = _get_aux_model()
    if not model:
        return {"has_conflict": False, "conflict_detail": ""}
    try:
        prompt = CONFLICT_PROMPT.format(documents=docs_text[:4000])
        result = model.with_structured_output(ConflictResult).invoke(
            [{"role": "user", "content": prompt}]
        )
        return {"has_conflict": result.has_conflict, "conflict_detail": result.conflict_detail}
    except Exception:
        return {"has_conflict": False, "conflict_detail": ""}


def verify_answer_against_docs(answer: str, docs_text: str) -> dict:
    """验证 AI 回答是否被检索文档支撑（使用 AUX_MODEL 更强推理）。"""
    if not answer or not docs_text:
        return {"verdict": "supported", "unsupported_claims": ""}
    model = _get_aux_model()
    if not model:
        return {"verdict": "supported", "unsupported_claims": ""}
    try:
        prompt = HALLUCINATION_PROMPT.format(documents=docs_text[:4000], answer=answer[:2000])
        result = model.with_structured_output(HallucinationCheck).invoke(
            [{"role": "user", "content": prompt}]
        )
        return {
            "verdict": result.verdict,
            "unsupported_claims": result.unsupported_claims,
        }
    except Exception:
        return {"verdict": "supported", "unsupported_claims": ""}


def build_rag_graph():
    graph = StateGraph(RAGState)

    # 节点注册
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("classify_complexity", classify_complexity)
    graph.add_node("decompose_question", decompose_question)
    graph.add_node("validate_sub_questions", validate_sub_questions)
    graph.add_node("retrieve_initial", retrieve_initial)
    graph.add_node("refuse_answer", refuse_answer)
    graph.add_node("grade_documents", grade_documents_node)
    graph.add_node("rewrite_question", rewrite_question_node)
    graph.add_node("retrieve_expanded", retrieve_expanded)
    graph.add_node("rag_sub_agent", rag_sub_agent)
    graph.add_node("synthesis", synthesis)

    # 入口：意图分类 → 路由
    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        _route_by_intent,
        {
            "retrieve_initial": "retrieve_initial",
            "decompose_question": "decompose_question",
            "refuse_answer": "refuse_answer",
        },
    )
    graph.add_edge("refuse_answer", END)

    # classify_complexity 节点保留（后续可用于分析统计），但路由已由 classify_intent 接管

    # 分解 → 验证覆盖度 → 通过 Send API 并行分发到 rag_sub_agent
    graph.add_edge("decompose_question", "validate_sub_questions")
    graph.add_conditional_edges("validate_sub_questions", _fanout_sub_questions)

    # 原有简单路径
    graph.add_conditional_edges(
        "retrieve_initial",
        _route_after_initial,
        {
            "grade_documents": "grade_documents",
            "rewrite_question": "rewrite_question",
        },
    )
    graph.add_conditional_edges(
        "grade_documents",
        lambda state: state.get("route"),
        {
            "generate_answer": END,
            "rewrite_question": "rewrite_question",
        },
    )
    graph.add_edge("rewrite_question", "retrieve_expanded")
    graph.add_edge("retrieve_expanded", END)

    # 并行子 Agent → 合成
    graph.add_edge("rag_sub_agent", "synthesis")
    graph.add_edge("synthesis", END)

    return graph.compile()


rag_graph = build_rag_graph()


def run_rag_graph(question: str, query_image_path: str | None = None) -> dict:
    return rag_graph.invoke({
        "question": question,
        "query_image_path": query_image_path,
        "query": question,
        "context": "",
        "docs": [],
        "route": None,
        "expansion_type": None,
        "expanded_query": None,
        "step_back_question": None,
        "step_back_answer": None,
        "hypothetical_doc": None,
        "rag_trace": None,
        # 复杂度路由新增字段
        "complexity": None,
        "complexity_reason": None,
        "sub_questions": None,
        "is_sub_agent": False,
        "sub_results": [],
    })
