"""
RAG 评估核心指标

参考：
- RAGAS (https://github.com/confident-ai/deepeval): Faithfulness / Answer Relevancy / Context Precision / Context Recall
- deepeval (https://github.com/confident-ai/deepeval): G-Eval / contextual metrics / LLM-as-Judge
- 经典 IR 指标: Recall@k / Precision@k / MRR / NDCG@k / Hit Rate
"""

import math
from typing import List, Dict, Any, Optional


# ─── Gold Chunk 指标（需要标注数据）───────────────────────────────────────────

def recall_at_k(relevant_ids: set[str], retrieved_ids: list[str], k: int = 5) -> float:
    """R@k: 前 k 个检索结果中命中相关 chunk 的比例。"""
    if not relevant_ids:
        return 1.0
    top_k = set(retrieved_ids[:k])
    return len(relevant_ids & top_k) / len(relevant_ids)


def precision_at_k(relevant_ids: set[str], retrieved_ids: list[str], k: int = 5) -> float:
    """P@k: 前 k 个检索结果中相关 chunk 的占比。"""
    if k == 0:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return len(relevant_ids & top_k) / k


def mrr(relevant_ids: set[str], retrieved_ids: list[str]) -> float:
    """MRR (Mean Reciprocal Rank): 第一个相关结果的倒数排名。"""
    for i, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_ids:
            return 1.0 / i
    return 0.0


def ndcg_at_k(relevant_ids: set[str], retrieved_ids: list[str], k: int = 5) -> float:
    """NDCG@k: 归一化折损累积增益（binary relevance）。"""
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k], 1):
        if doc_id in relevant_ids:
            dcg += 1.0 / math.log2(i + 1)

    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    if idcg == 0:
        return 1.0
    return dcg / idcg


def hit_rate_at_k(relevant_ids: set[str], retrieved_ids: list[str], k: int = 5) -> float:
    """HR@k: 前 k 结果中是否至少有一个相关。"""
    return 1.0 if set(retrieved_ids[:k]) & relevant_ids else 0.0


def hard_negative_precision_at_k(
    negative_ids: set[str], retrieved_ids: list[str], k: int = 5
) -> float:
    """HN-P@k: 前 k 个结果中包含 hard negative 的比例（越低越好）。"""
    if k == 0 or not negative_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return len(negative_ids & top_k) / k


def hard_negative_hit_rate_at_k(
    negative_ids: set[str], retrieved_ids: list[str], k: int = 5
) -> float:
    """HN-HR@k: hard negative 进入前 k 的概率（越低越好）。"""
    if not negative_ids:
        return 0.0
    return 1.0 if set(retrieved_ids[:k]) & negative_ids else 0.0


def ndcg_with_hard_negatives(
    relevant_ids: set[str],
    negative_ids: set[str],
    retrieved_ids: list[str],
    k: int = 5,
) -> float:
    """NDCG@k 扩展：正样本 +1 分，hard negative -1 分，普通无关 0 分。"""
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k], 1):
        if doc_id in relevant_ids:
            dcg += 1.0 / math.log2(i + 1)
        elif doc_id in negative_ids:
            dcg -= 1.0 / math.log2(i + 1)  # hard negative 惩罚

    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    if idcg == 0:
        return 1.0 if dcg >= 0 else 0.0
    return max(0.0, dcg / idcg)  # 下限 0


def answer_grounding_recall(
    span_chunk_ids: set[str], retrieved_ids: list[str], k: int = 10
) -> float:
    """Answer Grounding Recall: answer_span 中引用的 chunk 被检索到的比例。"""
    if not span_chunk_ids:
        return 1.0
    top_k = set(retrieved_ids[:k])
    return len(span_chunk_ids & top_k) / len(span_chunk_ids)


def span_recall_at_k(
    span_chunk_ids: set[str], retrieved_ids: list[str], k: int = 10
) -> float:
    """Span Recall@k: 同 answer_grounding_recall，显式指定 k。"""
    return answer_grounding_recall(span_chunk_ids, retrieved_ids, k)


def compute_gold_metrics_batched(
    gold_data: List[Dict[str, Any]],
    ks: tuple = (1, 3, 5, 10),
    dense_embeddings: Optional[List[List[float]]] = None,
    sparse_embeddings: Optional[List[dict]] = None,
) -> Dict[str, Any]:
    """
    compute_gold_metrics 的批量优化版：预计算所有 embedding，避免逐条调用。

    用法:
        questions = [item["question"] for item in gold_data]
        dense_all = embedding_service.get_embeddings(questions)  # 一次批量
        sparse_all = embedding_service.get_sparse_embeddings(questions)
        report = compute_gold_metrics_batched(gold_data, ks=(3,5,10),
                                              dense_embeddings=dense_all,
                                              sparse_embeddings=sparse_all)
    """
    from backend.indexing.milvus_client import get_milvus_store

    ms = get_milvus_store()
    filter_expr = "chunk_level == 3"

    if dense_embeddings is None:
        questions = [item["question"] for item in gold_data]
        from backend.indexing.embedding import embedding_service
        dense_embeddings = embedding_service.get_embeddings(questions)

    if sparse_embeddings is None:
        questions = [item["question"] for item in gold_data]
        from backend.indexing.embedding import embedding_service
        sparse_embeddings = embedding_service.get_sparse_embeddings(questions)

    max_k = max(ks)
    max_candidate_k = max(max_k * 3, max_k)

    has_negatives = any(item.get("negative_chunk_ids") for item in gold_data)
    has_span = any(item.get("answer_span") for item in gold_data)

    accum: Dict[str, list] = {}
    for k in ks:
        for prefix in ["recall", "precision", "mrr", "ndcg", "hit"]:
            accum[f"{prefix}@{k}"] = []
    if has_negatives:
        for k in ks:
            for prefix in ["hn_precision", "hn_hit", "ndcg_hn"]:
                accum[f"{prefix}@{k}"] = []
    if has_span:
        for k in ks:
            accum[f"grounding@{k}"] = []

    for i, item in enumerate(gold_data):
        relevant = set(item.get("relevant_chunk_ids", []))
        negatives = set(item.get("negative_chunk_ids", []))
        spans = item.get("answer_span", [])

        dense = dense_embeddings[i]
        sparse = sparse_embeddings[i]

        # Dense retrieve at max k
        try:
            dense_results = ms.dense_retrieve(dense, top_k=max_candidate_k, filter_expr=filter_expr)
        except Exception:
            dense_results = []
        dense_ids = [r.get("chunk_id", "") for r in dense_results]

        # Hybrid retrieve at max k
        try:
            hybrid_results = ms.hybrid_retrieve(dense, sparse, top_k=max_candidate_k, filter_expr=filter_expr)
        except Exception:
            hybrid_results = []
        hybrid_ids = [r.get("chunk_id", "") for r in hybrid_results]

        for k in ks:
            # Dense metrics
            ids_k = dense_ids[:k]
            if relevant:
                accum[f"recall@{k}"].append(recall_at_k(relevant, ids_k, k))
                accum[f"precision@{k}"].append(precision_at_k(relevant, ids_k, k))
                accum[f"mrr@{k}"].append(mrr(relevant, ids_k))
                accum[f"ndcg@{k}"].append(ndcg_at_k(relevant, ids_k, k))
                accum[f"hit@{k}"].append(hit_rate_at_k(relevant, ids_k, k))
            if negatives:
                accum[f"hn_precision@{k}"].append(hard_negative_precision_at_k(negatives, ids_k, k))
                accum[f"hn_hit@{k}"].append(hard_negative_hit_rate_at_k(negatives, ids_k, k))
                accum[f"ndcg_hn@{k}"].append(ndcg_with_hard_negatives(relevant, negatives, ids_k, k))
            if spans:
                span_ids = set(s.get("chunk_id", "") for s in spans if s.get("chunk_id"))
                if span_ids:
                    accum[f"grounding@{k}"].append(answer_grounding_recall(span_ids, ids_k, k))

        # Hybrid metrics use hybrid_ids (flat hybrid, no rerank/merge)
        # For now, store as a separate mode — run separately if needed

    summary: Dict[str, Any] = {}
    for key, vals in accum.items():
        summary[key] = _stats(vals) if vals else None
    summary["num_queries"] = len(gold_data)
    summary["has_negative_annotations"] = has_negatives
    summary["has_span_annotations"] = has_span
    return summary


def run_comparison_eval_batched(
    gold_data: List[Dict[str, Any]],
    ks: tuple = (1, 3, 5, 10),
) -> Dict[str, Any]:
    """
    批量四路对比：dense / sparse / hybrid / hybrid_rerank。

    比 run_comparison_eval 快 20-50 倍。
    """
    import os as _os
    from backend.indexing.embedding import embedding_service
    from backend.indexing.milvus_client import get_milvus_store

    ms = get_milvus_store()
    filter_expr = "chunk_level == 3"
    max_k = max(ks)
    candidate_k = max(max_k * 3, max_k)

    # 检查 rerank 是否可用
    rerank_available = bool(_os.getenv("RERANK_MODEL") and _os.getenv("RERANK_API_KEY") and _os.getenv("RERANK_BINDING_HOST"))
    modes = ["dense", "sparse", "hybrid", "ubg"]
    if rerank_available:
        modes.append("hybrid_rerank")

    questions = [item["question"] for item in gold_data]
    print(f"  批量嵌入 {len(questions)} 个问题...", flush=True)
    dense_all = embedding_service.get_embeddings(questions)
    sparse_all = embedding_service.get_sparse_embeddings(questions)
    has_negatives = any(item.get("negative_chunk_ids") for item in gold_data)

    # 预加载 rerank pipeline（避免每个问题重复导入）
    if rerank_available:
        from backend.rag.utils import _rerank_documents, _auto_merge_candidates, _sort_by_rank_score

    accum: Dict[str, Dict[str, list]] = {m: {} for m in modes}
    for mode in accum:
        for k in ks:
            for prefix in ["recall", "precision", "mrr", "ndcg", "hit"]:
                accum[mode][f"{prefix}@{k}"] = []
            if has_negatives:
                for prefix in ["hn_precision", "hn_hit", "ndcg_hn"]:
                    accum[mode][f"{prefix}@{k}"] = []

    print(f"  四路检索中 ({'+'.join(modes)})...", flush=True)
    for i, item in enumerate(gold_data):
        if i % 200 == 0 and i > 0:
            print(f"    {i}/{len(gold_data)}", flush=True)

        relevant = set(item.get("relevant_chunk_ids", []))
        negatives = set(item.get("negative_chunk_ids", []))
        dense = dense_all[i]
        sparse = sparse_all[i]

        # 1) Dense only
        d_results = _safe_search(lambda: ms.dense_retrieve(dense, top_k=candidate_k, filter_expr=filter_expr))
        d_ids = [r.get("chunk_id", "") for r in d_results]

        # 2) Sparse only
        s_results = _safe_search(lambda: ms.sparse_retrieve(sparse, top_k=candidate_k, filter_expr=filter_expr))
        s_ids = [r.get("chunk_id", "") for r in s_results]

        # 3) Hybrid (dense + sparse RRF, no rerank, no merge)
        h_raw = _safe_search(lambda: ms.hybrid_retrieve(dense, sparse, top_k=candidate_k, filter_expr=filter_expr))
        h_ids = [r.get("chunk_id", "") for r in h_raw]

        # 4) UBG (dense + sparse heuristic weighted fusion, no image)
        u_ids = list(d_ids)  # fallback
        try:
            from backend.rag.ubg_fusion import get_ubg_heuristic, compute_score_stats
            heuristic = get_ubg_heuristic()
            d_scores = [r.get("score", 0.0) for r in d_results]
            s_scores = [r.get("score", 0.0) for r in s_results]
            weights = heuristic.compute_weights(dense, None, has_image=False)
            # 加权合并去重
            merged: dict[str, float] = {}
            for r in d_results:
                merged[r.get("chunk_id", "")] = r.get("score", 0.0) * weights["dense"]
            for r in s_results:
                cid = r.get("chunk_id", "")
                merged[cid] = merged.get(cid, 0.0) + r.get("score", 0.0) * weights["sparse"]
            u_ids = [cid for cid, _ in sorted(merged.items(), key=lambda x: x[1], reverse=True)]
        except Exception:
            pass

        # 5) Hybrid + Rerank + Auto-merge (full pipeline)
        hr_ids = list(h_ids)  # fallback
        if rerank_available:
            try:
                reranked, _ = _rerank_documents(query=item["question"], docs=h_raw, top_k=max_k)
                merged_docs, _ = _auto_merge_candidates(reranked)
                hr_ids = [r.get("chunk_id", "") for r in _sort_by_rank_score(merged_docs)[:max_k]]
            except Exception:
                pass

        id_map = {"dense": d_ids, "sparse": s_ids, "hybrid": h_ids, "ubg": u_ids}
        if rerank_available:
            id_map["hybrid_rerank"] = hr_ids

        for k in ks:
            for mode_name, ids in id_map.items():
                ids_k = ids[:k]
                if relevant:
                    accum[mode_name][f"recall@{k}"].append(recall_at_k(relevant, ids_k, k))
                    accum[mode_name][f"precision@{k}"].append(precision_at_k(relevant, ids_k, k))
                    accum[mode_name][f"mrr@{k}"].append(mrr(relevant, ids_k))
                    accum[mode_name][f"ndcg@{k}"].append(ndcg_at_k(relevant, ids_k, k))
                    accum[mode_name][f"hit@{k}"].append(hit_rate_at_k(relevant, ids_k, k))
                if negatives:
                    accum[mode_name][f"hn_precision@{k}"].append(hard_negative_precision_at_k(negatives, ids_k, k))
                    accum[mode_name][f"hn_hit@{k}"].append(hard_negative_hit_rate_at_k(negatives, ids_k, k))
                    accum[mode_name][f"ndcg_hn@{k}"].append(ndcg_with_hard_negatives(relevant, negatives, ids_k, k))

    report: Dict[str, Any] = {"num_queries": len(gold_data)}
    report["has_negative_annotations"] = has_negatives
    for mode in modes:
        report[mode] = {}
        for key, vals in accum[mode].items():
            report[mode][key] = _stats(vals) if vals else None

    # Auto-save markdown report
    try:
        import time as _t
        from pathlib import Path as _P
        modes = [m for m in ["dense", "sparse", "hybrid", "hybrid_rerank"] if report.get(m)]
        lines = [f"# RAG Eval", f"", f"**Time**: {_t.strftime('%Y-%m-%d %H:%M:%S')}",
                 f"**Samples**: {len(gold_data)}", ""]
        lines.append("| Metric | " + " | ".join(modes) + " |")
        lines.append("|--------|" + "|".join(["--------"] * len(modes)) + "|")
        report_metrics = ["recall@3", "recall@5", "recall@10", "precision@5", "mrr@5", "ndcg@5", "hit@5"]
        if has_negatives:
            report_metrics.extend(["hn_precision@5", "hn_hit@5", "ndcg_hn@5"])
        for m in report_metrics:
            k = int(m.split("@")[-1])
            if k not in ks: continue
            row = f"| {m} |"
            for mode in modes: row += f" {report.get(mode,{}).get(m,{}).get('mean',0):.4f} |" if report.get(mode,{}).get(m) else " N/A |"
            lines.append(row)
        _P("reports").mkdir(exist_ok=True)
        (_P("reports") / f"eval_{_t.strftime('%Y%m%d_%H%M%S')}.md").write_text("\n".join(lines), encoding="utf-8")
    except Exception: pass

    return report


def eval_with_groups(
    gold_data: List[Dict[str, Any]],
    group_by: str = "domain",
    sample_per_group: int = 500,
    ks: tuple = (5,),
):
    """
    按指定字段分组评估，输出每个分组的 recall@k。

    group_by: "domain" | "language" | "difficulty" | "query_type"
    """
    from backend.eval.metrics import run_comparison_eval_batched

    groups: Dict[str, list] = {}
    for item in gold_data:
        key = item.get(group_by, "unknown")
        groups.setdefault(key, []).append(item)

    results = {}
    for gname, items in sorted(groups.items()):
        if len(items) > sample_per_group:
            import random
            random.seed(42)
            items = random.sample(items, sample_per_group)
        report = run_comparison_eval_batched(items, ks=ks)
        results[gname] = {
            "count": len(items),
            "dense_recall@5": report.get("dense", {}).get("recall@5", {}).get("mean", 0),
            "sparse_recall@5": report.get("sparse", {}).get("recall@5", {}).get("mean", 0),
            "hybrid_recall@5": report.get("hybrid", {}).get("recall@5", {}).get("mean", 0),
        }
    return results


def _safe_search(fn):
    try:
        return fn()
    except Exception:
        return []


def compute_gold_metrics(
    gold_data: List[Dict[str, Any]],
    retrieval_fn,
    ks: tuple = (1, 3, 5, 10),
) -> Dict[str, Any]:
    """
    基于 gold chunk 标注集评估检索质量。

    gold_data 格式:
        [{"question": "...", "relevant_chunk_ids": ["id1", "id2"], ...}, ...]

    retrieval_fn: callable(question, top_k) -> List[dict]  # 每个 dict 需含 "chunk_id"
    """
    summary: Dict[str, Any] = {}
    has_negatives = any(item.get("negative_chunk_ids") for item in gold_data)
    has_span = any(item.get("answer_span") for item in gold_data)

    for k in ks:
        all_recall: list[float] = []
        all_precision: list[float] = []
        all_mrr: list[float] = []
        all_ndcg: list[float] = []
        all_hit: list[float] = []
        all_hn_precision: list[float] = []
        all_hn_hit: list[float] = []
        all_ndcg_hn: list[float] = []
        all_grounding: list[float] = []

        for item in gold_data:
            question = item["question"]
            relevant = set(item.get("relevant_chunk_ids", []))
            negatives = set(item.get("negative_chunk_ids", []))
            spans = item.get("answer_span", [])

            results = retrieval_fn(question, top_k=k)
            retrieved_ids = [r.get("chunk_id", "") for r in results]

            if relevant:
                all_recall.append(recall_at_k(relevant, retrieved_ids, k))
                all_precision.append(precision_at_k(relevant, retrieved_ids, k))
                all_mrr.append(mrr(relevant, retrieved_ids))
                all_ndcg.append(ndcg_at_k(relevant, retrieved_ids, k))
                all_hit.append(hit_rate_at_k(relevant, retrieved_ids, k))

            if negatives:
                all_hn_precision.append(hard_negative_precision_at_k(negatives, retrieved_ids, k))
                all_hn_hit.append(hard_negative_hit_rate_at_k(negatives, retrieved_ids, k))
                all_ndcg_hn.append(ndcg_with_hard_negatives(relevant, negatives, retrieved_ids, k))

            if spans:
                span_ids = set(s.get("chunk_id", "") for s in spans if s.get("chunk_id"))
                if span_ids:
                    all_grounding.append(answer_grounding_recall(span_ids, retrieved_ids, k))

        n = len(all_recall)
        summary.update({
            f"recall@{k}": _stats(all_recall) if n else None,
            f"precision@{k}": _stats(all_precision) if n else None,
            f"mrr@{k}": _stats(all_mrr) if n else None,
            f"ndcg@{k}": _stats(all_ndcg) if n else None,
            f"hit_rate@{k}": _stats(all_hit) if n else None,
        })
        if has_negatives:
            summary.update({
                f"hard_negative_precision@{k}": _stats(all_hn_precision) if all_hn_precision else None,
                f"hard_negative_hit_rate@{k}": _stats(all_hn_hit) if all_hn_hit else None,
                f"ndcg_hard_neg@{k}": _stats(all_ndcg_hn) if all_ndcg_hn else None,
            })
        if has_span:
            summary.update({
                f"answer_grounding@{k}": _stats(all_grounding) if all_grounding else None,
            })

    summary["num_queries"] = len(gold_data)
    summary["has_negative_annotations"] = has_negatives
    summary["has_span_annotations"] = has_span
    return summary


# ─── RAGAS 风格指标（无需标注，LLM-as-Judge）───────────────────────────────────

def compute_ragas_metrics(
    questions: List[str],
    answers: List[str],
    contexts: List[List[str]],
    judge_fn=None,
) -> Dict[str, Any]:
    """
    计算 RAGAS 四大指标（需要 LLM judge）。

    Args:
        questions: 提问列表
        answers: 模型回答列表
        contexts: 每个回答对应的检索上下文列表（每个元素是 [chunk_text, ...]）
        judge_fn: 可选的评估 LLM 调用函数，签名 fn(prompt) -> str

    Returns:
        {faithfulness, answer_relevancy, context_precision, context_recall}
    """
    from backend.eval.llm_judge import (
        evaluate_faithfulness,
        evaluate_answer_relevancy,
        evaluate_context_precision,
        evaluate_context_recall,
    )

    faithfulness_scores = []
    relevancy_scores = []
    precision_scores = []
    recall_scores = []

    for q, a, ctx in zip(questions, answers, contexts):
        faithfulness_scores.append(evaluate_faithfulness(a, ctx, judge_fn))
        relevancy_scores.append(evaluate_answer_relevancy(q, a, judge_fn))
        precision_scores.append(evaluate_context_precision(q, ctx, judge_fn))
        recall_scores.append(evaluate_context_recall(q, ctx, judge_fn))

    return {
        "faithfulness": _stats(faithfulness_scores),
        "answer_relevancy": _stats(relevancy_scores),
        "context_precision": _stats(precision_scores),
        "context_recall": _stats(recall_scores),
        "num_samples": len(questions),
    }


# ─── 多路对比评测 ─────────────────────────────────────────────────────────────

def run_comparison_eval(
    gold_data: List[Dict[str, Any]],
    retrievers: Dict[str, callable],
    ks: tuple = (1, 3, 5, 10),
) -> Dict[str, Any]:
    """
    在同一个 gold 标注集上对比多种检索策略。

    retrievers: {"dense_only": fn, "sparse_only": fn, "hybrid": fn, "hybrid_rerank": fn}
    """
    report: Dict[str, Any] = {"num_queries": len(gold_data)}
    for name, retriever_fn in retrievers.items():
        report[name] = compute_gold_metrics(gold_data, retriever_fn, ks=ks)
    return report


# ─── 内部工具 ──────────────────────────────────────────────────────────────────

def _stats(values: list[float]) -> dict:
    if not values:
        return {"mean": 0.0, "count": 0}
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    return {
        "mean": round(sum(values) / n, 4),
        "median": round(sorted_vals[n // 2], 4),
        "min": round(sorted_vals[0], 4),
        "max": round(sorted_vals[-1], 4),
        "count": n,
    }
