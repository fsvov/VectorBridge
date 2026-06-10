"""
RAG 评估运行器 — 一键对比多种检索策略 + 生成 Markdown 报告

用法:
    from backend.eval.runner import EvalRunner
    runner = EvalRunner(gold_data_path="data/gold_chunks.json")
    report = runner.run_all()
    runner.save_report(report, "reports/eval_2026-06-03.md")
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable

from backend.eval.metrics import compute_gold_metrics, compute_ragas_metrics, run_comparison_eval
from backend.eval.llm_judge import evaluate_full as _llm_evaluate_full
from backend.indexing.embedding import embedding_service as _embedding_service
from backend.indexing.milvus_client import get_milvus_store


class EvalRunner:
    """RAG 评估运行器。"""

    def __init__(
        self,
        gold_data_path: str = "data/gold_chunks.json",
        output_dir: str = "reports",
    ):
        self.gold_data_path = Path(gold_data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._milvus = get_milvus_store()
        self._gold: List[dict] = []

    # ─── Gold 数据加载 ───────────────────────────────────────────────────────

    def load_gold(self) -> List[dict]:
        if not self.gold_data_path.exists():
            raise FileNotFoundError(
                f"Gold 标注文件不存在: {self.gold_data_path}\n"
                f"请先按模板创建标注文件（见 data/gold_chunks.json）"
            )
        raw = json.loads(self.gold_data_path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            self._gold = raw
        elif isinstance(raw, dict) and "items" in raw:
            self._gold = raw["items"]
        else:
            raise ValueError("Gold 文件格式错误，应为 JSON 数组或含 items 字段的对象")
        return self._gold

    # ─── 检索函数工厂 ─────────────────────────────────────────────────────────

    def _retriever_factory(self, mode: str) -> Callable:
        """创建不同检索模式的评估用检索函数。"""

        def _retrieve(query: str, top_k: int = 10, **kwargs) -> List[dict]:
            k = kwargs.get("top_k", top_k)
            filter_expr = "chunk_level == 3"

            if mode == "dense_only":
                emb = _embedding_service.get_embeddings([query])[0]
                # 在此作用域内复用已存在的 retrieve_documents 逻辑
                from backend.rag.utils import resolve_candidate_k

                candidate_k, _ = resolve_candidate_k(k)
                return self._milvus.dense_retrieve(emb, top_k=k, filter_expr=filter_expr)

            elif mode == "hybrid":
                from backend.rag.utils import retrieve_documents

                return retrieve_documents(query, top_k=k).get("docs", [])

            elif mode == "hybrid_no_rerank":
                # 临时禁用 rerank
                import backend.rag.utils as rag_utils

                old_model = os.environ.pop("RERANK_MODEL", None)
                try:
                    from backend.rag.utils import retrieve_documents

                    return retrieve_documents(query, top_k=k).get("docs", [])
                finally:
                    if old_model:
                        os.environ["RERANK_MODEL"] = old_model

            return []

        return _retrieve

    # ─── 主评估流程 ───────────────────────────────────────────────────────────

    def run_all(self, ks: tuple = (1, 3, 5, 10)) -> Dict[str, Any]:
        """执行完整评估：gold chunk 对比 + LLM-as-Judge 质量评估。"""
        gold = self.load_gold()
        if not gold:
            return {"error": "Gold 标注集为空"}

        print(f"加载 {len(gold)} 条标注问题")

        # ── 第一部分：Gold Chunk 多路对比 ──
        print("\n=== Gold Chunk 检索对比 ===")
        retrievers = {
            "dense_only": self._retriever_factory("dense_only"),
            "hybrid_no_rerank": self._retriever_factory("hybrid_no_rerank"),
            "hybrid": self._retriever_factory("hybrid"),
        }

        gold_comparison = run_comparison_eval(gold, retrievers, ks=ks)

        # ── 第二部分：LLM-as-Judge 质量评估 ──
        print("\n=== LLM-as-Judge 质量评估 ===")
        llm_results: List[dict] = []
        for item in gold[:20]:  # LLM judge 成本较高，默认取前 20 条
            question = item["question"]
            # 使用 hybrid 检索获取上下文
            retrieved = retrievers["hybrid"](question, top_k=5)
            contexts = [r.get("text", "") for r in retrieved]

            # 模拟回答（实际使用时替换为 Agent 真实回答）
            answer = item.get("reference_answer", "")

            eval_result = _llm_evaluate_full(question, answer, contexts)
            llm_results.append({
                "question": question,
                "answer": answer[:500],
                "num_contexts": len(contexts),
                **eval_result,
            })
            print(f"  [{len(llm_results)}/{min(len(gold), 20)}] {question[:50]}...")

        # ── 第三部分：汇总报告 ──
        report = {
            "title": "RAG 评估报告",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_queries": len(gold),
            "gold_comparison": gold_comparison,
            "llm_judge_samples": llm_results,
            "llm_judge_summary": self._summarize_llm_results(llm_results),
        }

        return report

    # ─── 报告生成 ─────────────────────────────────────────────────────────────

    def _summarize_llm_results(self, results: List[dict]) -> dict:
        metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        summary: Dict[str, Any] = {}
        for m in metrics:
            scores = [r.get(m, {}).get("score", 0) for r in results]
            if scores:
                summary[m] = {
                    "mean": round(sum(scores) / len(scores), 3),
                    "min": min(scores),
                    "max": max(scores),
                }
        return summary

    def save_report(self, report: dict, filename: str = None) -> str:
        if filename is None:
            filename = f"eval_{time.strftime('%Y%m%d_%H%M%S')}.md"
        path = self.output_dir / filename
        md = self._render_markdown(report)
        path.write_text(md, encoding="utf-8")
        print(f"报告已保存: {path}")
        return str(path)

    def _render_markdown(self, report: dict) -> str:
        lines = [
            f"# {report.get('title', 'RAG 评估报告')}",
            f"",
            f"**时间**: {report.get('timestamp')}",
            f"**问题数**: {report.get('num_queries')}",
            f"",
            "---",
            "",
            "## 一、Gold Chunk 检索对比",
            "",
        ]

        comparison = report.get("gold_comparison", {})
        lines.append("| 指标 | dense_only | hybrid (no rerank) | hybrid |")
        lines.append("|------|-----------|-------------------|--------|")

        base_metrics = ["recall@5", "precision@5", "mrr@5", "ndcg@5", "hit_rate@5"]
        has_neg = comparison.get("dense_only", {}).get("hard_negative_precision@5") is not None
        has_span = comparison.get("dense_only", {}).get("answer_grounding@5") is not None
        if has_neg:
            base_metrics.extend(["hard_negative_precision@5", "hard_negative_hit_rate@5", "ndcg_hard_neg@5"])
        if has_span:
            base_metrics.append("answer_grounding@5")

        for metric in base_metrics:
            row = f"| {metric} |"
            for mode in ["dense_only", "hybrid_no_rerank", "hybrid"]:
                data = comparison.get(mode, {}).get(metric, {})
                mean = data.get("mean", "-")
                row += f" {mean} |"
            lines.append(row)

        lines.extend([
            "",
            "---",
            "",
            "## 二、LLM-as-Judge 质量评估",
            "",
        ])

        judge_summary = report.get("llm_judge_summary", {})
        lines.append("| 指标 | 均分 | 最低 | 最高 |")
        lines.append("|------|------|------|------|")
        for name, stats in judge_summary.items():
            lines.append(f"| {name} | {stats.get('mean', '-')} | {stats.get('min', '-')} | {stats.get('max', '-')} |")

        lines.extend([
            "",
            "---",
            "",
            "## 三、分项详情（抽样）",
            "",
        ])

        for i, sample in enumerate(report.get("llm_judge_samples", [])[:5], 1):
            lines.append(f"### 样例 {i}: {sample.get('question', '')[:60]}")
            lines.append("")
            for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
                data = sample.get(m, {})
                lines.append(f"- **{m}**: {data.get('score', '-')} — {data.get('reason', '')}")
            lines.append("")

        return "\n".join(lines)
