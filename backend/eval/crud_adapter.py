"""
CRUD_RAG 数据集适配器 — 将 questanswer 任务映射到评估框架。

用法:
    from backend.eval.crud_adapter import load_crud_gold, create_crud_retriever

    gold = load_crud_gold()                        # List[dict] 兼容 compute_gold_metrics
    retriever = create_crud_retriever()            # (q, top_k) -> List[dict]
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Callable
from backend.eval.gold_utils import load_gold_items

_PROJ = Path(__file__).resolve().parent.parent.parent
_GOLD_PATH = _PROJ / "data" / "gold_chunks_crud.json"
_SPLIT_PATH = _PROJ / "data" / "CRUD_RAG" / "data" / "crud_split" / "split_merged.json"


def load_crud_gold(gold_path: str | Path | None = None) -> List[Dict[str, Any]]:
    """加载 CRUD_RAG gold 标注（由 scripts/ingest_crud_rag.py 生成）。"""
    path = Path(gold_path) if gold_path else _GOLD_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"CRUD gold 文件不存在: {path}\n"
            f"请先运行: HF_ENDPOINT=https://hf-mirror.com python scripts/ingest_crud_rag.py"
        )
    return load_gold_items(path)


def _load_split() -> dict:
    """加载原始 split_merged.json。"""
    if not _SPLIT_PATH.exists():
        raise FileNotFoundError(f"CRUD 数据集文件不存在: {_SPLIT_PATH}")
    return json.loads(_SPLIT_PATH.read_text(encoding="utf-8"))


def create_crud_retriever(mode: str = "hybrid") -> Callable[[str, int], List[dict]]:
    """创建 CRUD_RAG 评估用的检索函数。"""

    def _retrieve(query: str, top_k: int = 10, **kwargs) -> List[dict]:
        k = kwargs.get("top_k", top_k)
        from backend.rag.utils import retrieve_documents
        return retrieve_documents(query, top_k=k).get("docs", [])

    return _retrieve


def get_crud_llm_eval_data(max_samples: int = 100) -> tuple[List[str], List[str], List[List[str]]]:
    """
    获取 LLM-as-Judge 评估数据：问题、答案、上下文三件套。

    Returns:
        questions: 问题列表
        answers: 参考答案列表
        contexts: 每个问题检索到的上下文列表 (List[str])
    """
    gold = load_crud_gold()
    retriever = create_crud_retriever()

    questions: List[str] = []
    answers: List[str] = []
    contexts: List[List[str]] = []

    for item in gold[:max_samples]:
        question = item.get("question", "")
        answer = item.get("reference_answer", "")
        if not question or not answer:
            continue

        retrieved = retriever(question, top_k=5)
        ctx_texts = [r.get("text", "") for r in retrieved]

        questions.append(question)
        answers.append(answer)
        contexts.append(ctx_texts)

    return questions, answers, contexts


def get_crud_retrievers() -> Dict[str, Callable]:
    """返回 CRUD_RAG 评估的多路检索器字典，可直接注入 runner。"""
    from backend.rag.utils import retrieve_documents

    def dense_only(query: str, top_k: int = 10, **kwargs) -> List[dict]:
        k = kwargs.get("top_k", top_k)
        from backend.indexing.embedding import embedding_service
        from backend.indexing.milvus_client import get_milvus_store

        emb = embedding_service.get_embeddings([query])[0]
        ms = get_milvus_store()
        return ms.dense_retrieve(emb, top_k=k, filter_expr="chunk_level == 3")

    def hybrid(query: str, top_k: int = 10, **kwargs) -> List[dict]:
        k = kwargs.get("top_k", top_k)
        return retrieve_documents(query, top_k=k).get("docs", [])

    return {
        "dense_only": dense_only,
        "hybrid": hybrid,
    }
