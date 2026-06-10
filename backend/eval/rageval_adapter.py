"""
RAGEval / DRAGONBall 数据集适配器

将多领域（金融/法律/医疗）多语言（中/英）QA 数据集映射到评估框架。
文档存储在 data/RAGEval/dragonball_dataset/ 下。
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Callable
from backend.eval.gold_utils import load_gold_items

_PROJ = Path(__file__).resolve().parent.parent.parent
_RAGEVAL_DIR = _PROJ / "data" / "RAGEval" / "dragonball_dataset"
_QUERIES_PATH = _RAGEVAL_DIR / "dragonball_queries.jsonl"
_DOCS_PATH = _RAGEVAL_DIR / "dragonball_docs.jsonl"
_GOLD_PATH = _PROJ / "data" / "gold_chunks_rageval_refined.json"
_RAW_GOLD_PATH = _PROJ / "data" / "gold_chunks_rageval.json"


def load_queries() -> List[dict]:
    """加载所有 RAGEval query 条目。"""
    items = []
    with open(_QUERIES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_docs() -> dict:
    """加载所有 RAGEval 文档，返回 {doc_id: doc_dict}。"""
    docs = {}
    with open(_DOCS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                docs[str(d["doc_id"])] = d
    return docs


def load_rageval_gold(gold_path: str | Path | None = None) -> List[Dict[str, Any]]:
    """加载已生成的 RAGEval gold 标注。"""
    path = Path(gold_path) if gold_path else (_GOLD_PATH if _GOLD_PATH.exists() else _RAW_GOLD_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"RAGEval gold 文件不存在: {path}\n"
            f"请先运行: HF_ENDPOINT=https://hf-mirror.com python scripts/ingest_rageval.py"
        )
    return load_gold_items(path)


def get_rageval_retrievers() -> Dict[str, Callable]:
    """返回四路检索器。"""
    from backend.rag.utils import retrieve_documents as hybrid_retrieve
    from backend.indexing.embedding import embedding_service
    from backend.indexing.milvus_client import get_milvus_store

    ms = get_milvus_store()

    def dense_only(query: str, top_k: int = 10, **kwargs) -> List[dict]:
        k = kwargs.get("top_k", top_k)
        emb = embedding_service.get_embeddings([query])[0]
        return ms.dense_retrieve(emb, top_k=k, filter_expr="chunk_level == 3")

    def sparse_only(query: str, top_k: int = 10, **kwargs) -> List[dict]:
        k = kwargs.get("top_k", top_k)
        sparse = embedding_service.get_sparse_embedding(query)
        return ms.sparse_retrieve(sparse, top_k=k, filter_expr="chunk_level == 3")

    def hybrid(query: str, top_k: int = 10, **kwargs) -> List[dict]:
        k = kwargs.get("top_k", top_k)
        return hybrid_retrieve(query, top_k=k).get("docs", [])

    return {
        "dense": dense_only,
        "sparse": sparse_only,
        "hybrid": hybrid,
    }


def get_rageval_stats() -> dict:
    """RAGEval 数据集统计信息。"""
    queries = load_queries()
    domains: Dict[str, int] = {}
    langs: Dict[str, int] = {}
    qtypes: Dict[str, int] = {}
    for q in queries:
        domains[q.get("domain", "?")] = domains.get(q.get("domain", "?"), 0) + 1
        langs[q.get("language", "?")] = langs.get(q.get("language", "?"), 0) + 1
        qtypes[q.get("query", {}).get("query_type", "?")] = qtypes.get(q.get("query", {}).get("query_type", "?"), 0) + 1

    return {
        "total_queries": len(queries),
        "total_docs": len(load_docs()),
        "domains": domains,
        "languages": langs,
        "query_types": qtypes,
    }
