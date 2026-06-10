"""Shared helpers for generated gold chunk labels."""

from __future__ import annotations

import json
import os
import platform
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SOURCE_DOC_KEYS = ("news1", "news2", "news3")
UNANSWERABLE_MARKERS = ("无关", "无解", "不可答", "无法回答", "无答案")
UNANSWERABLE_ANSWERS = {"无法回答", "无答案", "不能回答", "不可回答", "unknown", "n/a"}


def unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def crud_doc_hash(task_key: str, event: str, doc_key: str) -> str:
    return f"{task_key}::{event}::{doc_key}"


def _safe_filename_part(value: str, max_len: int = 60) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", (value or "untitled").strip())
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    return (cleaned or "untitled")[:max_len]


def crud_filename(task_key: str, event: str, doc_key: str) -> str:
    return f"{task_key}/{doc_key}/{_safe_filename_part(event)}.txt"


def present_crud_doc_keys(item: dict[str, Any]) -> list[str]:
    return [doc_key for doc_key in SOURCE_DOC_KEYS if (item.get(doc_key) or "").strip()]


def build_crud_gold_item(
    task_key: str,
    item: dict[str, Any],
    filename_to_chunks: dict[str, list[str]],
) -> dict[str, Any] | None:
    question = (item.get("questions") or "").strip()
    answer = (item.get("answers") or "").strip()
    event = item.get("event", "untitled")
    if not question or not answer:
        return None

    relevant_ids: list[str] = []
    source_files: list[str] = []
    for doc_key in present_crud_doc_keys(item):
        filename = crud_filename(task_key, event, doc_key)
        source_files.append(filename)
        relevant_ids.extend(filename_to_chunks.get(filename, []))

    relevant_ids = unique_preserve_order(relevant_ids)
    if not relevant_ids:
        return None

    difficulty = "complex" if task_key in ("questanswer_2docs", "questanswer_3docs") else "simple"
    return {
        "question": question,
        "relevant_chunk_ids": relevant_ids,
        "negative_chunk_ids": [],
        "answer_span": [],
        "reference_answer": answer,
        "difficulty": difficulty,
        "source_files": source_files,
        "tags": [task_key, "CRUD_RAG"],
    }


def is_unanswerable_gold_item(item: dict[str, Any]) -> bool:
    query_type = str(item.get("query_type") or "")
    answer = str(item.get("reference_answer") or item.get("answer") or "").strip().lower()
    if any(marker in query_type for marker in UNANSWERABLE_MARKERS):
        return True
    return answer in UNANSWERABLE_ANSWERS


def build_rageval_gold_item(
    query: dict[str, Any],
    doc_chunk_map: dict[int, list[str]] | dict[str, list[str]],
) -> dict[str, Any] | None:
    question = query.get("query", {}).get("content", "")
    if not question:
        return None

    gt = query.get("ground_truth", {})
    doc_ids = gt.get("doc_ids", [])
    answer = gt.get("content", "")
    keypoints = gt.get("keypoints", [])
    qtype = query.get("query", {}).get("query_type", "")

    candidate_ids: list[str] = []
    for doc_id in doc_ids:
        did = int(doc_id) if isinstance(doc_id, str) and doc_id.isdigit() else doc_id
        candidate_ids.extend(doc_chunk_map.get(did, doc_chunk_map.get(str(did), [])))
    candidate_ids = unique_preserve_order(candidate_ids)

    difficulty = "complex" if len(doc_ids) >= 2 else "simple"
    item = {
        "question": question,
        "relevant_chunk_ids": candidate_ids,
        "negative_chunk_ids": [],
        "answer_span": [],
        "reference_answer": answer,
        "keypoints": keypoints,
        "difficulty": difficulty,
        "domain": query.get("domain", ""),
        "language": query.get("language", ""),
        "query_type": qtype,
        "source_doc_ids": doc_ids,
        "tags": ["RAGEval", query.get("domain", ""), query.get("language", ""), qtype],
    }

    if is_unanswerable_gold_item(item):
        item["relevant_chunk_ids"] = []
        item["negative_chunk_ids"] = candidate_ids
    return item


def refine_relevant_chunk_ids(
    item: dict[str, Any],
    scored_ids: Iterable[tuple[str, float]],
    threshold: float,
    top_n: int,
) -> list[str]:
    if is_unanswerable_gold_item(item):
        return []

    sorted_scores = sorted(scored_ids, key=lambda x: x[1], reverse=True)
    if not sorted_scores:
        return unique_preserve_order(item.get("relevant_chunk_ids", []))

    filtered = [cid for cid, score in sorted_scores if score >= threshold]
    if not filtered:
        filtered = [sorted_scores[0][0]]
    return unique_preserve_order(filtered)[:top_n]


def gold_file_candidates(base: Path, include_raw_fallback: bool = True) -> list[Path]:
    candidates = [
        base / "gold_chunks_crud.json",
        base / "gold_chunks_rageval_refined.json",
    ]
    if include_raw_fallback:
        candidates.append(base / "gold_chunks_rageval.json")
    return candidates


def load_gold_items(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        return data["items"]
    raise ValueError(f"Gold file must be a JSON array or an object with items: {path}")


def gold_generation_metadata(kind: str, **extra: Any) -> dict[str, Any]:
    metadata = {
        "kind": kind,
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "chunk_size_l3": os.getenv("CHUNK_SIZE_L3", "500"),
        "chunk_overlap_l3": os.getenv("CHUNK_OVERLAP_L3", "80"),
        "semantic_threshold": os.getenv("SEMANTIC_THRESHOLD", "0.65"),
        "llm_structure_enabled": os.getenv("LLM_STRUCTURE_ENABLED", "true"),
        "chunk_headers_enabled": os.getenv("CHUNK_HEADERS_ENABLED", "true"),
        "dense_embedding_model": os.getenv("DENSE_EMBEDDING_MODEL", ""),
        "sparse_embedding_model": os.getenv("SPARSE_EMBEDDING_MODEL", ""),
        "milvus_collection": os.getenv("MILVUS_COLLECTION", "embeddings_collection"),
        "python": platform.python_version(),
    }
    metadata.update(extra)
    return metadata


def write_gold_items(path: str | Path, items: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    payload = {
        "metadata": metadata,
        "items": items,
    }
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
