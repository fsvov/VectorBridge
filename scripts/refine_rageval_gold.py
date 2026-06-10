"""Refine RAGEval document-level gold labels into narrower chunk-level labels."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

_PROJ = Path(__file__).resolve().parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

from backend.env import load_env

load_env()

from backend.eval.gold_utils import (
    gold_generation_metadata,
    is_unanswerable_gold_item,
    load_gold_items,
    refine_relevant_chunk_ids,
    unique_preserve_order,
    write_gold_items,
)
from backend.indexing.embedding import embedding_service
from backend.indexing.milvus_client import get_milvus_store

GOLD_IN = _PROJ / "data" / "gold_chunks_rageval.json"
GOLD_OUT = _PROJ / "data" / "gold_chunks_rageval_refined.json"
SIMILARITY_THRESHOLD = float(os.getenv("GOLD_REFINE_THRESHOLD", "0.50"))
TOP_N_PER_DOC = int(os.getenv("GOLD_REFINE_TOP_N", "5"))
BATCH_SIZE = int(os.getenv("GOLD_REFINE_QUERY_BATCH", "200"))


def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def _milvus_string_list(values: list[str]) -> str:
    return json.dumps(values, ensure_ascii=False)


def fetch_chunk_texts(milvus, chunk_ids: set[str]) -> dict[str, str]:
    chunk_texts: dict[str, str] = {}
    if not chunk_ids:
        return chunk_texts

    with milvus.session() as client:
        collection = os.getenv("MILVUS_COLLECTION", "embeddings_collection")
        # 从 Milvus 直接全量拉取所有 RAGEval chunk（避免 chunk_id in [...] 查询限制）
        page = 0
        page_size = 500
        while True:
            try:
                results = client.query(
                    collection_name=collection,
                    filter='file_type == "RAGEval"',
                    output_fields=["chunk_id", "text"],
                    limit=page_size,
                    offset=page * page_size,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to fetch RAGEval chunks from Milvus (page {page})"
                ) from exc
            if not results:
                break
            for row in results:
                chunk_id = row.get("chunk_id", "")
                if chunk_id and chunk_id in chunk_ids:
                    chunk_texts[chunk_id] = row.get("text", "")
            page += 1
    return chunk_texts


def validate_chunk_text_coverage(expected_chunk_ids: set[str], chunk_texts: dict[str, str]) -> None:
    missing = sorted(
        chunk_id
        for chunk_id in expected_chunk_ids
        if not (chunk_texts.get(chunk_id) or "").strip()
    )
    if missing:
        sample = ", ".join(missing[:5])
        raise RuntimeError(
            f"Missing {len(missing)} RAGEval chunk texts from Milvus; "
            f"sample chunk_ids: {sample}"
        )


def score_answer_against_chunks(answer: str, chunk_ids: list[str], chunk_texts: dict[str, str]) -> list[tuple[str, float]]:
    texts: list[str] = []
    valid_ids: list[str] = []
    for chunk_id in unique_preserve_order(chunk_ids):
        text = chunk_texts.get(chunk_id, "")
        if text.strip():
            texts.append(text)
            valid_ids.append(chunk_id)

    if not texts:
        return []

    answer_emb = embedding_service.get_embeddings([answer])[0]
    chunk_embs = embedding_service.get_embeddings(texts)
    return [(chunk_id, float(cosine(answer_emb, emb))) for chunk_id, emb in zip(valid_ids, chunk_embs)]


def main():
    print(f"Loading RAGEval gold: {GOLD_IN}", flush=True)
    gold = load_gold_items(GOLD_IN)
    print(f"  items: {len(gold)}", flush=True)

    all_chunk_ids = set()
    for item in gold:
        for chunk_id in item.get("relevant_chunk_ids", []):
            all_chunk_ids.add(chunk_id)
        for chunk_id in item.get("negative_chunk_ids", []):
            all_chunk_ids.add(chunk_id)
    print(f"  unique candidate chunk ids: {len(all_chunk_ids)}", flush=True)

    print("Fetching candidate chunk texts from Milvus...", flush=True)
    chunk_texts = fetch_chunk_texts(get_milvus_store(), all_chunk_ids)
    missing = all_chunk_ids - set(chunk_texts.keys())
    if missing:
        print(f"  Warning: {len(missing)} chunk_ids not in Milvus (will skip)", flush=True)
    print(f"  fetched chunk texts: {len(chunk_texts)}", flush=True)

    refined: list[dict] = []
    for idx, item in enumerate(gold):
        old_ids = unique_preserve_order(item.get("relevant_chunk_ids", []))
        new_item = dict(item)
        new_item["relevant_chunk_ids"] = old_ids
        new_item["negative_chunk_ids"] = unique_preserve_order(item.get("negative_chunk_ids", []))
        new_item.setdefault("answer_span", [])

        if is_unanswerable_gold_item(new_item):
            new_item["negative_chunk_ids"] = unique_preserve_order(new_item["negative_chunk_ids"] + old_ids)
            new_item["relevant_chunk_ids"] = []
        elif old_ids and new_item.get("reference_answer"):
            try:
                scores = score_answer_against_chunks(new_item["reference_answer"], old_ids, chunk_texts)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to score RAGEval gold item {idx + 1}/{len(gold)}: "
                    f"{new_item.get('question', '')[:80]}"
                ) from exc
            new_item["relevant_chunk_ids"] = refine_relevant_chunk_ids(
                new_item,
                scores,
                threshold=SIMILARITY_THRESHOLD,
                top_n=TOP_N_PER_DOC,
            )

        new_item["_original_count"] = len(old_ids)
        new_item["_refined_count"] = len(new_item["relevant_chunk_ids"])
        refined.append(new_item)

        if (idx + 1) % 500 == 0:
            print(f"  {idx + 1}/{len(gold)}", flush=True)

    total_old = sum(len(unique_preserve_order(g.get("relevant_chunk_ids", []))) for g in gold)
    total_new = sum(len(g.get("relevant_chunk_ids", [])) for g in refined)
    with_ids = sum(1 for g in refined if g.get("relevant_chunk_ids"))
    unanswerable = sum(1 for g in refined if is_unanswerable_gold_item(g))

    write_gold_items(
        GOLD_OUT,
        refined,
        gold_generation_metadata(
            "RAGEval refined",
            source=str(GOLD_IN.name),
            similarity_threshold=SIMILARITY_THRESHOLD,
            top_n_per_doc=TOP_N_PER_DOC,
        ),
    )
    print("\nRefine complete:", flush=True)
    print(f"  chunks: {total_old} -> {total_new}", flush=True)
    print(f"  answerable items with labels: {with_ids}/{len(refined)}", flush=True)
    print(f"  unanswerable items: {unanswerable}", flush=True)
    print(f"  saved: {GOLD_OUT}", flush=True)


if __name__ == "__main__":
    main()
