"""
CRUD_RAG 数据集入库脚本

将 questanswer 任务的源文档分批写入 Milvus，
并自动生成 gold chunk 标注文件。

用法:
    HF_ENDPOINT=https://hf-mirror.com python scripts/ingest_crud_rag.py
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# 确保项目根目录在 sys.path
_PROJ = Path(__file__).resolve().parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

from backend.env import load_env

load_env()

from backend.indexing.document_loader import DocumentLoader
from backend.indexing.milvus_client import get_milvus_store
from backend.indexing.embedding import embedding_service
from backend.eval.gold_utils import (
    build_crud_gold_item,
    crud_doc_hash,
    crud_filename,
    gold_generation_metadata,
    write_gold_items,
)

CRUD_DIR = _PROJ / "data" / "CRUD_RAG" / "data" / "crud_split"
SPLIT_FILE = CRUD_DIR / "split_merged.json"
GOLD_OUTPUT = _PROJ / "data" / "gold_chunks_crud.json"
BATCH_SIZE = 50
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "embeddings_collection")


def load_articles() -> list[dict]:
    """从 CRUD_RAG 中提取所有 questanswer 源文档（去重）。"""
    if not SPLIT_FILE.exists():
        print(f"错误: 找不到 {SPLIT_FILE}")
        sys.exit(1)

    data = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    seen: set[str] = set()
    articles: list[dict] = []

    for task_key in ("questanswer_1doc", "questanswer_2docs", "questanswer_3docs"):
        for item in data.get(task_key, []):
            event = item.get("event", "untitled")
            for doc_key in ("news1", "news2", "news3"):
                text = (item.get(doc_key) or "").strip()
                if not text or len(text) < 50:
                    continue
                doc_hash = crud_doc_hash(task_key, event, doc_key)
                if doc_hash in seen:
                    continue
                seen.add(doc_hash)
                articles.append({
                    "filename": crud_filename(task_key, event, doc_key),
                    "text": text,
                    "doc_key": doc_key,
                    "event": event,
                    "task": task_key,
                })

    print(f"提取到 {len(articles)} 篇唯一源文档", flush=True)
    return articles


def ingest_articles(articles: list[dict], dry_run: bool = False):
    """将文章切分并批量写入 Milvus，返回 filename → [chunk_ids] 映射。"""
    loader = DocumentLoader()
    milvus = get_milvus_store()
    dense_dim = int(os.getenv("DENSE_EMBEDDING_DIM", "1024"))

    # 确保集合存在
    milvus.init_collection(dense_dim)

    filename_to_chunks: dict[str, list[str]] = {}
    total_chunks = 0

    # 断点续跑：检查上次进度
    checkpoint_path = _PROJ / "data" / ".crud_ingest_checkpoint.json"
    start_batch = 0
    if checkpoint_path.exists():
        try:
            ckpt = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            start_batch = ckpt.get("batch_index", 0)
            filename_to_chunks = ckpt.get("filename_to_chunks", {})
            total_chunks = ckpt.get("total_chunks", 0)
            print(f"从断点恢复: batch {start_batch}, 已写入 {total_chunks} chunks", flush=True)
        except Exception:
            pass

    print(f"开始入库 {len(articles)} 篇文章, batch_size={BATCH_SIZE}", flush=True)
    batch_idx = 0
    for batch_start in range(0, len(articles), BATCH_SIZE):
        if batch_idx < start_batch:
            batch_idx += 1
            continue
        batch = articles[batch_start : batch_start + BATCH_SIZE]
        batch_chunks: list[dict] = []
        print(f"  批次 {batch_start//BATCH_SIZE+1}: 处理 {len(batch)} 篇文章...", flush=True)

        for article in batch:
            try:
                chunks = loader.load_document_from_text(
                    text=article["text"],
                    filename=article["filename"],
                    file_type="TXT",
                )
            except Exception as e:
                print(f"  跳过 {article['filename'][:50]}...: {e}")
                continue

            for chunk in chunks:
                chunk["file_type"] = "CRUD_news"
                chunk["file_path"] = article["task"]
                t = (chunk.get("text") or "")
                chunk["text"] = t.encode("utf-8")[:2000].decode("utf-8", errors="ignore")

            batch_chunks.extend(chunks)
            filename_to_chunks[article["filename"]] = [
                c["chunk_id"] for c in chunks if c.get("chunk_level") == 3
            ]

        if not batch_chunks:
            continue

        # 只写 L3 叶子块
        leaf_chunks = [c for c in batch_chunks if int(c.get("chunk_level", 0)) == 3]
        if dry_run:
            print(f"  [dry-run] 批次 {batch_start // BATCH_SIZE + 1}: "
                  f"文章 {len(batch)}, L3块 {len(leaf_chunks)}")
            total_chunks += len(leaf_chunks)
            continue

        if leaf_chunks:
            texts = [c["text"] for c in leaf_chunks]
            embedding_service.increment_add_documents(texts)

            with milvus.session() as client:
                from backend.indexing.milvus_client import MilvusStore

                MilvusStore.ensure_collection(client, MILVUS_COLLECTION, dense_dim)

                dense_embeddings, sparse_embeddings = embedding_service.get_all_embeddings(texts)
                image_dim = int(os.getenv("IMAGE_VECTOR_DIM", "512"))
                zero_image = np.zeros(image_dim, dtype=np.float32).tolist()
                insert_data = []
                for doc, dense_emb, sparse_emb in zip(leaf_chunks, dense_embeddings, sparse_embeddings):
                    txt = str(doc.get("text", ""))
                    txt_bytes = txt.encode("utf-8")
                    if len(txt_bytes) > 2000:
                        txt = txt_bytes[:2000].decode("utf-8", errors="ignore")
                    insert_data.append({
                        "dense_embedding": dense_emb,
                        "sparse_embedding": sparse_emb,
                        "image_dense": zero_image,
                        "text": txt,
                        "filename": doc.get("filename", ""),
                        "file_type": doc.get("file_type", "CRUD_news"),
                        "file_path": doc.get("file_path", ""),
                        "page_number": doc.get("page_number", 0),
                        "chunk_idx": doc.get("chunk_idx", 0),
                        "chunk_id": doc.get("chunk_id", ""),
                        "parent_chunk_id": doc.get("parent_chunk_id", ""),
                        "root_chunk_id": doc.get("root_chunk_id", ""),
                        "chunk_level": doc.get("chunk_level", 0),
                    })
                client.insert(MILVUS_COLLECTION, insert_data)

            total_chunks += len(leaf_chunks)
            elapsed = time.time() - _start_time
            print(f"  [{batch_start + len(batch)}/{len(articles)}] "
                  f"已写入 {total_chunks} 个 L3 块 (耗时 {elapsed:.0f}s)")
            # 断点保存
            try:
                checkpoint_path.write_text(json.dumps({
                    "batch_index": batch_idx + 1,
                    "filename_to_chunks": filename_to_chunks,
                    "total_chunks": total_chunks,
                }), encoding="utf-8")
            except Exception: pass
            batch_idx += 1

    if not dry_run:
        # 清除断点
        try: checkpoint_path.unlink()
        except Exception: pass
        print(f"\n入库完成: {total_chunks} 个叶子块, "
              f"{len(filename_to_chunks)} 个源文档")
    return filename_to_chunks


def generate_gold_labels(filename_to_chunks: dict[str, list[str]]) -> list[dict]:
    """基于 questanswer 源文档映射生成 gold chunk 标注。"""
    data = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    gold_items: list[dict] = []

    for task_key in ("questanswer_1doc", "questanswer_2docs", "questanswer_3docs"):
        for item in data.get(task_key, []):
            gold_item = build_crud_gold_item(task_key, item, filename_to_chunks)
            if gold_item:
                gold_items.append(gold_item)

    print(f"Generated {len(gold_items)} CRUD_RAG gold labels")
    return gold_items

def main():
    global _start_time
    _start_time = time.time()

    articles = load_articles()

    dry_run = "--dry-run" in sys.argv
    filename_to_chunks = ingest_articles(articles, dry_run=dry_run)

    if dry_run:
        print(f"\n[dry-run] 将入库 {sum(len(v) for v in filename_to_chunks.values())} 个 L3 块")
        return

    gold = generate_gold_labels(filename_to_chunks)

    write_gold_items(
        GOLD_OUTPUT,
        gold,
        gold_generation_metadata("CRUD_RAG", source="CRUD_RAG questanswer split_merged.json"),
    )
    print(f"Gold 标注已保存: {GOLD_OUTPUT}")


if __name__ == "__main__":
    main()
