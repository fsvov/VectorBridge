"""
RAGEval / DRAGONBall 数据集入库脚本

将 216 篇合成文档逐一切分入 Milvus（单篇写入 + 长间隔），
支持 checkpoint 断点续跑和 Milvus 崩溃自动重试。

用法:
    HF_ENDPOINT=https://hf-mirror.com python scripts/ingest_rageval.py
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_PROJ = Path(__file__).resolve().parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

from backend.env import load_env

load_env()

from backend.indexing.document_loader import DocumentLoader
from backend.indexing.milvus_client import get_milvus_store, MilvusStore
from backend.indexing.embedding import embedding_service
from backend.eval.rageval_adapter import load_docs, load_queries
from backend.eval.gold_utils import build_rageval_gold_item, gold_generation_metadata, write_gold_items

RAGEVAL_DIR = _PROJ / "data" / "RAGEval" / "dragonball_dataset"
GOLD_OUTPUT = _PROJ / "data" / "gold_chunks_rageval.json"
CHECKPOINT_PATH = _PROJ / "data" / ".rageval_ingest_checkpoint.json"
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "embeddings_collection")
SLEEP_BETWEEN_DOCS = 5
MAX_MILVUS_RETRIES = 3

COMPOSE_FILE = _PROJ / "docker-compose.yml"


def _restart_milvus():
    """重启 Milvus standalone 容器并等待 healthy。"""
    print("[milvus] 重启 Milvus...", flush=True)
    subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE_FILE), "restart", "standalone"],
        capture_output=True, timeout=120,
    )
    # 等待 healthy
    for _ in range(30):
        time.sleep(2)
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=milvus-standalone", "--format", "{{.Status}}"],
            capture_output=True, text=True, timeout=10,
        )
        if "(healthy)" in result.stdout:
            print("[milvus] 已恢复 healthy", flush=True)
            return True
    print("[milvus] 重启超时", flush=True)
    return False


def _write_safe(retry_fn, desc: str):
    """带 Milvus 自动重启的写入重试。"""
    for attempt in range(MAX_MILVUS_RETRIES):
        try:
            return retry_fn()
        except Exception as e:
            msg = str(e)
            if "connect" in msg.lower() or "unavailable" in msg.lower() or "10061" in msg:
                print(f"[retry] {desc} 失败 (attempt {attempt+1}): 连接断开", flush=True)
                if attempt < MAX_MILVUS_RETRIES - 1:
                    _restart_milvus()
                    time.sleep(5)
            else:
                raise
    raise RuntimeError(f"{desc} 重试 {MAX_MILVUS_RETRIES} 次后仍失败")


def _load_checkpoint() -> tuple[int, dict, int]:
    """返回 (processed_count, doc_chunk_map, total_l3)。processed_count 是已处理的文档数。"""
    if CHECKPOINT_PATH.exists():
        try:
            ckpt = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
            raw_map = ckpt.get("doc_chunk_map", {})
            doc_chunk_map = {int(k): v for k, v in raw_map.items()}
            return ckpt.get("processed_count", 0), doc_chunk_map, ckpt.get("total_l3", 0)
        except Exception:
            pass
    return 0, {}, 0


def _save_checkpoint(processed_count: int, doc_chunk_map: dict, total_l3: int):
    try:
        CHECKPOINT_PATH.write_text(json.dumps({
            "processed_count": processed_count,
            "doc_chunk_map": {str(k): v for k, v in doc_chunk_map.items()},
            "total_l3": total_l3,
        }), encoding="utf-8")
    except Exception:
        pass


def main():
    global _start
    docs = load_docs()
    queries = load_queries()
    print(f"文档: {len(docs)} 篇, 问题: {len(queries)} 条", flush=True)

    loader = DocumentLoader()
    milvus = get_milvus_store()
    dense_dim = int(os.getenv("DENSE_EMBEDDING_DIM", "1024"))
    milvus.init_collection(dense_dim)

    # ── 断点恢复 ──
    processed_count, doc_chunk_map, total_l3 = _load_checkpoint()
    if processed_count > 0:
        print(f"[checkpoint] 已处理 {processed_count} 篇, L3={total_l3} chunks", flush=True)

    # ── 逐篇入库 ──
    print("\n--- 入库 RAGEval 文档 (逐篇, BATCH_SIZE=1) ---", flush=True)
    doc_list = list(docs.values())
    total_docs = len(doc_list)
    # 只取未处理的 doc_id
    remaining_docs = [d for d in doc_list if d["doc_id"] not in doc_chunk_map]

    if not remaining_docs:
        print("所有文档已处理完毕", flush=True)
    else:
        print(f"剩余: {len(remaining_docs)} 篇", flush=True)

    for i, doc in enumerate(remaining_docs):
        doc_id = doc["doc_id"]
        filename = f"RAGEval/{doc.get('domain', '')}/{doc_id}.txt"
        text = doc.get("content", "")

        # 切分
        try:
            chunks = loader.load_document_from_text(
                text=text, filename=filename, file_type="RAGEval"
            )
        except Exception as e:
            print(f"  [{doc_id}] 切分失败: {e}", flush=True)
            doc_chunk_map[doc_id] = []
            processed_count += 1
            _save_checkpoint(processed_count, doc_chunk_map, total_l3)
            continue

        for c in chunks:
            c["file_type"] = "RAGEval"
            c["file_path"] = f"{doc.get('domain', '')}_{doc.get('language', '')}"
            t = c.get("text", "") or ""
            c["text"] = t.encode("utf-8")[:2000].decode("utf-8", errors="ignore")

        doc_chunk_map[doc_id] = [
            c["chunk_id"] for c in chunks if c.get("chunk_level") == 3
        ]

        leaf_chunks = [c for c in chunks if int(c.get("chunk_level", 0)) == 3]
        if not leaf_chunks:
            processed_count += 1
            _save_checkpoint(processed_count, doc_chunk_map, total_l3)
            continue

        texts = [c["text"] for c in leaf_chunks]
        embedding_service.increment_add_documents(texts)

        # 写入 Milvus（带自动重启重试）
        _write_safe(lambda: _insert_batch(milvus, dense_dim, leaf_chunks, texts),
                    f"doc {doc_id}")

        total_l3 += len(leaf_chunks)
        processed_count += 1
        elapsed = time.time() - _start
        print(f"  [{processed_count}/{total_docs}] doc={doc_id} L3={len(leaf_chunks)} "
              f"(total={total_l3}, {elapsed:.0f}s)", flush=True)

        _save_checkpoint(processed_count, doc_chunk_map, total_l3)
        time.sleep(SLEEP_BETWEEN_DOCS)

    print(f"\n入库: {total_l3} 个 L3 块, {len(doc_chunk_map)} 篇文档", flush=True)

    # ── 生成 Gold 标注 ──
    print("\n--- 生成 Gold 标注 ---", flush=True)
    gold_items = []
    skipped = 0
    for q in queries:
        gold_item = build_rageval_gold_item(q, doc_chunk_map)
        if not gold_item:
            skipped += 1
            continue
        gold_items.append(gold_item)

    write_gold_items(
        GOLD_OUTPUT,
        gold_items,
        gold_generation_metadata("RAGEval", source="RAGEval dragonball_dataset"),
    )
    print(f"Gold: {len(gold_items)} 条 (跳过 {skipped})", flush=True)
    print(f"保存: {GOLD_OUTPUT}", flush=True)

    try:
        CHECKPOINT_PATH.unlink()
    except Exception:
        pass


def _insert_batch(milvus, dense_dim, leaf_chunks, texts):
    with milvus.session() as client:
        MilvusStore.ensure_collection(client, MILVUS_COLLECTION, dense_dim)
        dense_all, sparse_all = embedding_service.get_all_embeddings(texts)
        image_dim = int(os.getenv("IMAGE_VECTOR_DIM", "512"))
        zero_image = np.zeros(image_dim, dtype=np.float32).tolist()
        insert_data = []
        for c, d, s in zip(leaf_chunks, dense_all, sparse_all):
            txt = (c.get("text", "") or "").encode("utf-8")[:2000].decode("utf-8", errors="ignore")
            # 如有提取到的图片，计算 CLIP 向量，否则零向量
            img_vecs = []
            for img_info in c.get("images", []):
                try:
                    from backend.indexing.multimodal_embedding import get_multimodal_embedding_service
                    vec = get_multimodal_embedding_service().embed_image(img_info["path"])
                    if vec.any():
                        img_vecs.append(vec)
                except Exception:
                    pass
            image_vector = img_vecs[0] if img_vecs else zero_image
            insert_data.append({
                "dense_embedding": d,
                "sparse_embedding": s,
                "image_dense": image_vector.tolist() if isinstance(image_vector, np.ndarray) else image_vector,
                "text": txt,
                "filename": c.get("filename", ""),
                "file_type": "RAGEval",
                "file_path": c.get("file_path", ""),
                "page_number": c.get("page_number", 0),
                "chunk_idx": c.get("chunk_idx", 0),
                "chunk_id": c.get("chunk_id", ""),
                "parent_chunk_id": c.get("parent_chunk_id", ""),
                "root_chunk_id": c.get("root_chunk_id", ""),
                "chunk_level": c.get("chunk_level", 0),
            })
        client.insert(MILVUS_COLLECTION, insert_data)


if __name__ == "__main__":
    _start = time.time()
    main()
    print(f"\n总耗时: {time.time() - _start:.0f}s", flush=True)
