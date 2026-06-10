import os
import re
from pathlib import Path

from backend.indexing import (
    DocumentLoader,
    MilvusWriter,
    ParentChunkStore,
    embedding_service,
)
from backend.indexing.milvus_client import get_milvus_store

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR.parent / "data"
UPLOAD_DIR = DATA_DIR / "documents"

loader = DocumentLoader()
parent_chunk_store = ParentChunkStore()
milvus_manager = get_milvus_store()
milvus_writer = MilvusWriter(embedding_service=embedding_service, milvus_manager=milvus_manager)


def sanitize_upload_filename(filename: str) -> str:
    """Return a safe basename for uploaded documents."""
    raw_name = (filename or "").strip()
    if any(sep in raw_name for sep in ("/", "\\")):
        raise ValueError("文件名不能包含路径分隔符")
    name = Path(raw_name).name
    if not name or name in {".", ".."}:
        raise ValueError("文件名不能为空")
    if re.search(r"[\x00-\x1f]", name):
        raise ValueError("文件名不能包含控制字符")
    return name


def milvus_filename_filter(filename: str) -> str:
    escaped = filename.replace("\\", "\\\\").replace('"', '\\"')
    return f'filename == "{escaped}"'


def remove_bm25_stats_for_filename(filename: str) -> None:
    """删除 Milvus 中该文件对应 chunk 前，先从持久化 BM25 统计中扣减。"""
    rows = milvus_manager.query_all(
        filter_expr=milvus_filename_filter(filename),
        output_fields=["text"],
    )
    texts = [r.get("text") or "" for r in rows]
    embedding_service.increment_remove_documents(texts)


def is_supported_document(filename: str) -> bool:
    file_lower = filename.lower()
    return (
        file_lower.endswith(".pdf")
        or file_lower.endswith((".docx", ".doc"))
        or file_lower.endswith((".xlsx", ".xls"))
        or file_lower.endswith((".html", ".htm"))
    )


async def save_upload_file(file, file_path: Path) -> None:
    with open(file_path, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


def ensure_upload_dir() -> None:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
