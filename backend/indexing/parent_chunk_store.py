"""Parent chunk storage for auto-merging retrieval."""
from datetime import UTC, datetime
from typing import List

from backend.db.models import ParentChunk
from backend.infra.cache import cache
from backend.infra.database import SessionLocal


class ParentChunkStore:
    """PostgreSQL + Redis storage for parent chunks."""

    @staticmethod
    def _to_dict(item: ParentChunk) -> dict:
        return {
            "text": item.text,
            "filename": item.filename,
            "file_type": item.file_type,
            "file_path": item.file_path,
            "page_number": item.page_number,
            "chunk_id": item.chunk_id,
            "parent_chunk_id": item.parent_chunk_id,
            "root_chunk_id": item.root_chunk_id,
            "chunk_level": item.chunk_level,
            "chunk_idx": item.chunk_idx,
        }

    @staticmethod
    def _cache_key(chunk_id: str) -> str:
        return f"parent_chunk:{chunk_id}"

    @staticmethod
    def _row_from_doc(doc: dict) -> dict | None:
        chunk_id = (doc.get("chunk_id") or "").strip()
        if not chunk_id:
            return None
        return {
            "chunk_id": chunk_id,
            "text": doc.get("text", ""),
            "filename": doc.get("filename", ""),
            "file_type": doc.get("file_type", ""),
            "file_path": doc.get("file_path", ""),
            "page_number": int(doc.get("page_number", 0) or 0),
            "parent_chunk_id": doc.get("parent_chunk_id", ""),
            "root_chunk_id": doc.get("root_chunk_id", ""),
            "chunk_level": int(doc.get("chunk_level", 0) or 0),
            "chunk_idx": int(doc.get("chunk_idx", 0) or 0),
            "updated_at": datetime.now(UTC).replace(tzinfo=None),
        }

    def _prepare_rows(self, docs: List[dict]) -> list[dict]:
        rows_by_id = {}
        for doc in docs:
            row = self._row_from_doc(doc)
            if row is not None:
                rows_by_id[row["chunk_id"]] = row
        return list(rows_by_id.values())

    def upsert_documents(self, docs: List[dict]) -> int:
        """Batch upsert parent chunks and refresh their cache entries."""
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        rows = self._prepare_rows(docs)
        if not rows:
            return 0

        db = SessionLocal()
        try:
            stmt = pg_insert(ParentChunk).values(rows)
            stmt = stmt.on_conflict_do_update(
                index_elements=["chunk_id"],
                set_={
                    "text": stmt.excluded.text,
                    "filename": stmt.excluded.filename,
                    "file_type": stmt.excluded.file_type,
                    "file_path": stmt.excluded.file_path,
                    "page_number": stmt.excluded.page_number,
                    "parent_chunk_id": stmt.excluded.parent_chunk_id,
                    "root_chunk_id": stmt.excluded.root_chunk_id,
                    "chunk_level": stmt.excluded.chunk_level,
                    "chunk_idx": stmt.excluded.chunk_idx,
                    "updated_at": stmt.excluded.updated_at,
                },
            )
            db.execute(stmt)
            db.commit()
        finally:
            db.close()

        for row in rows:
            chunk_id = row["chunk_id"]
            cache.set_json(
                self._cache_key(chunk_id),
                {
                    "chunk_id": chunk_id,
                    "text": row["text"],
                    "filename": row["filename"],
                    "file_type": row["file_type"],
                    "file_path": row["file_path"],
                    "page_number": row["page_number"],
                    "parent_chunk_id": row["parent_chunk_id"],
                    "root_chunk_id": row["root_chunk_id"],
                    "chunk_level": row["chunk_level"],
                    "chunk_idx": row["chunk_idx"],
                },
            )
        return len(rows)

    def get_documents_by_ids(self, chunk_ids: List[str]) -> List[dict]:
        if not chunk_ids:
            return []

        ordered_results = {}
        missing_ids = []
        for chunk_id in chunk_ids:
            key = (chunk_id or "").strip()
            if not key:
                continue
            cached = cache.get_json(self._cache_key(key))
            if cached:
                ordered_results[key] = cached
            else:
                missing_ids.append(key)

        if missing_ids:
            db = SessionLocal()
            try:
                rows = db.query(ParentChunk).filter(ParentChunk.chunk_id.in_(missing_ids)).all()
                for row in rows:
                    payload = self._to_dict(row)
                    ordered_results[row.chunk_id] = payload
                    cache.set_json(self._cache_key(row.chunk_id), payload)
            finally:
                db.close()

        return [ordered_results[item] for item in chunk_ids if item in ordered_results]

    def delete_by_filename(self, filename: str) -> int:
        """Delete parent chunks by filename and return the deleted row count."""
        if not filename:
            return 0

        db = SessionLocal()
        try:
            rows = db.query(ParentChunk).filter(ParentChunk.filename == filename).all()
            chunk_ids = [row.chunk_id for row in rows]
            deleted = len(chunk_ids)
            if deleted > 0:
                db.query(ParentChunk).filter(ParentChunk.filename == filename).delete(synchronize_session=False)
                db.commit()
                for chunk_id in chunk_ids:
                    cache.delete(self._cache_key(chunk_id))
            return deleted
        finally:
            db.close()
