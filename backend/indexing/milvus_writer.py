"""文档向量化并写入 Milvus - 支持密集+稀疏+图片向量 + debug logging"""
import os

from backend.indexing.embedding import EmbeddingService, embedding_service as _default_embedding_service
from backend.indexing.milvus_client import MilvusStore, get_milvus_store


class MilvusWriter:
    """文档向量化并写入 Milvus 服务 - 支持混合检索"""

    def __init__(self, embedding_service: EmbeddingService = None, milvus_manager: MilvusStore = None):
        self.embedding_service = embedding_service or _default_embedding_service
        self.milvus_manager = milvus_manager or get_milvus_store()

    def write_documents(self, documents: list[dict], batch_size: int = 10, progress_callback=None):
        if not documents:
            return

        # PII 扫描与脱敏 + Injection 检测
        from backend.indexing.pii_scanner import scan_document_chunks, scan_injection
        documents, pii_stats = scan_document_chunks(documents)
        if pii_stats:
            import logging
            logging.getLogger(__name__).info(f"PII detected: {pii_stats}")
        # Prompt Injection 检测
        for doc in documents:
            if scan_injection(doc.get("text", "")):
                import logging
                logging.getLogger(__name__).warning(
                    f"Prompt injection detected in document: {doc.get('filename', 'unknown')}"
                )
                from backend.infra.event_logger import log_event
                log_event("injection_detected", {"filename": doc.get("filename", ""),
                          "chunk_id": doc.get("chunk_id", "")[:40]})

        dense_dim = int(os.getenv("DENSE_EMBEDDING_DIM", "1024"))
        image_dim = int(os.getenv("IMAGE_VECTOR_DIM", "512"))
        all_texts = [doc["text"] for doc in documents]
        self.embedding_service.increment_add_documents(all_texts)
        self.milvus_manager.init_collection(dense_dim=dense_dim, image_dim=image_dim)

        total = len(documents)
        for i in range(0, total, batch_size):
            batch = documents[i : i + batch_size]
            texts = [doc["text"] for doc in batch]
            dense_embeddings, sparse_embeddings = self.embedding_service.get_all_embeddings(texts)

            import numpy as np
            import logging
            _log = logging.getLogger(__name__)
            zero_image = np.zeros(image_dim, dtype=np.float32).tolist()

            insert_data = []
            for doc, dense_emb, sparse_emb in zip(batch, dense_embeddings, sparse_embeddings):
                has_imgs = bool(doc.get("images"))
                img_vecs = []
                for img_info in doc.get("images", []):
                    try:
                        from backend.indexing.multimodal_embedding import get_multimodal_embedding_service
                        vec = get_multimodal_embedding_service().embed_image(img_info["path"])
                        if vec.any():
                            img_vecs.append((vec, img_info))
                    except Exception:
                        pass

                def _image_vector_to_list(image_vector):
                    if isinstance(image_vector, np.ndarray):
                        return image_vector.astype(np.float32).tolist()
                    if isinstance(image_vector, list):
                        return image_vector
                    return zero_image

                image_entries = [
                    (_image_vector_to_list(vec), img_info)
                    for vec, img_info in img_vecs
                ] or [(zero_image, {})]
                for image_vector, img_info in image_entries:
                    if not isinstance(image_vector, list) or not all(isinstance(x, (int, float)) for x in image_vector[:3]):
                        _log.error(f"[writer] BAD image_dense: has_imgs={has_imgs}, n_vecs={len(img_vecs)}, "
                                   f"type={type(image_vector).__name__}, len={len(image_vector) if hasattr(image_vector, '__len__') else '?'}, "
                                   f"first_el_type={type(image_vector[0]).__name__ if hasattr(image_vector, '__getitem__') else '?'}")
                        image_vector = zero_image
                    insert_data.append({
                    "dense_embedding": dense_emb,
                    "sparse_embedding": sparse_emb,
                    "image_dense": image_vector,
                    "text": doc["text"].encode("utf-8")[:2000].decode("utf-8", errors="ignore"),
                    "filename": doc["filename"],
                    "file_type": doc["file_type"],
                    "file_path": doc.get("file_path", ""),
                    "page_number": doc.get("page_number", 0),
                    "chunk_idx": doc.get("chunk_idx", 0),
                    "chunk_id": doc.get("chunk_id", ""),
                    "parent_chunk_id": doc.get("parent_chunk_id", ""),
                    "root_chunk_id": doc.get("root_chunk_id", ""),
                    "chunk_level": doc.get("chunk_level", 0),
                    "image_path": img_info.get("path", ""),
                    "image_kind": img_info.get("kind", ""),
                    "image_page": int(img_info.get("page", doc.get("page_number", 0)) or 0),
                    "image_sort_order": int(img_info.get("sort_order", -999) or -999),
                    })

            self.milvus_manager.insert(insert_data)

            if progress_callback:
                processed = min(i + batch_size, total)
                progress_callback(processed, total)
