"""Milvus 访问层：连接池 + 短连接双模式。"""
from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator, TypeVar

from pymilvus import AnnSearchRequest, DataType, MilvusClient, RRFRanker

QUERY_MAX_LIMIT = 16384
T = TypeVar("T")


@dataclass(frozen=True)
class MilvusSettings:
    host: str
    port: str
    collection_name: str
    uri: str
    timeout: float

    @classmethod
    def from_env(cls) -> MilvusSettings:
        host = os.getenv("MILVUS_HOST", "localhost")
        port = os.getenv("MILVUS_PORT", "19530")
        collection = os.getenv("MILVUS_COLLECTION", "embeddings_collection")
        timeout = float(os.getenv("MILVUS_TIMEOUT", "30"))
        return cls(
            host=host,
            port=port,
            collection_name=collection,
            uri=f"http://{host}:{port}",
            timeout=timeout,
        )


@contextmanager
def milvus_client_session(settings: MilvusSettings | None = None) -> Iterator[MilvusClient]:
    cfg = settings or MilvusSettings.from_env()
    client = MilvusClient(uri=cfg.uri, timeout=cfg.timeout)
    try:
        yield client
    finally:
        client.close()


def _normalize_filter(filter_expr: str) -> str:
    return filter_expr.strip() if filter_expr.strip() else "id >= 0"


def _hnsw_search_params(limit: int) -> dict:
    """Milvus HNSW requires ef to be larger than the requested k/limit."""
    base_ef = int(os.getenv("MILVUS_HNSW_SEARCH_EF", "64"))
    ef = max(base_ef, int(limit) + 1)
    return {"metric_type": "IP", "params": {"ef": ef}}


class MilvusStore:
    """Milvus 集合读写；支持两种模式：
    - 短连接（_run）：每次调用新建连接，适合低频操作
    - 长连接池（persistent_session）：复用连接，适合批量评估
    """

    def __init__(self, settings: MilvusSettings | None = None):
        self._settings = settings or MilvusSettings.from_env()
        self._client: MilvusClient | None = None
        self._client_lock = threading.Lock()
        self._operation_lock = threading.Lock()
        self._pool_client: MilvusClient | None = None
        self._pool_lock = threading.Lock()
        self._pool_refcount = 0

    @property
    def collection_name(self) -> str:
        return self._settings.collection_name

    def _get_client(self) -> MilvusClient:
        with self._client_lock:
            if self._client is None:
                self._client = MilvusClient(uri=self._settings.uri, timeout=self._settings.timeout)
            return self._client

    def _reset_client(self) -> None:
        with self._client_lock:
            if self._client is not None:
                try:
                    self._client.close()
                except Exception:
                    pass
                self._client = None

    @staticmethod
    def _is_closed_client_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return (
            "client has been closed" in message
            or "client is closed" in message
            or "has been closed" in message
            or "closed" in message and "client" in message
        )

    def _run(self, operation: Callable[[MilvusClient], T]) -> T:
        """Run an operation on a reusable Milvus client; reconnect once if it was closed."""
        with self._operation_lock:
            client = self._get_client()
            try:
                return operation(client)
            except Exception as exc:
                if not self._is_closed_client_error(exc):
                    raise
                self._reset_client()
                return operation(self._get_client())

    @contextmanager
    def persistent_session(self):
        """长连接池：在此上下文内所有 _run 调用复用同一条连接。"""
        with self._pool_lock:
            if self._pool_client is None:
                self._pool_client = MilvusClient(uri=self._settings.uri, timeout=self._settings.timeout)
            self._pool_refcount += 1
        try:
            yield self._pool_client
        finally:
            with self._pool_lock:
                self._pool_refcount -= 1
                if self._pool_refcount <= 0:
                    try: self._pool_client.close()
                    except Exception: pass
                    self._pool_client = None
                    self._pool_refcount = 0

    @contextmanager
    def session(self) -> Iterator[MilvusClient]:
        """同一业务流内复用一条连接。"""
        with milvus_client_session(self._settings) as client:
            yield client

    @staticmethod
    def ensure_collection(client: MilvusClient, collection_name: str, dense_dim: int, image_dim: int = 512) -> None:
        if client.has_collection(collection_name):
            return

        schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("dense_embedding", DataType.FLOAT_VECTOR, dim=dense_dim)
        schema.add_field("sparse_embedding", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field("image_dense", DataType.FLOAT_VECTOR, dim=image_dim)
        schema.add_field("text", DataType.VARCHAR, max_length=2000)
        schema.add_field("filename", DataType.VARCHAR, max_length=255)
        schema.add_field("file_type", DataType.VARCHAR, max_length=50)
        schema.add_field("file_path", DataType.VARCHAR, max_length=1024)
        schema.add_field("page_number", DataType.INT64)
        schema.add_field("chunk_idx", DataType.INT64)
        schema.add_field("chunk_id", DataType.VARCHAR, max_length=512)
        schema.add_field("parent_chunk_id", DataType.VARCHAR, max_length=512)
        schema.add_field("root_chunk_id", DataType.VARCHAR, max_length=512)
        schema.add_field("chunk_level", DataType.INT64)

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="dense_embedding",
            index_type="HNSW",
            metric_type="IP",
            params={"M": 16, "efConstruction": 256},
        )
        index_params.add_index(
            field_name="sparse_embedding",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
            params={"drop_ratio_build": 0.2},
        )
        index_params.add_index(
            field_name="image_dense",
            index_type="HNSW",
            metric_type="IP",
            params={"M": 16, "efConstruction": 200},
        )
        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )

    def init_collection(self, dense_dim: int | None = None, image_dim: int | None = None) -> None:
        if dense_dim is None:
            dense_dim = int(os.getenv("DENSE_EMBEDDING_DIM", "1024"))
        if image_dim is None:
            image_dim = int(os.getenv("IMAGE_VECTOR_DIM", "512"))

        def _init(client: MilvusClient) -> None:
            self.ensure_collection(client, self.collection_name, dense_dim, image_dim)

        self._run(_init)

    @staticmethod
    def migrate_for_multimodal(client: MilvusClient, collection_name: str) -> bool:
        """迁移旧 collection：如缺少 image_dense 字段则重建。返回是否需要重新入库。"""
        try:
            if not client.has_collection(collection_name):
                return False
            info = client.describe_collection(collection_name)
            field_names = {f["name"] for f in info["fields"]}
            if "image_dense" in field_names:
                return False
            client.drop_collection(collection_name)
            return True
        except Exception:
            return False

    def insert(self, data: list[dict]):
        return self._run(lambda client: client.insert(self.collection_name, data))

    def query(
        self,
        filter_expr: str = "",
        output_fields: list[str] | None = None,
        limit: int = 10000,
        offset: int = 0,
    ):
        expr = _normalize_filter(filter_expr)
        fields = output_fields or ["filename", "file_type"]

        def _query(client: MilvusClient):
            return client.query(
                collection_name=self.collection_name,
                filter=expr,
                output_fields=fields,
                limit=min(limit, QUERY_MAX_LIMIT),
                offset=offset,
            )

        return self._run(_query)

    def query_all(self, filter_expr: str = "", output_fields: list[str] | None = None) -> list:
        """分页拉取；单次 session 内完成，避免每页新建连接。"""
        fields = output_fields or ["filename", "file_type"]
        expr = _normalize_filter(filter_expr)

        def _query_all(client: MilvusClient) -> list:
            out: list = []
            offset = 0
            while True:
                batch = client.query(
                    collection_name=self.collection_name,
                    filter=expr,
                    output_fields=fields,
                    limit=QUERY_MAX_LIMIT,
                    offset=offset,
                )
                if not batch:
                    break
                out.extend(batch)
                if len(batch) < QUERY_MAX_LIMIT:
                    break
                offset += len(batch)
            return out

        return self._run(_query_all)

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[dict]:
        ids = [item for item in chunk_ids if item]
        if not ids:
            return []
        quoted_ids = ", ".join(f'"{item}"' for item in ids)
        return self.query(
            filter_expr=f"chunk_id in [{quoted_ids}]",
            output_fields=[
                "text",
                "filename",
                "file_type",
                "page_number",
                "chunk_id",
                "parent_chunk_id",
                "root_chunk_id",
                "chunk_level",
                "chunk_idx",
            ],
            limit=len(ids),
        )

    def hybrid_retrieve(
        self,
        dense_embedding: list[float],
        sparse_embedding: dict,
        top_k: int = 5,
        rrf_k: int = 60,
        filter_expr: str = "",
    ) -> list[dict]:
        output_fields = [
            "text",
            "filename",
            "file_type",
            "page_number",
            "chunk_id",
            "parent_chunk_id",
            "root_chunk_id",
            "chunk_level",
            "chunk_idx",
        ]
        dense_limit = top_k * 2
        dense_search = AnnSearchRequest(
            data=[dense_embedding],
            anns_field="dense_embedding",
            param=_hnsw_search_params(dense_limit),
            limit=dense_limit,
            expr=filter_expr,
        )
        sparse_search = AnnSearchRequest(
            data=[sparse_embedding],
            anns_field="sparse_embedding",
            param={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}},
            limit=top_k * 2,
            expr=filter_expr,
        )
        reranker = RRFRanker(k=rrf_k)

        def _search(client: MilvusClient):
            return client.hybrid_search(
                collection_name=self.collection_name,
                reqs=[dense_search, sparse_search],
                ranker=reranker,
                limit=top_k,
                output_fields=output_fields,
            )

        results = self._run(_search)
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append({
                    "id": hit.get("id"),
                    "text": hit.get("text", ""),
                    "filename": hit.get("filename", ""),
                    "file_type": hit.get("file_type", ""),
                    "page_number": hit.get("page_number", 0),
                    "chunk_id": hit.get("chunk_id", ""),
                    "parent_chunk_id": hit.get("parent_chunk_id", ""),
                    "root_chunk_id": hit.get("root_chunk_id", ""),
                    "chunk_level": hit.get("chunk_level", 0),
                    "chunk_idx": hit.get("chunk_idx", 0),
                    "score": hit.get("distance", 0.0),
                })
        return formatted_results

    def sparse_retrieve(
        self,
        sparse_embedding: dict,
        top_k: int = 5,
        filter_expr: str = "",
    ) -> list[dict]:
        """纯稀疏向量检索（BM25 only）。"""
        def _search(client: MilvusClient):
            return client.search(
                collection_name=self.collection_name,
                data=[sparse_embedding],
                anns_field="sparse_embedding",
                search_params={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}},
                limit=top_k,
                output_fields=[
                    "text", "filename", "file_type", "page_number",
                    "chunk_id", "parent_chunk_id", "root_chunk_id",
                    "chunk_level", "chunk_idx",
                    "image_path", "image_kind", "image_page", "image_sort_order",
                ],
                filter=filter_expr,
            )

        results = self._run(_search)
        formatted_results = []
        for hits in results:
            for hit in hits:
                ent = hit.get("entity", {})
                formatted_results.append({
                    "id": hit.get("id"),
                    "text": ent.get("text", ""),
                    "filename": ent.get("filename", ""),
                    "file_type": ent.get("file_type", ""),
                    "page_number": ent.get("page_number", 0),
                    "chunk_id": ent.get("chunk_id", ""),
                    "parent_chunk_id": ent.get("parent_chunk_id", ""),
                    "root_chunk_id": ent.get("root_chunk_id", ""),
                    "chunk_level": ent.get("chunk_level", 0),
                    "chunk_idx": ent.get("chunk_idx", 0),
                    "image_path": ent.get("image_path", ""),
                    "image_kind": ent.get("image_kind", ""),
                    "image_page": ent.get("image_page", ent.get("page_number", 0)),
                    "image_sort_order": ent.get("image_sort_order", -999),
                    "score": hit.get("distance", 0.0),
                })
        return formatted_results

    def dense_retrieve(
        self,
        dense_embedding: list[float],
        top_k: int = 5,
        filter_expr: str = "",
    ) -> list[dict]:
        def _search(client: MilvusClient):
            return client.search(
                collection_name=self.collection_name,
                data=[dense_embedding],
                anns_field="dense_embedding",
                search_params=_hnsw_search_params(top_k),
                limit=top_k,
                output_fields=[
                    "text",
                    "filename",
                    "file_type",
                    "page_number",
                    "chunk_id",
                    "parent_chunk_id",
                    "root_chunk_id",
                    "chunk_level",
                    "chunk_idx",
                ],
                filter=filter_expr,
            )

        results = self._run(_search)
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append({
                    "id": hit.get("id"),
                    "text": hit.get("entity", {}).get("text", ""),
                    "filename": hit.get("entity", {}).get("filename", ""),
                    "file_type": hit.get("entity", {}).get("file_type", ""),
                    "page_number": hit.get("entity", {}).get("page_number", 0),
                    "chunk_id": hit.get("entity", {}).get("chunk_id", ""),
                    "parent_chunk_id": hit.get("entity", {}).get("parent_chunk_id", ""),
                    "root_chunk_id": hit.get("entity", {}).get("root_chunk_id", ""),
                    "chunk_level": hit.get("entity", {}).get("chunk_level", 0),
                    "chunk_idx": hit.get("entity", {}).get("chunk_idx", 0),
                    "score": hit.get("distance", 0.0),
                })
        return formatted_results

    def delete(self, filter_expr: str):
        return self._run(
            lambda client: client.delete(collection_name=self.collection_name, filter=filter_expr)
        )

    def image_retrieve(
        self,
        image_embedding: list[float],
        top_k: int = 5,
        filter_expr: str = "",
    ) -> list[dict]:
        """图片向量检索 — 搜索 image_dense 字段（跨模态图文检索）。"""
        def _search(client: MilvusClient):
            return client.search(
                collection_name=self.collection_name,
                data=[image_embedding],
                anns_field="image_dense",
                search_params=_hnsw_search_params(top_k),
                limit=top_k,
                output_fields=[
                    "text", "filename", "file_type", "page_number",
                    "chunk_id", "parent_chunk_id", "root_chunk_id",
                    "chunk_level", "chunk_idx",
                    "image_path", "image_kind", "image_page", "image_sort_order",
                ],
                filter=filter_expr,
            )

        results = self._run(_search)
        formatted_results = []
        for hits in results:
            for hit in hits:
                ent = hit.get("entity", {})
                formatted_results.append({
                    "id": hit.get("id"),
                    "text": ent.get("text", ""),
                    "filename": ent.get("filename", ""),
                    "file_type": ent.get("file_type", ""),
                    "page_number": ent.get("page_number", 0),
                    "chunk_id": ent.get("chunk_id", ""),
                    "parent_chunk_id": ent.get("parent_chunk_id", ""),
                    "root_chunk_id": ent.get("root_chunk_id", ""),
                    "chunk_level": ent.get("chunk_level", 0),
                    "chunk_idx": ent.get("chunk_idx", 0),
                    "image_path": ent.get("image_path", ""),
                    "image_kind": ent.get("image_kind", ""),
                    "image_page": ent.get("image_page", ent.get("page_number", 0)),
                    "image_sort_order": ent.get("image_sort_order", -999),
                    "score": hit.get("distance", 0.0),
                })
        return formatted_results

    def has_collection(self) -> bool:
        return self._run(lambda client: client.has_collection(self.collection_name))

    def drop_collection(self) -> None:
        def _drop(client: MilvusClient) -> None:
            if client.has_collection(self.collection_name):
                client.drop_collection(self.collection_name)

        self._run(_drop)


# 兼容旧名；全项目共用同一无状态 Store 实例即可（不缓存连接）
MilvusManager = MilvusStore

_store: MilvusStore | None = None


def get_milvus_store() -> MilvusStore:
    global _store
    if _store is None:
        _store = MilvusStore()
    return _store
