import importlib.util
import os
import sys
import time
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("CHUNK_HEADERS_ENABLED", "false")
os.environ.setdefault("LLM_STRUCTURE_ENABLED", "false")


def load_module(module_name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(module_name, PROJECT_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_rag_utils_with_fakes(module_name: str):
    fake_milvus_client = SimpleNamespace(get_milvus_store=lambda: SimpleNamespace())
    fake_embedding = SimpleNamespace(embedding_service=SimpleNamespace())
    fake_parent_store = SimpleNamespace(ParentChunkStore=lambda: SimpleNamespace())
    fake_chat_models = SimpleNamespace(init_chat_model=lambda *args, **kwargs: SimpleNamespace())
    fake_requests = SimpleNamespace()

    with patch.dict(
        sys.modules,
        {
            "requests": fake_requests,
            "backend.indexing.milvus_client": fake_milvus_client,
            "backend.indexing.embedding": fake_embedding,
            "backend.indexing.parent_chunk_store": fake_parent_store,
            "langchain.chat_models": fake_chat_models,
        },
    ):
        return load_module(module_name, "backend/rag/utils.py")


class ChunkIngestionTest(unittest.TestCase):
    def test_thread_timeout_does_not_wait_for_stuck_pdf_worker(self):
        document_loader = load_module(
            "document_loader_timeout_under_test",
            "backend/indexing/document_loader.py",
        )
        self.assertTrue(hasattr(document_loader, "_run_with_timeout"))

        def stuck_task():
            time.sleep(2)
            return []

        started = time.monotonic()
        with self.assertRaises(Exception):
            document_loader._run_with_timeout(
                stuck_task,
                timeout_seconds=0.1,
                timeout_message="timeout",
            )

        self.assertLess(time.monotonic() - started, 1.0)

    def test_chunk_ids_are_unique_when_one_page_has_multiple_segments(self):
        document_loader = load_module(
            "document_loader_under_test",
            "backend/indexing/document_loader.py",
        )
        document_loader._generate_chunk_header = lambda text, filename: ""

        loader = document_loader.DocumentLoader()
        loader._preprocess_and_segment = lambda text: [
            "first segment text",
            "second segment text",
        ]

        docs = loader._load_from_langchain_docs(
            [SimpleNamespace(page_content="ignored", metadata={"page": 3})],
            file_path="sample.pdf",
            filename="sample.pdf",
            doc_type="PDF",
        )

        chunk_ids = [doc["chunk_id"] for doc in docs]
        duplicates = [chunk_id for chunk_id, count in Counter(chunk_ids).items() if count > 1]

        self.assertEqual([], duplicates)
        self.assertIn("sample.pdf::p3::l1::0", chunk_ids)
        self.assertIn("sample.pdf::p3::l1::1", chunk_ids)

    def test_parent_chunk_rows_are_deduplicated_before_bulk_upsert(self):
        parent_store = load_module(
            "parent_chunk_store_under_test",
            "backend/indexing/parent_chunk_store.py",
        )

        rows = parent_store.ParentChunkStore()._prepare_rows([
            {
                "chunk_id": "same",
                "text": "old",
                "filename": "a.pdf",
                "chunk_level": 1,
            },
            {
                "chunk_id": "same",
                "text": "new",
                "filename": "a.pdf",
                "chunk_level": 1,
            },
            {
                "chunk_id": "other",
                "text": "other",
                "filename": "a.pdf",
                "chunk_level": 2,
            },
        ])

        self.assertEqual(2, len(rows))
        self.assertEqual("new", {row["chunk_id"]: row for row in rows}["same"]["text"])


class RetrievalFusionConfigTest(unittest.TestCase):
    def test_retrieval_fusion_defaults_to_ubg_when_env_is_unset(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RETRIEVAL_FUSION_METHOD", None)

            rag_utils = load_rag_utils_with_fakes("rag_utils_default_fusion_under_test")

        self.assertEqual("ubg", rag_utils.RETRIEVAL_FUSION_METHOD)

    def test_retrieval_fusion_allows_explicit_rrf_override(self):
        with patch.dict(os.environ, {"RETRIEVAL_FUSION_METHOD": "rrf"}):
            rag_utils = load_rag_utils_with_fakes("rag_utils_rrf_fusion_under_test")

        self.assertEqual("rrf", rag_utils.RETRIEVAL_FUSION_METHOD)


class MilvusWriterClientLifecycleTest(unittest.TestCase):
    def test_writer_inserts_each_batch_through_store_without_persistent_client(self):
        milvus_writer = load_module(
            "milvus_writer_under_test",
            "backend/indexing/milvus_writer.py",
        )

        class FakeEmbeddingService:
            def __init__(self):
                self.added = []

            def increment_add_documents(self, texts):
                self.added.extend(texts)

            def get_all_embeddings(self, texts):
                dense = [[1.0, 0.0] for _ in texts]
                sparse = [{"indices": [0], "values": [1.0]} for _ in texts]
                return dense, sparse

        class FakeMilvusStore:
            collection_name = "test_collection"

            def __init__(self):
                self.init_calls = []
                self.inserted_batches = []

            def init_collection(self, dense_dim=None, image_dim=None):
                self.init_calls.append((dense_dim, image_dim))

            def insert(self, data):
                self.inserted_batches.append(data)

            def session(self):
                raise AssertionError("writer should not keep a persistent Milvus client during embedding")

        fake_embedding = FakeEmbeddingService()
        fake_store = FakeMilvusStore()
        writer = milvus_writer.MilvusWriter(fake_embedding, fake_store)
        docs = [
            {
                "text": f"text {idx}",
                "filename": "a.pdf",
                "file_type": "PDF",
                "chunk_id": f"c{idx}",
            }
            for idx in range(3)
        ]

        writer.write_documents(docs, batch_size=2)

        self.assertEqual([(1024, 512)], fake_store.init_calls)
        self.assertEqual(2, len(fake_store.inserted_batches))
        self.assertEqual([2, 1], [len(batch) for batch in fake_store.inserted_batches])


class MilvusStoreReconnectTest(unittest.TestCase):
    def test_run_reconnects_once_when_client_was_closed(self):
        milvus_client = load_module(
            "milvus_client_under_test",
            "backend/indexing/milvus_client.py",
        )

        created_clients = []

        class FakeMilvusClient:
            def __init__(self, uri=None, timeout=None):
                self.closed = False
                self.calls = 0
                created_clients.append(self)

            def close(self):
                self.closed = True

        with patch.object(milvus_client, "MilvusClient", FakeMilvusClient):
            store = milvus_client.MilvusStore(
                milvus_client.MilvusSettings(
                    host="localhost",
                    port="19530",
                    collection_name="test",
                    uri="http://localhost:19530",
                    timeout=1,
                )
            )

            def operation(client):
                client.calls += 1
                if len(created_clients) == 1:
                    raise RuntimeError("Cannot send a request, as the client has been closed.")
                return "ok"

            self.assertEqual("ok", store._run(operation))

        self.assertEqual(2, len(created_clients))
        self.assertTrue(created_clients[0].closed)
        self.assertFalse(created_clients[1].closed)


class RagPipelineStreamingHelperTest(unittest.TestCase):
    def test_rag_step_helpers_bridge_to_streaming_module(self):
        from backend.rag import pipeline

        calls = []
        fake_streaming = SimpleNamespace(
            emit_rag_step=lambda icon, label, detail="": calls.append(("emit", icon, label, detail)),
            clear_sub_agent_group=lambda: calls.append(("clear",)),
        )

        with patch.object(pipeline, "_streaming_mod", fake_streaming):
            pipeline._emit_rag_step("icon", "label", "detail")
            pipeline._clear_sub_agent_group()

        self.assertEqual(
            [
                ("emit", "icon", "label", "detail"),
                ("clear",),
            ],
            calls,
        )

    def test_grade_parser_error_falls_back_to_answer_route_when_docs_exist(self):
        from backend.rag import pipeline

        class FakeGrader:
            def invoke(self, _messages):
                raise ValueError("expected value at line 1 column 1")

        state = {
            "question": "Moon-25",
            "context": "retrieved context",
            "docs": [{"text": "retrieved context", "score": 0.8}],
            "rag_trace": {},
        }

        with patch.object(pipeline, "_get_grader_model", return_value=FakeGrader()):
            with patch.object(pipeline, "_emit_rag_step", lambda *args, **kwargs: None):
                result = pipeline.grade_documents_node(state)

        self.assertEqual("generate_answer", result["route"])
        self.assertEqual("unknown", result["rag_trace"]["grade_score"])
        self.assertIn("expected value", result["rag_trace"]["grade_error"])

    def test_grade_plain_text_yes_routes_to_answer(self):
        from backend.rag import pipeline

        class FakeGrader:
            def invoke(self, _messages):
                return SimpleNamespace(content="yes")

        state = {
            "question": "Moon-25",
            "context": "retrieved context",
            "docs": [{"text": "retrieved context", "score": 0.8}],
            "rag_trace": {},
        }

        with patch.object(pipeline, "_get_grader_model", return_value=FakeGrader()):
            with patch.object(pipeline, "_emit_rag_step", lambda *args, **kwargs: None):
                result = pipeline.grade_documents_node(state)

        self.assertEqual("generate_answer", result["route"])
        self.assertEqual("yes", result["rag_trace"]["grade_score"])
        self.assertNotIn("grade_error", result["rag_trace"])

    def test_domain_lexical_boost_promotes_parallel_distributed_journals(self):
        from backend.rag import utils

        docs = [
            {
                "chunk_id": "generic",
                "score": 240.0,
                "text": "中国计算机学会推荐国际学术期刊 JSAC IEEE Journal on Selected Areas in Communications",
            },
            {
                "chunk_id": "tpds",
                "score": 115.0,
                "text": "TPDS IEEE Transactions on Parallel and Distributed Systems",
            },
        ]

        boosted = utils._sort_by_rank_score(
            utils._apply_domain_lexical_boost(
                "推荐一个并行与分布计算相关的国际学术期刊",
                docs,
            )
        )

        self.assertEqual("tpds", boosted[0]["chunk_id"])
        self.assertGreater(boosted[0].get("domain_lexical_boost", 0), 0)

    def test_domain_lexical_boost_promotes_ccf_b_class_journals(self):
        from backend.rag import utils

        docs = [
            {
                "chunk_id": "a-class-storage",
                "score": 180.0,
                "text": "一、A 类 TOS ACM Transactions on Storage",
            },
            {
                "chunk_id": "b-class-systems",
                "score": 140.0,
                "text": "二、B 类 TAAS ACM Transactions on Autonomous and Adaptive Systems TCC IEEE Transactions on Cloud Computing",
            },
        ]

        boosted = utils._sort_by_rank_score(
            utils._apply_domain_lexical_boost(
                "推荐一下存储系统相关的B类国际期刊",
                docs,
            )
        )

        self.assertEqual("b-class-systems", boosted[0]["chunk_id"])
        self.assertGreater(boosted[0].get("domain_lexical_boost", 0), 0)

    def test_catalog_query_detection_for_ccf_class_questions(self):
        from backend.rag import utils

        self.assertTrue(utils._is_catalog_query("推荐一下计算机网络相关的C类国际会议"))
        self.assertFalse(utils._is_catalog_query("赵鹏是谁"))

    def test_domain_lexical_boost_promotes_requested_c_class_conference(self):
        from backend.rag import utils

        docs = [
            {
                "chunk_id": "network-a",
                "score": 240.0,
                "text": "中国计算机学会推荐国际学术会议 （计算机网络） 一、A 类 SIGCOMM ACM International Conference",
            },
            {
                "chunk_id": "network-c",
                "score": 80.0,
                "text": "三、C 类 会议简称 ANCS ACM/IEEE Symposium on Architectures for Networking and Communication Systems",
            },
        ]

        boosted = utils._sort_by_rank_score(
            utils._apply_domain_lexical_boost(
                "推荐一下计算机网络相关的C类国际会议",
                docs,
            )
        )

        self.assertEqual("network-c", boosted[0]["chunk_id"])
        self.assertGreater(boosted[0].get("domain_lexical_boost", 0), 0)


if __name__ == "__main__":
    unittest.main()
