import unittest
import sys
import types
from pathlib import Path
from unittest.mock import patch

from backend.eval.gold_utils import (
    build_crud_gold_item,
    build_rageval_gold_item,
    crud_doc_hash,
    crud_filename,
    gold_file_candidates,
    is_unanswerable_gold_item,
    refine_relevant_chunk_ids,
    unique_preserve_order,
)


class GoldGenerationRegressionTest(unittest.TestCase):
    def test_unique_preserve_order_removes_duplicate_chunk_ids(self):
        self.assertEqual(["a", "b", "c"], unique_preserve_order(["a", "b", "a", "", "c", "b"]))

    def test_crud_filename_and_hash_include_doc_key_and_task(self):
        filename_1 = crud_filename("questanswer_2docs", "same event", "news1")
        filename_2 = crud_filename("questanswer_2docs", "same event", "news2")
        filename_other_task = crud_filename("questanswer_3docs", "same event", "news1")

        self.assertNotEqual(filename_1, filename_2)
        self.assertNotEqual(filename_1, filename_other_task)
        self.assertIn("news1", filename_1)

        self.assertNotEqual(
            crud_doc_hash("questanswer_2docs", "same event", "news1"),
            crud_doc_hash("questanswer_3docs", "same event", "news1"),
        )

    def test_crud_gold_uses_only_present_source_documents_once(self):
        item = {
            "questions": "question",
            "answers": "answer",
            "event": "same event",
            "news1": "source one text",
            "news2": "source two text",
            "news3": "",
        }
        chunks = {
            crud_filename("questanswer_2docs", "same event", "news1"): ["n1-c1", "n1-c1", "n1-c2"],
            crud_filename("questanswer_2docs", "same event", "news2"): ["n2-c1"],
            crud_filename("questanswer_2docs", "same event", "news3"): ["n3-should-not-appear"],
        }

        gold = build_crud_gold_item("questanswer_2docs", item, chunks)

        self.assertEqual(["n1-c1", "n1-c2", "n2-c1"], gold["relevant_chunk_ids"])

    def test_rageval_unanswerable_item_has_empty_relevant_and_negative_chunks(self):
        query = {
            "query": {"content": "unknown question", "query_type": "无关无解问"},
            "ground_truth": {"doc_ids": [5], "content": "无法回答", "keypoints": []},
            "domain": "Finance",
            "language": "zh",
        }

        gold = build_rageval_gold_item(query, {5: ["c1", "c1", "c2"]})

        self.assertEqual([], gold["relevant_chunk_ids"])
        self.assertEqual(["c1", "c2"], gold["negative_chunk_ids"])
        self.assertTrue(is_unanswerable_gold_item(gold))

    def test_refine_does_not_force_top1_for_unanswerable_items(self):
        item = {
            "question": "unknown question",
            "reference_answer": "无法回答",
            "query_type": "无关无解问",
            "relevant_chunk_ids": ["c1", "c2"],
        }
        scores = [("c1", 0.99), ("c2", 0.98)]

        self.assertEqual([], refine_relevant_chunk_ids(item, scores, threshold=0.5, top_n=5))

    def test_refine_deduplicates_answerable_chunks_and_keeps_best_fallback(self):
        item = {
            "question": "answerable question",
            "reference_answer": "concrete answer",
            "query_type": "事实问",
            "relevant_chunk_ids": ["c1", "c1", "c2"],
        }
        scores = [("c1", 0.40), ("c1", 0.40), ("c2", 0.30)]

        self.assertEqual(["c1"], refine_relevant_chunk_ids(item, scores, threshold=0.5, top_n=5))

    def test_gold_file_candidates_prefer_refined_rageval(self):
        base = Path("data")

        self.assertEqual(
            [
                base / "gold_chunks_crud.json",
                base / "gold_chunks_rageval_refined.json",
                base / "gold_chunks_rageval.json",
            ],
            gold_file_candidates(base, include_raw_fallback=True),
        )

    def test_batched_comparison_reports_hard_negative_metrics(self):
        from backend.eval.metrics import run_comparison_eval_batched

        fake_embedding = types.ModuleType("backend.indexing.embedding")
        fake_embedding.embedding_service = types.SimpleNamespace(
            get_embeddings=lambda questions: [[1.0] for _ in questions],
            get_sparse_embeddings=lambda questions: [{"token": 1.0} for _ in questions],
        )

        class FakeMilvus:
            def dense_retrieve(self, *args, **kwargs):
                return [{"chunk_id": "negative", "score": 2.0}]

            def sparse_retrieve(self, *args, **kwargs):
                return [{"chunk_id": "negative", "score": 2.0}]

            def hybrid_retrieve(self, *args, **kwargs):
                return [{"chunk_id": "negative", "score": 2.0}]

        fake_milvus = types.ModuleType("backend.indexing.milvus_client")
        fake_milvus.get_milvus_store = lambda: FakeMilvus()

        fake_ubg = types.ModuleType("backend.rag.ubg_fusion")
        fake_ubg.compute_score_stats = lambda scores: {}
        fake_ubg.get_ubg_heuristic = lambda: types.SimpleNamespace(
            compute_weights=lambda *args, **kwargs: {"dense": 1.0, "sparse": 0.0}
        )

        gold = [{
            "question": "unanswerable question",
            "relevant_chunk_ids": [],
            "negative_chunk_ids": ["negative"],
        }]

        with patch.dict(sys.modules, {
            "backend.indexing.embedding": fake_embedding,
            "backend.indexing.milvus_client": fake_milvus,
            "backend.rag.ubg_fusion": fake_ubg,
        }):
            report = run_comparison_eval_batched(gold, ks=(1,))

        self.assertEqual(1.0, report["dense"]["hn_precision@1"]["mean"])
        self.assertEqual(1.0, report["dense"]["hn_hit@1"]["mean"])
        self.assertEqual(0.0, report["dense"]["ndcg_hn@1"]["mean"])

    def test_refine_fetch_chunk_texts_raises_on_query_failure(self):
        fake_numpy = types.ModuleType("numpy")
        fake_embedding = types.ModuleType("backend.indexing.embedding")
        fake_embedding.embedding_service = types.SimpleNamespace()
        fake_milvus_client = types.ModuleType("backend.indexing.milvus_client")
        fake_milvus_client.get_milvus_store = lambda: None

        class FailingClient:
            def query(self, **kwargs):
                raise RuntimeError("milvus unavailable")

        class FailingMilvus:
            def session(self):
                return self

            def __enter__(self):
                return FailingClient()

            def __exit__(self, exc_type, exc, tb):
                return False

        with patch.dict(sys.modules, {
            "numpy": fake_numpy,
            "backend.indexing.embedding": fake_embedding,
            "backend.indexing.milvus_client": fake_milvus_client,
        }):
            from scripts.refine_rageval_gold import fetch_chunk_texts

        with self.assertRaises(RuntimeError):
            fetch_chunk_texts(FailingMilvus(), {"chunk-a"})

    def test_refine_validates_all_candidate_chunk_texts_are_available(self):
        fake_numpy = types.ModuleType("numpy")
        fake_embedding = types.ModuleType("backend.indexing.embedding")
        fake_embedding.embedding_service = types.SimpleNamespace()
        fake_milvus_client = types.ModuleType("backend.indexing.milvus_client")
        fake_milvus_client.get_milvus_store = lambda: None

        with patch.dict(sys.modules, {
            "numpy": fake_numpy,
            "backend.indexing.embedding": fake_embedding,
            "backend.indexing.milvus_client": fake_milvus_client,
        }):
            from scripts.refine_rageval_gold import validate_chunk_text_coverage

        with self.assertRaisesRegex(RuntimeError, "Missing 1 RAGEval chunk texts"):
            validate_chunk_text_coverage({"chunk-a", "chunk-b"}, {"chunk-a": "text"})

        validate_chunk_text_coverage({"chunk-a"}, {"chunk-a": "text"})


if __name__ == "__main__":
    unittest.main()
