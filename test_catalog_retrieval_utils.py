import unittest
from unittest.mock import patch

from backend.chat.image_evidence import format_image_evidence_for_prompt
from backend.rag import utils


class CatalogRetrievalUtilsTest(unittest.TestCase):
    def test_image_evidence_prompt_tells_model_to_use_visual_retrieval(self):
        trace = {
            "has_query_image": True,
            "image_matches": [
                {
                    "filename": "mainland-china-securities-survey-2025.pdf",
                    "page_number": 31,
                    "image_page": 32,
                    "image_kind": "embedded_image",
                    "score": 0.91,
                    "text": "赵鹏\n毕马威中国\n香港金融风险咨询及中资金融机构业务联席主管合伙人",
                }
            ],
        }

        evidence = format_image_evidence_for_prompt(trace)

        self.assertIn("IMAGE_QUERY_EVIDENCE", evidence)
        self.assertIn("不要回答“无法查看图片”", evidence)
        self.assertIn("mainland-china-securities-survey-2025.pdf", evidence)
        self.assertIn("image page 32", evidence)

    def test_catalog_boost_resorts_requested_class_kind_and_domain(self):
        docs = [
            {
                "chunk_id": "network-a-conf",
                "text": "目录方向：计算机网络\n一、A 类\nSIGCOMM\n会议简称\n推荐国际学术会议",
                "score": 100.0,
            },
            {
                "chunk_id": "network-c-conf",
                "text": "目录方向：计算机网络\n三、C 类\nANCS\nAPNOMS\n会议简称\n推荐国际学术会议\n/conf/ancs",
                "score": 1.0,
            },
        ]

        boosted = utils._apply_domain_lexical_boost("推荐一下计算机网络相关的C类国际会议", docs)

        self.assertEqual("network-c-conf", boosted[0]["chunk_id"])

    def test_image_first_search_keeps_low_confidence_visual_context_for_trace(self):
        class FakeMilvus:
            def image_retrieve(self, *_args, **_kwargs):
                return [
                    {
                        "chunk_id": "portrait-page",
                        "filename": "mainland-china-securities-survey-2025.pdf",
                        "page_number": 12,
                        "text": "赵鹏，德勤中国相关负责人。",
                        "score": 0.12,
                        "image_kind": "embedded_image",
                    }
                ]

        with patch.object(utils, "_milvus_manager", FakeMilvus()):
            docs, meta = utils._image_first_search([0.1, 0.2, 0.3], 3, "chunk_level == 3")

        self.assertEqual("portrait-page", docs[0]["chunk_id"])
        self.assertTrue(docs[0]["_image_context_fallback"])
        self.assertTrue(meta["image_context_fallback"])
        self.assertEqual("portrait-page", meta["image_matches"][0]["chunk_id"])

    def test_retrieve_documents_appends_ocr_text_to_text_query_for_image_search(self):
        seen = {}

        class FakeEmbedding:
            def get_embeddings(self, texts):
                seen["dense_query"] = texts[0]
                return [[0.1, 0.2, 0.3]]

            def get_sparse_embedding(self, text):
                seen["sparse_query"] = text
                return {1: 1.0}

        class FakeMultimodal:
            def embed_image(self, _path):
                return [0.1, 0.2, 0.3]

        class FakeMilvus:
            def image_retrieve(self, *_args, **_kwargs):
                return [
                    {
                        "chunk_id": "chart-page",
                        "filename": "report.pdf",
                        "page_number": 4,
                        "text": "图表5 收入结构",
                        "score": 0.5,
                        "image_kind": "page_render",
                    }
                ]

            def dense_retrieve(self, *_args, **_kwargs):
                return []

            def sparse_retrieve(self, *_args, **_kwargs):
                return []

            def query(self, *_args, **_kwargs):
                return []

        with (
            patch.object(utils, "_embedding_service", FakeEmbedding()),
            patch.object(utils, "_milvus_manager", FakeMilvus()),
            patch.object(utils, "_extract_query_image_ocr_text", return_value="图表5 收入结构"),
            patch("backend.indexing.multimodal_embedding.get_multimodal_embedding_service", return_value=FakeMultimodal()),
        ):
            utils._clear_retrieval_cache()
            result = utils.retrieve_documents("这是什么", top_k=1, query_image_path="chart.png")

        self.assertIn("图表5 收入结构", seen["dense_query"])
        self.assertIn("图表5 收入结构", seen["sparse_query"])
        self.assertEqual("图表5 收入结构", result["meta"]["query_image_ocr_text"])


if __name__ == "__main__":
    unittest.main()
