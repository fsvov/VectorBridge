import unittest
import time
from unittest.mock import patch

from langchain_core.messages import HumanMessage

from backend.chat import service


class ChatRoutingTest(unittest.TestCase):
    def test_document_questions_use_direct_rag_instead_of_agent_tool_choice(self):
        self.assertFalse(service._should_use_agent_tools("查询一下StrongBuild Construction相关消息"))
        self.assertFalse(service._should_use_agent_tools("告诉我阿联酋迪拜的相关信息"))

    def test_weather_questions_keep_agent_tool_path(self):
        self.assertTrue(service._should_use_agent_tools("今天上海天气怎么样"))

    def test_image_questions_use_direct_rag(self):
        self.assertFalse(service._should_use_agent_tools("这是什么", query_image_path="query.png"))

    def test_direct_rag_prompt_requires_synthesis_not_verbatim_copy(self):
        prompt = service._build_direct_rag_prompt(
            user_text="查询一下StrongBuild Construction相关消息",
            context="[1] RAGEval/Finance/45.txt (Page 1):\nLong retrieved text",
        )

        self.assertIn("先综合概括", prompt)
        self.assertIn("不要大段照抄", prompt)
        self.assertIn("禁止输出“参考原文”", prompt)
        self.assertIn("必须使用中文回答", prompt)
        self.assertIn("引用", prompt)
        self.assertIn("StrongBuild Construction", prompt)

    def test_chinese_rewrite_request_is_not_treated_as_new_rag_query(self):
        self.assertTrue(service._is_rewrite_previous_answer_request("说中文"))
        self.assertTrue(service._is_rewrite_previous_answer_request("请用中文回答"))
        self.assertTrue(service._is_rewrite_previous_answer_request("翻译成中文"))
        self.assertFalse(service._is_rewrite_previous_answer_request("查询一下中国游戏市场相关消息"))

    def test_followup_question_detection_only_catches_context_dependent_questions(self):
        self.assertTrue(service._is_followup_question("\u76f8\u5173\u4f1a\u8bae\u5462"))
        self.assertTrue(service._is_followup_question("\u8fd8\u6709\u54ea\u4e9b"))
        self.assertTrue(service._is_followup_question("\u90a3B\u7c7b\u5462"))
        self.assertFalse(service._is_followup_question("\u67e5\u8be2\u4e00\u4e0bStrongBuild Construction\u76f8\u5173\u6d88\u606f"))

    def test_followup_query_expands_previous_topic_without_document_filtering(self):
        metadata = {
            "last_rag_topic": {
                "effective_query": "\u8ba1\u7b97\u673a\u7f51\u7edc\u76f8\u5173 CCF \u63a8\u8350\u56fd\u9645\u5b66\u672f\u671f\u520a",
                "filenames": ["ccf.pdf"],
            }
        }

        query = service._build_followup_query("\u76f8\u5173\u4f1a\u8bae\u5462", metadata)

        self.assertIn("\u8ba1\u7b97\u673a\u7f51\u7edc", query)
        self.assertIn("\u56fd\u9645\u5b66\u672f\u4f1a\u8bae", query)
        self.assertNotIn("ccf.pdf", query)

    def test_verbatim_dump_detection_catches_reference_original_sections(self):
        self.assertTrue(service._needs_summary_rewrite("总结如下。\n\n参考原文：\n大段原文内容"))
        self.assertTrue(service._needs_summary_rewrite("检索片段如下：\n[1] 原文"))
        self.assertTrue(service._needs_summary_rewrite("[4] RAGEval/Finance/45.txt (Page 1):\nraw chunk"))
        self.assertFalse(service._needs_summary_rewrite("StrongBuild 的资产和治理信息可概括为两点[1]。"))

    def test_strip_source_dump_sections_removes_copied_chunks_before_summary(self):
        answer = (
            "These strategic investments strengthened the company.\n\n"
            "[4] RAGEval/Finance/45.txt (Page 1):\n"
            "[StrongBuild Construction | 合规与] This commitment to compliance reduces legal risk.\n\n"
            "[5] RAGEval/Finance/45.txt (Page 1):\n"
            "[StrongBuild Construction | 提升盈利] By reducing expenses and streamlining operations.\n\n"
            "IMAGE_QUERY_EVIDENCE:\n"
            "无\n\n"
            "根据检索到的信息，StrongBuild Construction 在2019年提升了盈利能力并增强了市场地位。\n\n"
            "关键信息如下：\n"
            "- 公司通过削减开支、精简运营降低成本 [5]。\n"
        )

        cleaned = service._strip_source_dump_sections(answer)

        self.assertNotIn("RAGEval/Finance/45.txt (Page 1):", cleaned)
        self.assertNotIn("IMAGE_QUERY_EVIDENCE", cleaned)
        self.assertIn("根据检索到的信息", cleaned)
        self.assertIn("削减开支", cleaned)

    def test_direct_rag_docs_are_capped_and_deduplicated_for_generation(self):
        docs = [
            {"chunk_id": "a", "filename": "f.txt", "page_number": 1, "text": "A" * 1000},
            {"chunk_id": "a", "filename": "f.txt", "page_number": 1, "text": "duplicate"},
            {"chunk_id": "b", "filename": "f.txt", "page_number": 1, "text": "B"},
            {"chunk_id": "c", "filename": "f.txt", "page_number": 1, "text": "C"},
            {"chunk_id": "d", "filename": "f.txt", "page_number": 1, "text": "D"},
            {"chunk_id": "e", "filename": "f.txt", "page_number": 1, "text": "E"},
        ]

        formatted = service._format_direct_rag_docs(docs, max_docs=3, max_chars_per_doc=80)

        self.assertIn("[1]", formatted)
        self.assertIn("[3]", formatted)
        self.assertNotIn("[4]", formatted)
        self.assertNotIn("duplicate", formatted)
        self.assertLess(len(formatted), 500)

    def test_direct_rag_rewrites_answer_that_contains_reference_original_section(self):
        class FakeResponse:
            def __init__(self, content):
                self.content = content

        class FakeModel:
            def __init__(self):
                self.prompts = []

            def invoke(self, messages):
                prompt = messages[0].content
                self.prompts.append(prompt)
                if len(self.prompts) == 1:
                    return FakeResponse("结论。\n\n参考原文：\nStrongBuild Construction's total assets stood at $750 million.")
                return FakeResponse("StrongBuild Construction 的资料显示其总资产为 7.5 亿美元，应重点关注资产规模和治理改进[1]。")

        fake_model = FakeModel()
        with (
            patch.object(service, "run_rag_graph", return_value={
                "docs": [{
                    "filename": "RAGEval/Finance/45.txt",
                    "page_number": 1,
                    "text": "StrongBuild Construction's total assets stood at $750 million.",
                }],
                "rag_trace": {"retrieved_chunks": []},
            }),
            patch.object(service.runtime, "model", fake_model),
            patch.object(service.runtime, "fast_model", fake_model),
        ):
            answer = service._direct_rag_answer_sync("查询一下StrongBuild Construction相关消息")

        self.assertNotIn("参考原文", answer)
        self.assertIn("7.5 亿美元", answer)
        self.assertEqual(2, len(fake_model.prompts))

    def test_direct_rag_uses_fast_model_when_primary_generation_times_out(self):
        class FakeResponse:
            def __init__(self, content):
                self.content = content

        class SlowModel:
            def invoke(self, _messages):
                time.sleep(0.05)
                return FakeResponse("slow")

        class FastModel:
            def invoke(self, _messages):
                return FakeResponse("fast answer [1]")

        with (
            patch.object(service, "DIRECT_RAG_ANSWER_TIMEOUT", 0.001),
            patch.object(service, "run_rag_graph", return_value={
                "docs": [{
                    "filename": "RAGEval/Finance/45.txt",
                    "page_number": 1,
                    "text": "StrongBuild Construction data.",
                }],
                "rag_trace": {"retrieved_chunks": []},
            }),
            patch.object(service.runtime, "model", SlowModel()),
            patch.object(service.runtime, "fast_model", FastModel()),
        ):
            answer = service._direct_rag_answer_sync("查询一下StrongBuild Construction相关消息")

        self.assertEqual("fast answer [1]", answer)

    def test_direct_rag_defaults_to_fast_model_without_waiting_for_main(self):
        class FakeResponse:
            def __init__(self, content):
                self.content = content

        class MainModel:
            def invoke(self, _messages):
                raise AssertionError("main model should not be called by default")

        class FastModel:
            def invoke(self, _messages):
                return FakeResponse("fast direct answer [1]")

        with (
            patch.object(service, "DIRECT_RAG_ANSWER_MODEL", "fast"),
            patch.object(service, "run_rag_graph", return_value={
                "docs": [{
                    "filename": "RAGEval/Finance/45.txt",
                    "page_number": 1,
                    "text": "StrongBuild Construction data.",
                }],
                "rag_trace": {"retrieved_chunks": []},
            }),
            patch.object(service.runtime, "model", MainModel()),
            patch.object(service.runtime, "fast_model", FastModel()),
        ):
            answer = service._direct_rag_answer_sync("查询一下StrongBuild Construction相关消息")

        self.assertEqual("fast direct answer [1]", answer)

    def test_direct_rag_returns_extractive_fallback_when_models_fail(self):
        class FailingModel:
            def invoke(self, _messages):
                raise RuntimeError("model down")

        with (
            patch.object(service, "run_rag_graph", return_value={
                "docs": [{
                    "filename": "RAGEval/Finance/45.txt",
                    "page_number": 1,
                    "text": "StrongBuild Construction reduced expenses and improved margins.",
                }],
                "rag_trace": {"retrieved_chunks": []},
            }),
            patch.object(service.runtime, "model", FailingModel()),
            patch.object(service.runtime, "fast_model", FailingModel()),
        ):
            answer = service._direct_rag_answer_sync("查询一下StrongBuild Construction相关消息")

        self.assertIn("生成模型响应超时", answer)
        self.assertIn("StrongBuild Construction", answer)

    def test_chat_with_agent_bypasses_agent_for_document_question(self):
        class FakeStorage:
            def load_with_meta(self, *_args, **_kwargs):
                return [HumanMessage(content="previous")], {"persistent_note": ""}

            def save(self, *_args, **_kwargs):
                return None

        with (
            patch.object(service, "storage", FakeStorage()),
            patch.object(service.runtime, "_ensure_agent_initialized", return_value=None),
            patch.object(service.runtime, "agent") as fake_agent,
            patch.object(service, "_direct_rag_answer_sync", return_value="direct answer") as direct,
            patch.object(service, "_update_persistent_note_sync", return_value=""),
        ):
            fake_agent.invoke.side_effect = AssertionError("agent should not be used")
            result = service.chat_with_agent(
                "查询一下StrongBuild Construction相关消息",
                user_id="test",
                session_id="test",
            )

        self.assertEqual("direct answer", result["response"])
        direct.assert_called_once()

    def test_chat_with_agent_rewrites_previous_answer_for_chinese_request(self):
        class FakeStorage:
            def __init__(self):
                self.saved = []

            def load_with_meta(self, *_args, **_kwargs):
                from langchain_core.messages import AIMessage
                return [HumanMessage(content="previous question"), AIMessage(content="English answer")], {"persistent_note": ""}

            def save(self, *args, **_kwargs):
                self.saved.append(args)

        fake_storage = FakeStorage()
        with (
            patch.object(service, "storage", fake_storage),
            patch.object(service.runtime, "_ensure_agent_initialized", return_value=None),
            patch.object(service, "_rewrite_previous_answer_sync", return_value="中文回答") as rewrite,
            patch.object(service, "_direct_rag_answer_sync") as direct,
            patch.object(service, "_update_persistent_note_sync", return_value=""),
        ):
            result = service.chat_with_agent("说中文", user_id="test", session_id="test")

        self.assertEqual("中文回答", result["response"])
        rewrite.assert_called_once()
        direct.assert_not_called()


    def test_chat_with_agent_uses_rewritten_followup_query_for_direct_rag(self):
        from langchain_core.messages import AIMessage

        class FakeStorage:
            def load_with_meta(self, *_args, **_kwargs):
                return [
                    HumanMessage(content="\u63a8\u8350\u8ba1\u7b97\u673a\u7f51\u7edc\u671f\u520a"),
                    AIMessage(content="\u53ef\u4ee5\u5173\u6ce8\u82e5\u5e72\u671f\u520a"),
                ], {
                    "persistent_note": "",
                    "last_rag_topic": {
                        "effective_query": "\u8ba1\u7b97\u673a\u7f51\u7edc\u76f8\u5173 CCF \u63a8\u8350\u56fd\u9645\u5b66\u672f\u671f\u520a"
                    },
                }

            def save(self, *_args, **_kwargs):
                return None

        with (
            patch.object(service, "storage", FakeStorage()),
            patch.object(service.runtime, "_ensure_agent_initialized", return_value=None),
            patch.object(service, "_direct_rag_answer_sync", return_value="\u76f8\u5173\u4f1a\u8bae\u7b54\u6848") as direct,
            patch.object(service, "_update_persistent_note_sync", return_value=""),
        ):
            result = service.chat_with_agent(
                "\u76f8\u5173\u4f1a\u8bae\u5462",
                user_id="test",
                session_id="test",
            )

        self.assertEqual("\u76f8\u5173\u4f1a\u8bae\u7b54\u6848", result["response"])
        direct.assert_called_once()
        effective_query = direct.call_args.args[0]
        self.assertIn("\u8ba1\u7b97\u673a\u7f51\u7edc", effective_query)
        self.assertIn("\u56fd\u9645\u5b66\u672f\u4f1a\u8bae", effective_query)
        self.assertEqual("\u76f8\u5173\u4f1a\u8bae\u5462", direct.call_args.kwargs["answer_user_text"])

    def test_chat_with_agent_saves_last_rag_topic_after_successful_rag(self):
        class FakeStorage:
            def __init__(self):
                self.saved_metadata = []

            def load_with_meta(self, *_args, **_kwargs):
                return [], {"persistent_note": ""}

            def save(self, *_args, **kwargs):
                if kwargs.get("metadata"):
                    self.saved_metadata.append(kwargs["metadata"])

        def fake_direct(*_args, **_kwargs):
            service.record_rag_context({
                "retrieved_chunks": [
                    {"filename": "ccf.pdf", "text": "\u8ba1\u7b97\u673a\u7f51\u7edc\u671f\u520a"}
                ]
            })
            return "\u53ef\u4ee5\u63a8\u8350\u76f8\u5173\u671f\u520a[1]"

        fake_storage = FakeStorage()
        with (
            patch.object(service, "storage", fake_storage),
            patch.object(service.runtime, "_ensure_agent_initialized", return_value=None),
            patch.object(service, "_generate_session_title_sync", return_value="\u65b0\u4f1a\u8bdd"),
            patch.object(service, "_direct_rag_answer_sync", side_effect=fake_direct),
            patch.object(service, "_update_persistent_note_sync", return_value=""),
        ):
            service.chat_with_agent(
                "\u63a8\u8350\u8ba1\u7b97\u673a\u7f51\u7edc\u671f\u520a",
                user_id="test",
                session_id="test",
            )

        final_meta = fake_storage.saved_metadata[-1]
        self.assertIn("last_rag_topic", final_meta)
        self.assertEqual(
            "\u63a8\u8350\u8ba1\u7b97\u673a\u7f51\u7edc\u671f\u520a",
            final_meta["last_rag_topic"]["effective_query"],
        )
        self.assertEqual(["ccf.pdf"], final_meta["last_rag_topic"]["filenames"])


if __name__ == "__main__":
    unittest.main()
