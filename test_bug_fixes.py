import contextvars
import unittest

from backend.api.resources import milvus_filename_filter, sanitize_upload_filename
from backend.chat.rag_context import get_last_rag_context, record_rag_context


class BugFixRegressionTest(unittest.TestCase):
    def test_sanitize_upload_filename_rejects_path_components(self):
        self.assertEqual("report.pdf", sanitize_upload_filename("report.pdf"))

        with self.assertRaises(ValueError):
            sanitize_upload_filename("../report.pdf")

        with self.assertRaises(ValueError):
            sanitize_upload_filename("folder/report.pdf")

        with self.assertRaises(ValueError):
            sanitize_upload_filename(r"C:\fake\report.pdf")

    def test_milvus_filename_filter_escapes_quotes_and_backslashes(self):
        expr = milvus_filename_filter(r'a"b\c.pdf')

        self.assertEqual(r'filename == "a\"b\\c.pdf"', expr)

    def test_rag_context_is_isolated_between_contextvars_contexts(self):
        ctx_a = contextvars.Context()
        ctx_b = contextvars.Context()

        ctx_a.run(record_rag_context, {"retrieved_chunks": [{"filename": "a.pdf"}]})
        ctx_b.run(record_rag_context, {"retrieved_chunks": [{"filename": "b.pdf"}]})

        self.assertEqual("a.pdf", ctx_a.run(get_last_rag_context, False)["rag_trace"]["retrieved_chunks"][0]["filename"])
        self.assertEqual("b.pdf", ctx_b.run(get_last_rag_context, False)["rag_trace"]["retrieved_chunks"][0]["filename"])


if __name__ == "__main__":
    unittest.main()
