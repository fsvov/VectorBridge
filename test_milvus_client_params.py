import unittest

from backend.indexing.milvus_client import _hnsw_search_params


class MilvusClientParamsTest(unittest.TestCase):
    def test_hnsw_ef_is_larger_than_limit(self):
        params = _hnsw_search_params(240)

        self.assertGreater(params["params"]["ef"], 240)


if __name__ == "__main__":
    unittest.main()
