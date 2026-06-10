import os
import unittest

from backend import env


class EnvLoadingTest(unittest.TestCase):
    def test_load_env_strips_hf_endpoint_from_cmd_set_spacing(self):
        old_value = os.environ.get("HF_ENDPOINT")
        try:
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com "
            env._strip_env_values()
            self.assertEqual("https://hf-mirror.com", os.environ["HF_ENDPOINT"])
        finally:
            if old_value is None:
                os.environ.pop("HF_ENDPOINT", None)
            else:
                os.environ["HF_ENDPOINT"] = old_value


if __name__ == "__main__":
    unittest.main()
