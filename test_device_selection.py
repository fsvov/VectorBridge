import os
import unittest
from unittest.mock import patch

from backend.indexing.device import resolve_torch_device


class DeviceSelectionTest(unittest.TestCase):
    def test_explicit_cpu_is_preserved(self):
        with patch.dict(os.environ, {"TEST_DEVICE": "cpu"}, clear=False):
            self.assertEqual("cpu", resolve_torch_device("TEST_DEVICE"))

    def test_auto_returns_supported_torch_device(self):
        with patch.dict(os.environ, {"TEST_DEVICE": "auto"}, clear=False):
            self.assertIn(resolve_torch_device("TEST_DEVICE"), {"cpu", "cuda"})


if __name__ == "__main__":
    unittest.main()
