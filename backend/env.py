"""从项目根目录加载 .env；由应用入口或独立脚本在 import 其它 backend 模块前调用一次。"""
from pathlib import Path
import os

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LOADED = False


_STRIP_ENV_VARS = (
    "HF_ENDPOINT",
    "BASE_URL",
    "RERANK_BINDING_HOST",
    "EMBEDDING_MODEL",
    "MULTIMODAL_EMBEDDING_MODEL",
)


def _strip_env_values() -> None:
    for name in _STRIP_ENV_VARS:
        value = os.environ.get(name)
        if value is not None:
            os.environ[name] = value.strip()


def load_env() -> None:
    global _LOADED
    if _LOADED:
        return
    load_dotenv(PROJECT_ROOT / ".env")
    _strip_env_values()
    _LOADED = True
