"""Local model device selection helpers."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def resolve_torch_device(env_name: str, default: str = "auto") -> str:
    """Resolve a torch device string from env, with CUDA auto-detection."""
    requested = (os.getenv(env_name) or default or "auto").strip().lower()
    if requested in {"", "auto"}:
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except Exception as exc:
            logger.warning("[%s] CUDA auto-detection failed, falling back to CPU: %s", env_name, exc)
        return "cpu"

    if requested.startswith("cuda"):
        try:
            import torch

            if torch.cuda.is_available():
                return requested
            logger.warning("[%s]=%s requested, but CUDA is unavailable; using CPU", env_name, requested)
        except Exception as exc:
            logger.warning("[%s]=%s requested, but torch CUDA check failed; using CPU: %s", env_name, requested, exc)
        return "cpu"

    return requested
