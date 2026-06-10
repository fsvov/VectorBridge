"""多模态嵌入服务 — CLIP 文本+图片统一向量空间（单例 + LRU 缓存）"""
import logging
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from backend.indexing.device import resolve_torch_device

logger = logging.getLogger(__name__)

_IMAGE_CACHE_SIZE = 500
_IMAGE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "images"


class MultimodalEmbeddingService:
    """CLIP ViT-B/32 嵌入服务 — 图片/文本 → 512维归一化向量。"""

    def __init__(self):
        self._model = None
        self._processor = None
        self._device = "cpu"
        self._dim = 512
        self._lock = threading.Lock()
        self._image_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._text_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._loaded = False
        self._load_error: Optional[str] = None

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def available(self) -> bool:
        return self._loaded

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def _ensure_loaded(self) -> bool:
        if self._loaded:
            return True
        with self._lock:
            if self._loaded:
                return True
            try:
                import torch
                from transformers import CLIPModel, CLIPProcessor
                model_name = os.getenv("MULTIMODAL_EMBEDDING_MODEL", "openai/clip-vit-base-patch32")
                self._device = resolve_torch_device("MULTIMODAL_EMBEDDING_DEVICE", default="auto")
                dtype = torch.float16 if self._device.startswith("cuda") else torch.float32
                self._model = CLIPModel.from_pretrained(model_name, torch_dtype=dtype, ignore_mismatched_sizes=True).to(self._device).eval()
                self._processor = CLIPProcessor.from_pretrained(model_name)
                self._loaded = True
                logger.info(f"[multimodal] CLIP 加载完成: {model_name} on {self._device}")
            except Exception as e:
                self._load_error = str(e)
                logger.warning(f"[multimodal] CLIP 加载失败，多模态功能禁用: {e}")
                return False
            return True

    def embed_image(self, image_path: str) -> np.ndarray:
        """单张图片 → 512维归一化向量。"""
        if not self._ensure_loaded():
            return np.zeros(self._dim, dtype=np.float32)

        cache_key = image_path
        with self._lock:
            if cache_key in self._image_cache:
                self._image_cache.move_to_end(cache_key)
                return self._image_cache[cache_key].copy()

        try:
            import torch
            image = Image.open(image_path).convert("RGB")
            inputs = self._processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._model.vision_model(**{k: v for k, v in inputs.items() if k == 'pixel_values'})
                pooled = outputs.pooler_output  # (1, 768)
                if pooled is None:
                    pooled = outputs.last_hidden_state[:, 0, :]  # CLS token
                projected = self._model.visual_projection(pooled)  # (1, 512)
            vec = projected[0].cpu().numpy().astype(np.float32)
            vec = vec / (np.linalg.norm(vec) + 1e-8)
        except Exception as e:
            logger.warning(f"[multimodal] 图片向量化失败 {image_path}: {e}")
            return np.zeros(self._dim, dtype=np.float32)

        with self._lock:
            self._image_cache[cache_key] = vec.copy()
            if len(self._image_cache) > _IMAGE_CACHE_SIZE:
                self._image_cache.popitem(last=False)
        return vec

    def embed_images(self, paths: list[str]) -> list[np.ndarray]:
        return [self.embed_image(p) for p in paths]

    def embed_text(self, text: str) -> np.ndarray:
        """文字 → CLIP 向量空间（用于跨模态检索）。"""
        if not self._ensure_loaded():
            return np.zeros(self._dim, dtype=np.float32)

        cache_key = text
        with self._lock:
            if cache_key in self._text_cache:
                self._text_cache.move_to_end(cache_key)
                return self._text_cache[cache_key].copy()

        try:
            import torch
            inputs = self._processor(text=[text], return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                text_outputs = self._model.text_model(**{k: v for k, v in inputs.items() if k in ('input_ids', 'attention_mask')})
                pooled = text_outputs.pooler_output
                if pooled is None:
                    attn = inputs.get('attention_mask')
                    if attn is not None:
                        pooled = (text_outputs.last_hidden_state * attn.unsqueeze(-1)).sum(dim=1) / attn.sum(dim=1, keepdim=True)
                    else:
                        pooled = text_outputs.last_hidden_state[:, 0, :]
                projected = self._model.text_projection(pooled)  # (1, 512)
            vec = projected[0].cpu().numpy().astype(np.float32)
            vec = vec / (np.linalg.norm(vec) + 1e-8)
        except Exception as e:
            logger.warning(f"[multimodal] 文本向量化失败: {e}")
            return np.zeros(self._dim, dtype=np.float32)

        with self._lock:
            self._text_cache[cache_key] = vec.copy()
            if len(self._text_cache) > _IMAGE_CACHE_SIZE:
                self._text_cache.popitem(last=False)
        return vec


_multimodal_svc: Optional[MultimodalEmbeddingService] = None
_multimodal_lock = threading.Lock()


def get_multimodal_embedding_service() -> MultimodalEmbeddingService:
    global _multimodal_svc
    if _multimodal_svc is None:
        with _multimodal_lock:
            if _multimodal_svc is None:
                _multimodal_svc = MultimodalEmbeddingService()
    return _multimodal_svc


def get_image_dir() -> Path:
    img_dir = Path(os.getenv("IMAGE_STORAGE_DIR", str(_IMAGE_DIR)))
    img_dir.mkdir(parents=True, exist_ok=True)
    return img_dir
