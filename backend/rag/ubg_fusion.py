"""UBG 检索融合 — 从 CRANE 迁移的 Uncertainty-Bidirectional Gate。

替代固定参数 RRF(k=60)，让模型学习每个查询应该更信任哪一路检索结果。
"""

import logging
import math
import os
from typing import Optional

import torch
from torch import nn

from backend.indexing.device import resolve_torch_device

logger = logging.getLogger(__name__)

# 默认融合权重：dense/sparse/image（不含 image 时只用前两个）
DEFAULT_WEIGHTS_HEURISTIC = {"dense": 0.5, "sparse": 0.4, "image": 0.1}
DEFAULT_WEIGHTS_NO_IMAGE = {"dense": 0.6, "sparse": 0.4}


class UBGRetrievalFusion(nn.Module):
    """检索路径学习的 UBG 融合器。

    输入：query 的密集向量 + 三路检索的分数分布统计量
    输出：三路权重 [w_dense, w_sparse, w_image]（softmax 归一化）

    训练信号：gold chunk 来自哪一路（或加权回归目标）
    """

    def __init__(self, query_dim: int = 1024, hidden_dim: int = 256):
        super().__init__()
        # 分数统计：每路取 top-k 的 mean/std/min/max → 3路 × 4统计 = 12维
        self.score_stats_dim = 12
        input_dim = query_dim + self.score_stats_dim

        self.conf_proj_dense = nn.Linear(input_dim, 1)
        self.conf_proj_sparse = nn.Linear(input_dim, 1)
        self.conf_proj_image = nn.Linear(input_dim, 1)

        self._last_weights: Optional[dict] = None

    def forward(self, query_vec: torch.Tensor, score_stats: torch.Tensor) -> torch.Tensor:
        """前向传播返回三路 softmax 权重。

        Args:
            query_vec: [batch, query_dim] 查询向量
            score_stats: [batch, 12] 三路分数统计 (dense_mean/std/min/max, sparse_..., image_...)

        Returns:
            weights: [batch, 3] 归一化权重
        """
        combined = torch.cat([query_vec, score_stats], dim=-1)

        w_d = torch.sigmoid(self.conf_proj_dense(combined))
        w_s = torch.sigmoid(self.conf_proj_sparse(combined))
        w_i = torch.sigmoid(self.conf_proj_image(combined))

        weights = torch.cat([w_d, w_s, w_i], dim=-1)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        self._last_weights = {
            "dense": float(w_d.mean().item()),
            "sparse": float(w_s.mean().item()),
            "image": float(w_i.mean().item()),
        }
        return weights

    @property
    def last_weights(self) -> Optional[dict]:
        return self._last_weights


class UBGHeuristicFusion:
    """训练前的启发式 UBG：基于 query embedding 的统计特性做快速权重估算。

    规则：
    - query 较短/口语化 → 偏向 sparse（关键词匹配）
    - query 较长/结构化 → 偏向 dense（语义匹配）
    - 有图片查询时 → 分配 ~15% 给 image 路
    """

    def __init__(self):
        pass

    def compute_weights(self, query_vec, score_stats, has_image: bool = False) -> dict:
        """返回权重 dict。"""
        # 基于 query 范数估算复杂度（长 query 通常有更大范数）
        import numpy as np
        if isinstance(query_vec, torch.Tensor):
            query_norm = float((query_vec ** 2).sum() ** 0.5)
        elif isinstance(query_vec, np.ndarray):
            query_norm = float(np.linalg.norm(query_vec))
        else:
            query_norm = 1.0
        norm_factor = min(query_norm / 10.0, 1.0)  # [0, 1]

        if has_image:
            w_d = 0.35 + 0.15 * norm_factor
            w_s = 0.50 - 0.15 * norm_factor
            w_i = 0.15
            total = w_d + w_s + w_i
            return {"dense": w_d / total, "sparse": w_s / total, "image": w_i / total}
        else:
            w_d = 0.45 + 0.15 * norm_factor
            w_s = 0.55 - 0.15 * norm_factor
            return {"dense": w_d, "sparse": w_s, "image": 0.0}


# ── 全局单例 ──

_ubg_model: Optional[UBGRetrievalFusion] = None
_ubg_heuristic: Optional[UBGHeuristicFusion] = None


def load_ubg_fusion(model_path: str | None = None) -> UBGRetrievalFusion:
    """加载训练好的 UBG 模型，失败时返回 None（调用方降级到启发式）。"""
    global _ubg_model
    if _ubg_model is not None:
        return _ubg_model

    path = model_path or os.getenv("UBG_MODEL_PATH", "")
    if not path:
        return None

    try:
        device = resolve_torch_device("UBG_DEVICE", default="auto")
        model = UBGRetrievalFusion()
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device).eval()
        _ubg_model = model
        logger.info(f"[UBG] 模型已加载: {path} on {device}")
        return _ubg_model
    except Exception as e:
        logger.warning(f"[UBG] 模型加载失败，使用启发式: {e}")
        return None


def get_ubg_heuristic() -> UBGHeuristicFusion:
    global _ubg_heuristic
    if _ubg_heuristic is None:
        _ubg_heuristic = UBGHeuristicFusion()
    return _ubg_heuristic


def compute_score_stats(scores: list[float]) -> dict:
    """计算一路检索分数的统计量。"""
    if not scores:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    n = len(scores)
    mean = sum(scores) / n
    var = sum((s - mean) ** 2 for s in scores) / n
    return {"mean": mean, "std": math.sqrt(var), "min": min(scores), "max": max(scores)}
