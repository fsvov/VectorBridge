"""Conformal Retrieval Confidence — 从 CRANE 迁移的检索置信度校准。

对检索结果的 max relevance score 做 conformal 校准，
输出有统计保证的检索命中概率，替代固定 BLINDSPOT_MIN_SCORE。
"""

import json
import logging
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_STATE_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "conformal_state.json"


class ConformalRetrievalCalibrator:
    """Split Conformal 检索置信度校准器。

    校准集：gold 标注中随机抽取 N 条
    nonconformity score: 1 - max_relevance  (命中时), 1.0 (未命中时)
    """

    def __init__(self, alpha: float = 0.10):
        self.alpha = alpha
        self._quantile: float = 1.0
        self._calibrated = False
        self._cal_size: int = 0
        self._coverage_target: float = 1.0 - alpha

    @property
    def calibrated(self) -> bool:
        return self._calibrated

    @property
    def threshold(self) -> float:
        """置信度阈值：max_score >= threshold 则覆盖。"""
        return max(0.0, 1.0 - self._quantile)

    def calibrate(self, calibration_scores: list[dict]):
        """用校准集计算 nonconformity 分位数。

        calibration_scores: [{"max_score": float, "hit": bool}, ...]
        """
        if len(calibration_scores) < 20:
            logger.warning("[Conformal] 校准集太小 (<20)，使用默认阈值")
            self._quantile = 0.7  # 对应 threshold ≈ 0.3
            self._calibrated = True
            self._cal_size = len(calibration_scores)
            return

        n = len(calibration_scores)
        scores = []
        for item in calibration_scores:
            if item["hit"]:
                scores.append(1.0 - item["max_score"])
            else:
                scores.append(1.0)

        scores.sort()
        # 分位数索引：⌈(n+1)(1-alpha)⌉
        idx = math.ceil((n + 1) * self._coverage_target) - 1
        idx = max(0, min(idx, n - 1))
        self._quantile = scores[idx]
        self._calibrated = True
        self._cal_size = n

        # 在校准集上验证覆盖率
        covered = sum(1 for s in scores if s <= self._quantile)
        actual_coverage = covered / n
        logger.info(
            f"[Conformal] 校准完成: n={n}, alpha={self.alpha:.2f}, "
            f"q={self._quantile:.4f}, threshold={self.threshold:.4f}, "
            f"coverage={actual_coverage:.2%} (target={self._coverage_target:.2%})"
        )

    def predict_confidence(self, max_score: float) -> dict:
        """给定检索最高分，返回置信度判断。未校准时必须由调用方检查 calibrated 字段。"""
        if not self._calibrated:
            return {"covered": True, "confidence": 1.0, "threshold": 0.0, "calibrated": False, "WARNING": "NOT_CALIBRATED"}

        covered = max_score >= self.threshold
        # 将分数映射到置信度区间
        if covered:
            confidence = 0.90 + 0.10 * (max_score - self.threshold) / max(1.0 - self.threshold, 0.01)
        else:
            confidence = 0.90 * max_score / max(self.threshold, 0.01)
        confidence = max(0.0, min(1.0, confidence))

        return {
            "covered": covered,
            "confidence": round(confidence, 4),
            "threshold": round(self.threshold, 4),
            "calibrated": True,
        }


class MondrianRetrievalCalibrator:
    """Mondrian 分层 Conformal — 按 domain 分组校准。"""

    def __init__(self, alpha: float = 0.10):
        self.alpha = alpha
        self._calibrators: dict[str, ConformalRetrievalCalibrator] = {}
        self._default = ConformalRetrievalCalibrator(alpha)

    def calibrate(self, calibration_scores: list[dict]):
        """calibration_scores 每项需含 'domain' 字段。"""
        by_domain: dict[str, list] = {}
        for item in calibration_scores:
            domain = item.get("domain", "default")
            by_domain.setdefault(domain, []).append(item)

        for domain, items in by_domain.items():
            cal = ConformalRetrievalCalibrator(self.alpha)
            cal.calibrate(items)
            self._calibrators[domain] = cal

        self._default.calibrate(calibration_scores)
        logger.info(
            f"[MondrianConformal] {len(self._calibrators)} 个领域已校准: "
            f"{list(self._calibrators.keys())}"
        )

    def predict_confidence(self, max_score: float, domain: str = "default") -> dict:
        cal = self._calibrators.get(domain, self._default)
        return cal.predict_confidence(max_score)


# ── 持久化 ──

def save_calibrator(calibrator: ConformalRetrievalCalibrator, path: str | None = None):
    p = Path(path or os.getenv("CONFORMAL_STATE_PATH", str(_DEFAULT_STATE_PATH)))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "alpha": calibrator.alpha,
        "quantile": calibrator._quantile,
        "cal_size": calibrator._cal_size,
        "coverage_target": calibrator._coverage_target,
    }), encoding="utf-8")


def load_calibrator(alpha: float = 0.10, path: str | None = None) -> ConformalRetrievalCalibrator:
    p = Path(path or os.getenv("CONFORMAL_STATE_PATH", str(_DEFAULT_STATE_PATH)))
    calibrator = ConformalRetrievalCalibrator(alpha)
    if not p.exists():
        return calibrator
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        calibrator.alpha = data.get("alpha", alpha)
        calibrator._quantile = data.get("quantile", 1.0)
        calibrator._calibrated = True
        calibrator._cal_size = data.get("cal_size", 0)
        calibrator._coverage_target = data.get("coverage_target", 1.0 - alpha)
        logger.info(f"[Conformal] 加载校准状态: n={calibrator._cal_size}, threshold={calibrator.threshold:.4f}")
    except Exception as e:
        logger.warning(f"[Conformal] 加载失败: {e}")
    return calibrator
