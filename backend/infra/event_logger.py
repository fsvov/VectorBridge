"""结构化事件日志 — JSONL 格式持久化，支持按类型统计"""

import json
import os
import time
import threading
from pathlib import Path

_EVENT_LOG_PATH = Path(os.getenv("EVENT_LOG_PATH", "data/rag_events.jsonl"))
_EVENT_LOCK = threading.Lock()

# 内存计数器（快速查询，重启清零）
_counters: dict[str, int] = {}
_counter_timestamps: dict[str, list[float]] = {}


def _ensure_dir():
    _EVENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def log_event(event_type: str, detail: dict = None, **kwargs):
    """记录事件到 JSONL 文件。"""
    event = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "timestamp": time.time(),
        "type": event_type,
    }
    if detail:
        event.update(detail)
    if kwargs:
        event.update(kwargs)

    with _EVENT_LOCK:
        _ensure_dir()
        with open(_EVENT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

        _counters[event_type] = _counters.get(event_type, 0) + 1
        _counter_timestamps.setdefault(event_type, []).append(time.time())

    # 内存中只保留最近 1 小时的时间戳
    cutoff = time.time() - 3600
    for k in list(_counter_timestamps.keys()):
        _counter_timestamps[k] = [t for t in _counter_timestamps[k] if t > cutoff]


def log_hybrid_fallback(query: str, reason: str):
    log_event("hybrid_fallback", {"query": query[:100], "reason": reason})


def log_blindspot(query: str, max_score: float):
    log_event("blindspot_hit", {"query": query[:100], "max_score": max_score})


def log_rerank_fallback(query: str, error: str):
    log_event("rerank_fallback", {"query": query[:100], "error": error})


def log_pii_detected(filename: str, stats: dict):
    log_event("pii_detected", {"filename": filename, "stats": stats})


def get_stats(hours: int = 1) -> dict:
    """获取最近 N 小时的事件统计。"""
    cutoff = time.time() - hours * 3600
    stats = {}
    with _EVENT_LOCK:
        for etype, timestamps in _counter_timestamps.items():
            recent = sum(1 for t in timestamps if t > cutoff)
            stats[etype] = {"total": _counters.get(etype, 0), f"last_{hours}h": recent}
    return stats
