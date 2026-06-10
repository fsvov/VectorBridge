"""管理员接口 — 系统统计与监控"""

from pathlib import Path
from fastapi import APIRouter, Depends, Query

from backend.infra.auth import require_admin, User
from backend.infra.event_logger import get_stats as get_event_stats
from backend.indexing.embedding import embedding_service

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/stats")
async def admin_stats(
    hours: int = Query(default=1, ge=1, le=168),
    _: User = Depends(require_admin),
):
    """获取系统运行统计（仅管理员）。"""
    events = get_event_stats(hours)
    latest_eval = _get_latest_eval()

    return {
        "events": events,
        "embedding": {
            "cache_size": len(embedding_service._embed_cache),
            "vocab_size": len(embedding_service._vocab),
            "total_docs": embedding_service._total_docs,
        },
        "latest_eval": latest_eval,
        "timestamp": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
    }


def _get_latest_eval() -> dict | None:
    reports_dir = Path("reports")
    if not reports_dir.exists():
        return None
    files = sorted(reports_dir.glob("eval_*.md"), reverse=True)
    if not files:
        return None
    latest = files[0]
    content = latest.read_text(encoding="utf-8")[:3000]
    return {"name": latest.name, "size": len(content), "preview": content[:1500]}
