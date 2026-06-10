"""单轮对话内 RAG trace 暂存（工具执行 → 流式结束后写入会话）。"""

import contextvars
from typing import Optional

_LAST_RAG_CONTEXT: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar(
    "last_rag_context",
    default=None,
)


def get_last_rag_context(clear: bool = True) -> Optional[dict]:
    """获取最近一次 RAG 检索上下文，默认读取后清空。"""
    context = _LAST_RAG_CONTEXT.get()
    if clear:
        _LAST_RAG_CONTEXT.set(None)
    return context


def record_rag_context(rag_trace: dict) -> None:
    if rag_trace:
        _LAST_RAG_CONTEXT.set({"rag_trace": rag_trace})
