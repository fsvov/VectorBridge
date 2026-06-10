"""RAG 检索步骤的 SSE 实时推送（跨线程安全，供 rag pipeline 在 worker 线程中调用）。"""

import contextvars
import threading

_RAG_STEP_QUEUE = contextvars.ContextVar("rag_step_queue", default=None)
_RAG_STEP_LOOP = contextvars.ContextVar("rag_step_loop", default=None)

# 子 Agent 分组上下文（线程级，支持并行子 Agent 各自独立标识）
_sub_agent_context = threading.local()


def set_rag_step_queue(queue) -> None:
    """设置 RAG 步骤队列，并捕获当前事件循环以便跨线程调度。"""
    if queue:
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()
    else:
        loop = None
    _RAG_STEP_QUEUE.set(queue)
    _RAG_STEP_LOOP.set(loop)


def set_sub_agent_group(group: str) -> None:
    """设置当前线程的子 Agent 分组标识。"""
    _sub_agent_context.group = group


def clear_sub_agent_group() -> None:
    """清除当前线程的子 Agent 分组标识。"""
    _sub_agent_context.group = None


def get_sub_agent_group():
    """获取当前线程的子 Agent 分组标识。"""
    return getattr(_sub_agent_context, "group", None)


def emit_rag_step(icon: str, label: str, detail: str = "") -> None:
    """向队列发送一个 RAG 检索步骤。支持跨线程安全调用。"""
    queue = _RAG_STEP_QUEUE.get()
    loop = _RAG_STEP_LOOP.get()
    if queue is not None and loop is not None:
        step = {"icon": icon, "label": label, "detail": detail}
        group = get_sub_agent_group()
        if group:
            step["group"] = group
        try:
            if not loop.is_closed():
                loop.call_soon_threadsafe(queue.put_nowait, step)
        except Exception:
            pass
