from langchain_core.tools import tool
from contextvars import ContextVar

from backend.chat.rag_context import record_rag_context
from backend.chat.image_evidence import format_image_evidence_for_prompt
from backend.rag.pipeline import run_rag_graph

_KNOWLEDGE_TOOL_CALLS_THIS_TURN = 0
_QUERY_IMAGE_PATH: ContextVar[str | None] = ContextVar("query_image_path", default=None)


def set_query_image_path(path: str | None):
    return _QUERY_IMAGE_PATH.set(path)


def reset_query_image_path(token) -> None:
    _QUERY_IMAGE_PATH.reset(token)


def reset_knowledge_tool_calls() -> None:
    """每轮对话开始时重置知识库工具调用计数。"""
    global _KNOWLEDGE_TOOL_CALLS_THIS_TURN
    _KNOWLEDGE_TOOL_CALLS_THIS_TURN = 0


def _try_acquire_knowledge_tool_call() -> bool:
    global _KNOWLEDGE_TOOL_CALLS_THIS_TURN
    if _KNOWLEDGE_TOOL_CALLS_THIS_TURN >= 1:
        return False
    _KNOWLEDGE_TOOL_CALLS_THIS_TURN += 1
    return True


@tool("search_knowledge_base")
def search_knowledge_base(query: str) -> str:
    """Search for information in the knowledge base using hybrid retrieval (dense + sparse vectors)."""
    if not _try_acquire_knowledge_tool_call():
        return (
            "TOOL_CALL_LIMIT_REACHED: search_knowledge_base has already been called once in this turn. "
            "Use the existing retrieval result and provide the final answer directly."
        )

    query_image_path = _QUERY_IMAGE_PATH.get()
    try:
        rag_result = run_rag_graph(query, query_image_path=query_image_path)
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"RAG graph failed: {e}")
        return f"RAG_ERROR: Retrieval pipeline encountered an error. Please try a different query or check the knowledge base status. Details: {str(e)[:200]}"

    docs = rag_result.get("docs", []) if isinstance(rag_result, dict) else []
    rag_trace = rag_result.get("rag_trace", {}) if isinstance(rag_result, dict) else {}
    record_rag_context(rag_trace)

    if not docs:
        return (
            "KNOWLEDGE_BLINDSPOT: No relevant documents found in the knowledge base. "
            "This question cannot be answered from available documents. "
            "You MUST inform the user that the knowledge base does not contain relevant information "
            "and suggest they upload related documents. Do NOT fabricate an answer."
        )

    # 盲区检测：所有文档分太低
    if rag_trace.get("is_blindspot"):
        blindspot_max = rag_trace.get("blindspot_max_score", 0)
        blindspot_threshold = rag_trace.get("blindspot_threshold", 0.3)
        return (
            f"LOW_CONFIDENCE: Retrieved documents have low relevance (max score {blindspot_max:.3f} < threshold {blindspot_threshold}). "
            "The available documents likely do not contain the answer. "
            "You MUST inform the user that the knowledge base may not have relevant information. "
            "Do NOT fabricate an answer based on weakly related documents."
        )

    formatted = []
    for i, result in enumerate(docs, 1):
        source = result.get("filename", "Unknown")
        page = result.get("page_number", "N/A")
        text = result.get("text", "")
        formatted.append(f"[{i}] {source} (Page {page}):\n{text}")

    image_evidence = format_image_evidence_for_prompt(rag_trace)
    sections = []
    if image_evidence:
        sections.append(image_evidence)
    sections.append("Retrieved Chunks:\n" + "\n\n---\n\n".join(formatted))
    return "\n\n".join(sections)
