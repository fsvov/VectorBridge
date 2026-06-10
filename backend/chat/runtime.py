import os
import logging

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

from backend.tools import get_current_weather, search_knowledge_base

logger = logging.getLogger(__name__)

API_KEY = os.getenv("LLM_API_KEY")
MODEL = os.getenv("MODEL")
FAST_MODEL = os.getenv("FAST_MODEL") or MODEL
BASE_URL = os.getenv("BASE_URL")
# 备选模型（逗号分隔），主模型不可用时按顺序尝试
_MODEL_BACKUPS_RAW = os.getenv("MODEL_BACKUPS", "")
MODEL_BACKUPS = [m.strip() for m in _MODEL_BACKUPS_RAW.split(",") if m.strip()]

SYSTEM_PROMPT = (
    "You are a professional knowledge assistant powered by RAG (Retrieval-Augmented Generation). "
    "Your role is to provide accurate, well-sourced answers based SOLELY on retrieved documents. "
    "Respond concisely and professionally without casual or anthropomorphic expressions. "
    "Use search_knowledge_base when users ask document/knowledge questions. "
    "If the current turn includes an uploaded image, search_knowledge_base can use the hidden image context for retrieval; call it before answering. "
    "Do not call the same tool repeatedly in one turn. At most one knowledge tool call per turn. "
    "Once you call search_knowledge_base and receive its result, you MUST immediately produce the Final Answer based on that result. "
    "After receiving search_knowledge_base result, you MUST NOT call any tool again. "
    "When answering based on retrieved chunks, you MUST cite the source chunks using their index numbers inline, for example [1] or [2][3]. "
    "If tool results include a Step-back Question/Answer, use that general principle to reason and answer, "
    "but do not reveal chain-of-thought.\n"
    "CRITICAL RULES:\n"
    "- ONLY answer based on the provided document chunks. Do NOT use your training data or general knowledge.\n"
    "- If the chunks do not contain sufficient information, say: '文档未提供相关信息，无法回答此问题。建议补充相关文档。'\n"
    "- If the tool returns KNOWLEDGE_BLINDSPOT or LOW_CONFIDENCE, immediately refuse to answer.\n"
    "- Never fabricate facts, numbers, dates, or names. If unsure, admit it.\n"
    "- When chunks conflict, note the conflict explicitly and cite both sources."
)


def _init_model_with_fallback(model_name: str, temperature: float = 0.3, stream_usage: bool = True):
    """初始化模型，失败时尝试备选。"""
    candidates = [model_name] + [m for m in MODEL_BACKUPS if m != model_name]

    for i, name in enumerate(candidates):
        try:
            m = init_chat_model(
                model=name,
                model_provider="openai",
                api_key=API_KEY,
                base_url=BASE_URL,
                temperature=temperature,
                stream_usage=stream_usage,
                request_timeout=120,
                max_retries=2,
            )
            if i > 0:
                logger.warning(f"主模型 {model_name} 不可用，已切换到备选: {name}")
            return m, name
        except Exception as e:
            if i < len(candidates) - 1:
                logger.warning(f"模型 {name} 初始化失败: {e}，尝试下一个...")
            else:
                raise RuntimeError(f"所有模型初始化失败: {candidates}") from e

    raise RuntimeError(f"无法初始化任何模型")


def create_agent_instance():
    model, _used = _init_model_with_fallback(MODEL, temperature=0.3)
    fast_model, _ = _init_model_with_fallback(FAST_MODEL, temperature=0.2)

    if MODEL_BACKUPS:
        logger.info(f"主模型: {MODEL}, 备选: {MODEL_BACKUPS}")

    agent = create_agent(
        model=model,
        tools=[get_current_weather, search_knowledge_base],
        system_prompt=SYSTEM_PROMPT,
    )
    return agent, model, fast_model


agent, model, fast_model = None, None, None
_init_lock = __import__("threading").Lock()


def _ensure_agent_initialized():
    global agent, model, fast_model
    if agent is not None:
        return
    with _init_lock:
        if agent is not None:
            return
        agent, model, fast_model = create_agent_instance()
