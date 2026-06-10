import asyncio
import concurrent.futures
import contextvars
import json
import os
import re

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage

import backend.chat.runtime as runtime
from backend.chat.image_evidence import format_image_evidence_for_prompt
from backend.chat.storage import ConversationStorage
from backend.chat.rag_context import get_last_rag_context
from backend.chat.rag_context import record_rag_context
from backend.chat.streaming import emit_rag_step, set_rag_step_queue
from backend.rag.pipeline import run_rag_graph, verify_answer_against_docs
from backend.tools import (
    reset_knowledge_tool_calls,
    reset_query_image_path,
    set_query_image_path,
)

storage = ConversationStorage()

CONTEXT_WINDOW_MESSAGES = 6
DIRECT_RAG_ANSWER_TIMEOUT = float(os.getenv("DIRECT_RAG_ANSWER_TIMEOUT", "90"))
DIRECT_RAG_REWRITE_TIMEOUT = float(os.getenv("DIRECT_RAG_REWRITE_TIMEOUT", "30"))
DIRECT_RAG_ANSWER_MODEL = os.getenv("DIRECT_RAG_ANSWER_MODEL", "fast").strip().lower()
_WEATHER_KEYWORDS = (
    "天气",
    "气温",
    "温度",
    "降雨",
    "下雨",
    "晴天",
    "阴天",
    "weather",
    "temperature",
    "rain",
)


def _should_use_agent_tools(user_text: str, query_image_path: str | None = None) -> bool:
    """Keep agent tool routing only for non-RAG tools whose intent is explicit."""
    if query_image_path:
        return False
    text = (user_text or "").lower()
    return any(keyword in text for keyword in _WEATHER_KEYWORDS)


def _is_rewrite_previous_answer_request(user_text: str) -> bool:
    text = (user_text or "").strip().lower()
    if not text or len(text) > 40:
        return False
    rewrite_markers = (
        "说中文",
        "用中文",
        "中文回答",
        "翻译成中文",
        "换成中文",
        "改成中文",
        "in chinese",
        "chinese please",
    )
    return any(marker in text for marker in rewrite_markers)


def _is_followup_question(user_text: str) -> bool:
    text = (user_text or "").strip()
    if not text:
        return False

    lowered = text.lower()
    if len(text) > 40:
        return False

    explicit_new_query_markers = (
        "\u67e5\u8be2",
        "\u641c\u7d22",
        "\u627e\u4e00\u4e0b",
        "\u4ecb\u7ecd",
        "\u63a8\u8350\u4e00\u4e2a",
        "\u63a8\u8350\u4e00\u4e0b",
        "search",
        "query",
    )
    if any(marker in lowered for marker in explicit_new_query_markers):
        return False

    followup_prefixes = (
        "\u90a3",
        "\u90a3\u4e48",
        "\u8fd9\u4e2a",
        "\u8fd9\u4e9b",
        "\u5b83",
        "\u4ed6",
        "\u5979",
        "\u8be5",
        "\u4e0a\u8ff0",
        "\u4e0a\u9762",
        "\u524d\u9762",
        "\u8fd8\u6709",
        "\u7ee7\u7eed",
        "\u518d",
    )
    if text.startswith(followup_prefixes):
        return True

    followup_markers = (
        "\u76f8\u5173",
        "\u540c\u7c7b",
        "\u7c7b\u4f3c",
        "\u6362\u6210",
        "\u6539\u6210",
        "\u8fd8\u6709\u54ea\u4e9b",
        "\u522b\u7684",
    )
    if any(marker in text for marker in followup_markers):
        return True

    return text.endswith(("\u5462", "\u5462\uff1f", "?")) and len(text) <= 20


def _get_last_rag_topic(metadata: dict | None) -> dict:
    if not isinstance(metadata, dict):
        return {}
    topic = metadata.get("last_rag_topic")
    return topic if isinstance(topic, dict) else {}


def _replace_academic_object(topic: str, followup: str) -> str:
    replacements = (
        ("\u671f\u520a", "\u4f1a\u8bae"),
        ("\u4f1a\u8bae", "\u671f\u520a"),
        ("journal", "conference"),
        ("conference", "journal"),
    )
    result = topic
    for source, target in replacements:
        if target in followup.lower() and source in result.lower():
            flags = re.IGNORECASE if source.isascii() else 0
            result = re.sub(re.escape(source), target, result, flags=flags)

    class_match = re.search(r"([ABCabc])\s*\u7c7b", followup)
    if class_match:
        class_text = f"{class_match.group(1).upper()}\u7c7b"
        if re.search(r"[ABCabc]\s*\u7c7b", result):
            result = re.sub(r"[ABCabc]\s*\u7c7b", class_text, result)
        else:
            result = f"{result} {class_text}"

    return result


def _is_short_query(user_text: str) -> bool:
    """Very short queries likely lack enough context for good retrieval."""
    return len((user_text or "").strip()) <= 10


def _extract_topic_context(persistent_note: str, metadata: dict | None) -> str:
    """Extract likely topic entities from persistent note or last RAG trace.

    Returns a brief context string for query expansion, or empty string.
    """
    topic = _get_last_rag_topic(metadata)
    # Prefer last RAG topic's effective query
    base = (
        topic.get("effective_query")
        or topic.get("user_question")
        or ""
    )
    if base:
        return str(base).strip()

    # Fall back to first meaningful line of persistent note
    if persistent_note:
        lines = [l.strip("- ・*•#●○ ") for l in persistent_note.split("\n") if l.strip()]
        for line in lines[:3]:
            if len(line) >= 6:
                return line
    return ""


def _build_followup_query(user_text: str, metadata: dict | None, previous_answer: str = "", persistent_note: str = "") -> str:
    topic_context = _extract_topic_context(persistent_note, metadata)
    short_fallback = _is_short_query(user_text) and bool(topic_context)
    if not _is_followup_question(user_text) and not short_fallback:
        return user_text

    topic = _get_last_rag_topic(metadata)
    base = (
        topic.get("effective_query")
        or topic.get("user_question")
        or topic.get("topic")
        or ""
    )
    base = str(base).strip()
    if not base:
        base = topic_context
    if not base:
        return user_text

    expanded = _replace_academic_object(base, user_text.strip())
    if expanded == base:
        expanded = f"{base} {user_text.strip()}"

    return re.sub(r"\s+", " ", expanded).strip()


def _build_last_rag_topic(
    user_text: str,
    effective_query: str,
    rag_trace: dict | None,
    response_content: str,
) -> dict | None:
    if not rag_trace or not rag_trace.get("retrieved_chunks"):
        return None
    if "\u6587\u6863\u672a\u63d0\u4f9b\u76f8\u5173\u4fe1\u606f" in (response_content or ""):
        return None

    filenames: list[str] = []
    for chunk in rag_trace.get("retrieved_chunks", []):
        filename = chunk.get("filename") if isinstance(chunk, dict) else None
        if filename and filename not in filenames:
            filenames.append(filename)

    return {
        "user_question": user_text,
        "effective_query": effective_query,
        "answer_excerpt": (response_content or "")[:500],
        "filenames": filenames[:5],
    }


def _last_ai_message_content(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return (msg.content or "").strip() if isinstance(msg.content, str) else str(msg.content)
    return ""


def _looks_like_json_parser_error(error: Exception) -> bool:
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "expected value",
            "jsondecodeerror",
            "outputparserexception",
            "invalid json",
            "could not parse",
        )
    )


def _format_direct_rag_docs(
    docs: list[dict],
    max_docs: int = 5,
    max_chars_per_doc: int = 900,
) -> str:
    formatted = []
    seen: set[str] = set()
    selected: list[dict] = []
    for doc in docs:
        key = doc.get("chunk_id") or f"{doc.get('filename')}::{doc.get('page_number')}::{doc.get('text', '')[:80]}"
        if key in seen:
            continue
        seen.add(key)
        selected.append(doc)
        if len(selected) >= max_docs:
            break

    for i, doc in enumerate(selected, 1):
        source = doc.get("filename", "Unknown")
        page = doc.get("page_number", "N/A")
        text = (doc.get("text", "") or "")[:max_chars_per_doc]
        formatted.append(f"[{i}] {source} (Page {page}):\n{text}")
    return "\n\n---\n\n".join(formatted)


def _build_direct_rag_prompt(user_text: str, context: str, image_evidence: str = "") -> str:
    image_instruction = f"{image_evidence}\n\n" if image_evidence else ""
    return (
        "你是一个基于知识库检索结果回答问题的 RAG 助手。\n"
        "回答规则：\n"
        "1. 只能使用下方 Retrieved chunks 和 IMAGE_QUERY_EVIDENCE 中的信息，不要使用外部常识补充事实。\n"
        "2. 必须使用中文回答；除非用户明确要求其他语言。英文原文中的名称、金额、日期可保留原样。\n"
        "3. 先综合概括检索片段的共同结论，再按要点回答用户问题。\n"
        "4. 不要大段照抄原文；除专有名词、数字、日期、表格字段外，请用自己的话压缩和归纳。\n"
        "5. 每个关键结论后用 [1]、[2][3] 这样的编号引用来源。\n"
        "6. 如果只能定位到文档或页面但不能确定更细粒度信息，要直接说明边界。\n"
        "7. 如果检索片段不足以回答，回答：文档未提供相关信息，无法回答此问题。建议补充相关文档。\n\n"
        "禁止事项：\n"
        "禁止输出“参考原文”“原文如下”“检索片段”“相关原文”“文档原文”等段落标题。\n"
        "禁止把 Retrieved chunks 当作附录复制到正文中。\n"
        "禁止连续引用超过 40 个汉字或 80 个英文字符的原文。\n\n"
        "输出格式：\n"
        "先给 1-2 句直接结论；再用 2-5 个要点补充关键信息。不要输出长篇原文摘录。\n\n"
        f"{image_instruction}"
        f"User question:\n{user_text}\n\n"
        f"Retrieved chunks:\n{context}"
    )


_VERBATIM_DUMP_MARKERS = (
    "参考原文",
    "原文如下",
    "相关原文",
    "检索片段",
    "Retrieved chunks",
    "Retrieved Chunks",
    "文档原文",
    "以下是原文",
)
_SOURCE_DUMP_HEADER_RE = re.compile(r"^\[\d+\]\s+.+?\s+\(Page\s+[^)]*\):\s*$")


def _needs_summary_rewrite(answer: str) -> bool:
    if not answer:
        return False
    if any(marker in answer for marker in _VERBATIM_DUMP_MARKERS):
        return True
    return any(_SOURCE_DUMP_HEADER_RE.match(line.strip()) for line in answer.splitlines())


def _strip_source_dump_sections(answer: str) -> str:
    if not answer:
        return answer

    kept: list[str] = []
    skipping_source_block = False
    skipping_image_evidence = False
    changed = False

    for raw_line in answer.splitlines():
        line = raw_line.strip()

        if _SOURCE_DUMP_HEADER_RE.match(line):
            skipping_source_block = True
            skipping_image_evidence = False
            changed = True
            continue

        if line == "IMAGE_QUERY_EVIDENCE:":
            skipping_image_evidence = True
            skipping_source_block = False
            changed = True
            continue

        if skipping_source_block:
            changed = True
            if not line:
                skipping_source_block = False
            continue

        if skipping_image_evidence:
            changed = True
            if not line:
                skipping_image_evidence = False
            continue

        kept.append(raw_line)

    if not changed:
        return answer

    cleaned = "\n".join(kept)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def _build_summary_rewrite_prompt(user_text: str, answer: str) -> str:
    return (
        "请把下面这个 RAG 回答改写成简洁的综合答案。\n"
        "要求：\n"
        "1. 删除“参考原文”“原文如下”“检索片段”等原文展示段落。\n"
        "2. 保留答案中的事实、数字、名称和引用编号。\n"
        "3. 用自己的话总结，不要大段照抄。\n"
        "4. 输出 1-2 句直接结论 + 2-5 个要点。\n\n"
        f"用户问题：\n{user_text}\n\n"
        f"待改写回答：\n{answer}"
    )


def _rewrite_verbatim_answer_if_needed(answer_model, user_text: str, answer: str) -> str:
    answer = _strip_source_dump_sections(answer)
    if not _needs_summary_rewrite(answer):
        return answer
    try:
        prompt = _build_summary_rewrite_prompt(user_text, answer)
        response = _invoke_model_with_timeout(
            answer_model,
            [SystemMessage(content=prompt)],
            timeout_seconds=DIRECT_RAG_REWRITE_TIMEOUT,
        )
        rewritten = (response.content or "").strip()
        return _strip_source_dump_sections(rewritten) or answer
    except Exception:
        return answer


def _build_previous_answer_rewrite_prompt(user_text: str, previous_answer: str) -> str:
    return (
        "请按用户最新要求改写上一条回答，不要重新检索知识库，不要引入新事实。\n"
        "如果用户要求中文，请完整翻译/改写为中文，保留原有事实、数字、名称和引用编号。\n"
        "删除冗余原文展示段落，保持简洁。\n\n"
        f"用户最新要求：\n{user_text}\n\n"
        f"上一条回答：\n{previous_answer}"
    )


def _rewrite_previous_answer_sync(user_text: str, previous_answer: str) -> str:
    if not previous_answer:
        return "没有可改写的上一条回答。"
    model = _select_direct_answer_models()[0]
    prompt = _build_previous_answer_rewrite_prompt(user_text, previous_answer)
    try:
        response = _invoke_model_with_timeout(
            model,
            [SystemMessage(content=prompt)],
            timeout_seconds=DIRECT_RAG_REWRITE_TIMEOUT,
        )
        rewritten = (response.content or "").strip()
        return _strip_source_dump_sections(rewritten) or previous_answer
    except Exception:
        return previous_answer


def _invoke_model_with_timeout(model, messages: list, timeout_seconds: float):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(model.invoke, messages)
    try:
        return future.result(timeout=timeout_seconds)
    except concurrent.futures.TimeoutError:
        future.cancel()
        raise TimeoutError(f"model invocation timed out after {timeout_seconds}s")
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


async def _run_in_executor_with_context(func):
    loop = asyncio.get_running_loop()
    ctx = contextvars.copy_context()
    return await loop.run_in_executor(None, lambda: ctx.run(func))


def _build_extractive_fallback_answer(user_text: str, docs: list[dict]) -> str:
    lines = ["生成模型响应超时，以下为基于检索片段的简要结果："]
    for i, doc in enumerate(docs[:3], 1):
        text = " ".join((doc.get("text") or "").split())
        if len(text) > 220:
            text = text[:220].rstrip() + "..."
        source = doc.get("filename", "Unknown")
        page = doc.get("page_number", "N/A")
        lines.append(f"- {text} [{i}]")
        lines.append(f"  来源：{source} p{page}")
    return "\n".join(lines)


def _select_direct_answer_models():
    main_model = runtime.model
    fast_model = runtime.fast_model or runtime.model
    if DIRECT_RAG_ANSWER_MODEL in ("main", "model", "primary"):
        primary = main_model or fast_model
        fallback = fast_model if fast_model is not primary else None
        return primary, fallback
    primary = fast_model or main_model
    fallback = main_model if main_model is not primary else None
    return primary, fallback


def _generate_direct_answer_with_fallback(prompt: str, user_text: str, docs: list[dict]):
    primary_model, fallback_model = _select_direct_answer_models()

    emit_rag_step("📝", "正在生成答案...", f"超时阈值 {DIRECT_RAG_ANSWER_TIMEOUT:.0f}s")
    try:
        return _invoke_model_with_timeout(
            primary_model,
            [SystemMessage(content=prompt)],
            timeout_seconds=DIRECT_RAG_ANSWER_TIMEOUT,
        ), primary_model
    except Exception as exc:
        if fallback_model is not None:
            emit_rag_step("⚠️", "主模型生成失败，切换快速模型", str(exc)[:120])
            try:
                return _invoke_model_with_timeout(
                    fallback_model,
                    [SystemMessage(content=prompt)],
                    timeout_seconds=DIRECT_RAG_ANSWER_TIMEOUT,
                ), fallback_model
            except Exception as fallback_exc:
                emit_rag_step("⚠️", "快速模型生成失败，使用片段摘要", str(fallback_exc)[:120])
        else:
            emit_rag_step("⚠️", "模型生成失败，使用片段摘要", str(exc)[:120])
        return type("FallbackResponse", (), {"content": _build_extractive_fallback_answer(user_text, docs)})(), None


def _direct_rag_answer_sync(
    user_text: str,
    query_image_path: str | None = None,
    answer_user_text: str | None = None,
) -> tuple[str, dict | None]:
    rag_result = run_rag_graph(user_text, query_image_path=query_image_path)
    docs = rag_result.get("docs", []) if isinstance(rag_result, dict) else []
    rag_trace = rag_result.get("rag_trace", {}) if isinstance(rag_result, dict) else {}
    record_rag_context(rag_trace)

    if not docs:
        return "文档未提供相关信息，无法回答此问题。建议补充相关文档。", rag_trace

    context = _format_direct_rag_docs(docs)
    image_evidence = format_image_evidence_for_prompt(rag_trace)
    prompt_user_text = answer_user_text or user_text
    if answer_user_text and answer_user_text.strip() != user_text.strip():
        prompt_user_text = (
            f"{answer_user_text}\n\n"
            f"Standalone retrieval query used for this follow-up:\n{user_text}"
        )
    prompt = _build_direct_rag_prompt(prompt_user_text, context, image_evidence)
    response, answer_model = _generate_direct_answer_with_fallback(prompt, prompt_user_text, docs)
    answer = (response.content or "").strip()
    if answer_model is None:
        return answer, rag_trace
    return _rewrite_verbatim_answer_if_needed(answer_model, prompt_user_text, answer), rag_trace


def _build_context_messages(
    messages: list,
    persistent_note: str,
    user_text: str,
    has_query_image: bool = False,
) -> list:
    short_term = messages[-CONTEXT_WINDOW_MESSAGES:] if len(messages) > CONTEXT_WINDOW_MESSAGES else messages
    context_messages: list = []
    if persistent_note:
        context_messages.append(
            SystemMessage(
                content=(
                    "【对话持久化笔记（你的工作记忆）】\n"
                    f"{persistent_note}\n"
                    "请参考以上笔记保持对话连贯性，避免重复回答已解决的问题。"
                )
            )
        )
    context_messages.extend(short_term)
    if has_query_image:
        context_messages.append(
            SystemMessage(
                content=(
                    "当前用户本轮消息附带了一张图片。你不能直接视觉读取图片，"
                    "但 search_knowledge_base 工具会自动使用这张图片的隐藏向量检索上下文。"
                    "对于本轮问题，你必须先调用 search_knowledge_base；不要回答“无法查看图片”，"
                    "也不要要求用户重新描述图片。工具返回的 IMAGE_QUERY_EVIDENCE 是图片检索证据，"
                    "你必须优先用它判断图片可能出自哪个文档、页面或附近内容。"
                )
            )
        )
    context_messages.append(HumanMessage(content=user_text))
    return context_messages


async def update_persistent_note(
    current_note: str,
    user_text: str,
    ai_response: str,
) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: _update_persistent_note_sync(current_note, user_text, ai_response),
    )


def _generate_session_title_sync(user_text: str) -> str:
    try:
        prompt = (
            "请根据用户的首次提问，生成一个简短的对话标题（控制在 10 个字以内，不要标点）。\n"
            f"用户提问：{user_text}"
        )
        runtime._ensure_agent_initialized()
        res = runtime.fast_model.invoke([SystemMessage(content=prompt)])
        title = (res.content or "").strip().strip('"').strip("。")
        return title or "新会话"
    except Exception as e:
        print(f"Title generation error: {e}")
        return "新会话"


async def generate_session_title(user_text: str) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: _generate_session_title_sync(user_text))


def _update_persistent_note_sync(current_note: str, user_text: str, ai_response: str) -> str:
    try:
        # 使用 AUX_MODEL 压缩长上下文笔记
        from backend.rag.pipeline import _get_aux_model as _aux

        aux = _aux()
        if aux is None:
            runtime._ensure_agent_initialized()
            aux = runtime.fast_model
        note_len = len(current_note or "")
        compress_hint = ""
        if note_len > 400:
            compress_hint = (
                "⚠️ 现有笔记已超出 400 字，请大幅压缩：合并同类信息、删除非关键细节、"
                "只保留核心结论与尚未解决的问题。控制在 300 字以内。\n"
            )
        prompt = (
            "你是一个【Context Manager Agent】(上下文管理器)，负责维护多轮对话中的「持久化笔记」。\n"
            "笔记是模型在有限上下文窗口下的长效工作记忆，记录已解决的问题与关键事实。\n\n"
            "更新规则：\n"
            "1. 将新信息与现有笔记智能合并，不要简单拼接。\n"
            "2. 过滤噪音，控制在 500 字以内，用简明条目输出。\n"
            "3. 若信息冲突，保留最可靠或最新版本。\n"
            "4. 当笔记过长时，主动压缩旧信息，为新信息腾出空间。\n\n"
            f"{compress_hint}"
            f"▼ 现有笔记：\n{current_note if current_note else '无'}\n\n"
            f"▼ 最新一轮对话：\n用户：{user_text}\nAI：{ai_response}\n\n"
            "请直接输出更新后的笔记（纯文本，不要解释或 Markdown 代码块）："
        )
        res = aux.invoke([SystemMessage(content=prompt)])
        return (res.content or "").strip()
    except Exception as e:
        print(f"Context Manager Error: {e}")
        return current_note


def chat_with_agent(
    user_text: str,
    user_id: str = "default_user",
    session_id: str = "default_session",
    query_image_path: str | None = None,
):
    runtime._ensure_agent_initialized()
    messages, metadata = storage.load_with_meta(user_id, session_id)
    persistent_note = metadata.get("persistent_note", "")
    is_first_message = len(messages) == 0

    get_last_rag_context(clear=True)
    reset_knowledge_tool_calls()
    image_token = set_query_image_path(query_image_path)

    try:
        previous_answer = _last_ai_message_content(messages)
        effective_query = _build_followup_query(user_text, metadata, previous_answer, persistent_note)
        context_messages = _build_context_messages(
            messages,
            persistent_note,
            user_text,
            has_query_image=bool(query_image_path),
        )
        messages.append(HumanMessage(content=user_text))
        storage.save(user_id, session_id, messages)

        if _is_rewrite_previous_answer_request(user_text):
            response_content = _rewrite_previous_answer_sync(user_text, previous_answer)
            result = AIMessage(content=response_content)
        elif not _should_use_agent_tools(user_text, query_image_path):
            if effective_query != user_text:
                emit_rag_step(
                    "\U0001f9ed",
                    "\u68c0\u6d4b\u5230\u4e0a\u4e0b\u6587\u8ffd\u95ee",
                    f"\u6539\u5199\u68c0\u7d22\u67e5\u8be2: {effective_query}",
                )
            response_content, rag_trace = _direct_rag_answer_sync(
                effective_query,
                query_image_path,
                answer_user_text=user_text if effective_query != user_text else None,
            )
            record_rag_context(rag_trace)
            result = AIMessage(content=response_content)
        else:
            try:
                result = runtime.agent.invoke(
                    {"messages": context_messages},
                    config={"recursion_limit": 8},
                )
            except Exception as e:
                if not _looks_like_json_parser_error(e):
                    raise
                response_content, rag_trace = _direct_rag_answer_sync(
                    effective_query,
                    query_image_path,
                    answer_user_text=user_text if effective_query != user_text else None,
                )
                record_rag_context(rag_trace)
                result = AIMessage(content=response_content)
    finally:
        set_rag_step_queue(None)
        reset_query_image_path(image_token)

    response_content = ""
    if isinstance(result, dict):
        if "output" in result:
            response_content = result["output"]
        elif "messages" in result and result["messages"]:
            msg = result["messages"][-1]
            response_content = getattr(msg, "content", str(msg))
        else:
            response_content = str(result)
    elif hasattr(result, "content"):
        response_content = result.content
    else:
        response_content = str(result)

    messages.append(AIMessage(content=response_content))

    rag_context = get_last_rag_context(clear=True)
    rag_trace = rag_context.get("rag_trace") if rag_context else None

    # 幻觉检测：验证回答是否有文档支撑
    hallucination_info = {"verdict": "supported", "unsupported_claims": ""}
    if rag_trace and rag_trace.get("retrieved_chunks"):
        docs_text_for_verify = "\n\n".join(
            c.get("text", "") for c in rag_trace.get("retrieved_chunks", [])
        )
        hallucination_info = verify_answer_against_docs(response_content, docs_text_for_verify)

    if rag_trace:
        rag_trace["hallucination"] = hallucination_info

    save_meta = dict(metadata)
    if is_first_message:
        save_meta["title"] = _generate_session_title_sync(user_text)
    last_rag_topic = _build_last_rag_topic(
        user_text,
        effective_query,
        rag_trace,
        response_content,
    )
    if last_rag_topic:
        save_meta["last_rag_topic"] = last_rag_topic
    save_meta["persistent_note"] = _update_persistent_note_sync(
        persistent_note, user_text, response_content
    )

    extra_message_data = [None] * (len(messages) - 1) + [{"rag_trace": rag_trace}]
    storage.save(
        user_id,
        session_id,
        messages,
        metadata=save_meta,
        extra_message_data=extra_message_data,
    )

    return {
        "response": response_content,
        "rag_trace": rag_trace,
    }


async def chat_with_agent_stream(
    user_text: str,
    user_id: str = "default_user",
    session_id: str = "default_session",
    query_image_path: str | None = None,
):
    runtime._ensure_agent_initialized()
    messages, metadata = storage.load_with_meta(user_id, session_id)
    persistent_note = metadata.get("persistent_note", "")
    is_first_message = len(messages) == 0

    get_last_rag_context(clear=True)
    reset_knowledge_tool_calls()
    image_token = set_query_image_path(query_image_path)

    output_queue = asyncio.Queue()

    class _RagStepProxy:
        def put_nowait(self, step):
            output_queue.put_nowait({"type": "rag_step", "step": step})

    set_rag_step_queue(_RagStepProxy())
    previous_answer = _last_ai_message_content(messages)
    effective_query = _build_followup_query(user_text, metadata, previous_answer, persistent_note)

    context_messages = _build_context_messages(
        messages,
        persistent_note,
        user_text,
        has_query_image=bool(query_image_path),
    )
    messages.append(HumanMessage(content=user_text))
    storage.save(user_id, session_id, messages)

    title_task = None
    if is_first_message:

        def _on_title_done(fut):
            try:
                title = fut.result()
                output_queue.put_nowait(
                    {"type": "session_title", "title": title, "session_id": session_id}
                )
            except Exception as e:
                print(f"Title task error: {e}")

        title_task = asyncio.create_task(generate_session_title(user_text))
        title_task.add_done_callback(_on_title_done)

    full_response = ""
    use_agent_tools = _should_use_agent_tools(user_text, query_image_path)
    rewrite_previous = _is_rewrite_previous_answer_request(user_text)

    async def _agent_worker():
        nonlocal full_response
        try:
            if rewrite_previous:
                rewritten = await _run_in_executor_with_context(
                    lambda: _rewrite_previous_answer_sync(user_text, previous_answer),
                )
                if rewritten:
                    full_response += rewritten
                    await output_queue.put({"type": "content", "content": rewritten})
                return

            if not use_agent_tools:
                if effective_query != user_text:
                    emit_rag_step(
                        "\U0001f9ed",
                        "\u68c0\u6d4b\u5230\u4e0a\u4e0b\u6587\u8ffd\u95ee",
                        f"\u6539\u5199\u68c0\u7d22\u67e5\u8be2: {effective_query}",
                    )
                direct_result = await _run_in_executor_with_context(
                    lambda: _direct_rag_answer_sync(
                        effective_query,
                        query_image_path,
                        answer_user_text=user_text if effective_query != user_text else None,
                    ),
                )
                direct_response, rag_trace_direct = direct_result if isinstance(direct_result, tuple) else (direct_result, None)
                record_rag_context(rag_trace_direct)
                if direct_response:
                    full_response += direct_response
                    await output_queue.put({"type": "content", "content": direct_response})
                return

            async for msg, _metadata in runtime.agent.astream(
                {"messages": context_messages},
                stream_mode="messages",
                config={"recursion_limit": 8},
            ):
                if not isinstance(msg, AIMessageChunk):
                    continue
                if getattr(msg, "tool_call_chunks", None):
                    continue

                content = ""
                if isinstance(msg.content, str):
                    content = msg.content
                elif isinstance(msg.content, list):
                    for block in msg.content:
                        if isinstance(block, str):
                            content += block
                        elif isinstance(block, dict) and block.get("type") == "text":
                            content += block.get("text", "")

                if content:
                    full_response += content
                    await output_queue.put({"type": "content", "content": content})
        except Exception as e:
            if not _looks_like_json_parser_error(e) or full_response:
                await output_queue.put({"type": "error", "content": str(e)})
                return
            try:
                fallback_result = await _run_in_executor_with_context(
                    lambda: _direct_rag_answer_sync(
                        effective_query,
                        query_image_path,
                        answer_user_text=user_text if effective_query != user_text else None,
                    ),
                )
                fallback_response, rag_trace_fallback = fallback_result if isinstance(fallback_result, tuple) else (fallback_result, None)
                record_rag_context(rag_trace_fallback)
                if fallback_response:
                    full_response += fallback_response
                    await output_queue.put({"type": "content", "content": fallback_response})
            except Exception as fallback_error:
                await output_queue.put({"type": "error", "content": str(fallback_error)})
        finally:
            await output_queue.put(None)

    agent_task = asyncio.create_task(_agent_worker())

    try:
        while True:
            event = await output_queue.get()
            if event is None:
                break
            yield f"data: {json.dumps(event)}\n\n"
    except GeneratorExit:
        agent_task.cancel()
        try:
            await agent_task
        except asyncio.CancelledError:
            pass
        raise
    finally:
        reset_query_image_path(image_token)
        set_rag_step_queue(None)
        if not agent_task.done():
            agent_task.cancel()

    rag_context = get_last_rag_context(clear=True)
    rag_trace = rag_context.get("rag_trace") if rag_context else None

    # 幻觉检测
    hallucination_info = {"verdict": "supported", "unsupported_claims": ""}
    if rag_trace and rag_trace.get("retrieved_chunks"):
        docs_text_for_verify = "\n\n".join(
            c.get("text", "") for c in rag_trace.get("retrieved_chunks", [])
        )
        hallucination_info = verify_answer_against_docs(full_response, docs_text_for_verify)

    if rag_trace:
        rag_trace["hallucination"] = hallucination_info
        yield f"data: {json.dumps({'type': 'trace', 'rag_trace': rag_trace})}\n\n"

    yield "data: [DONE]\n\n"

    save_meta = dict(metadata)
    if is_first_message and title_task is not None:
        try:
            save_meta["title"] = await title_task
        except Exception:
            pass

    last_rag_topic = _build_last_rag_topic(
        user_text,
        effective_query,
        rag_trace,
        full_response,
    )
    if last_rag_topic:
        save_meta["last_rag_topic"] = last_rag_topic

    try:
        save_meta["persistent_note"] = await update_persistent_note(
            persistent_note, user_text, full_response
        )
    except Exception as e:
        print(f"Update persistent note error: {e}")

    messages.append(AIMessage(content=full_response))
    extra_message_data = [None] * (len(messages) - 1) + [{"rag_trace": rag_trace}]
    storage.save(
        user_id,
        session_id,
        messages,
        metadata=save_meta,
        extra_message_data=extra_message_data,
    )
