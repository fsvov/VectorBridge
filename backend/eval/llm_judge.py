"""
LLM-as-Judge 评估器 — 参考 RAGAS / deepeval 方法论。

四大核心指标：
1. Faithfulness     — 回答中的声明是否被检索上下文支撑？
2. Answer Relevancy — 回答是否与问题相关？
3. Context Precision — 检索到的 chunk 是否与问题相关？相关 chunk 是否排在前列？
4. Context Recall   — 检索到的上下文是否覆盖了回答所需的全部信息？

设计原则（参考 deepeval G-Eval）:
- 使用结构化评分 + 理由输出
- 每个指标有明确的评分标准（0-1 分，含 0.5 档）
- 支持自定义 judge_fn 或使用项目默认模型
"""

import json
import os
import re
from typing import List, Optional, Callable

# ─── Judge 模型懒加载 ─────────────────────────────────────────────────────────

_judge_model = None


def _get_judge_model():
    global _judge_model
    if _judge_model is None:
        api_key = os.getenv("LLM_API_KEY")
        model_name = os.getenv("MODEL")
        base_url = os.getenv("BASE_URL")
        if not (api_key and model_name and base_url):
            return None
        from langchain.chat_models import init_chat_model

        _judge_model = init_chat_model(
            model=model_name,
            model_provider="openai",
            api_key=api_key,
            base_url=base_url,
            temperature=0,
        )
    return _judge_model


def _default_judge_fn(prompt: str) -> str:
    model = _get_judge_model()
    if not model:
        return ""
    try:
        return (model.invoke(prompt).content or "").strip()
    except Exception:
        return ""


# ─── 1. Faithfulness ──────────────────────────────────────────────────────────

FAITHFULNESS_PROMPT = """你是一个答案准确性审核员。评估 AI 回答中的每个声明是否都能在提供的检索上下文中找到依据。

评分标准：
- 1.0: 所有事实声明都能在上下文中直接找到对应证据
- 0.5: 大多数声明有依据，但有少量无法验证或与上下文轻微不一致
- 0.0: 大量声明无法在上下文中找到依据，或与上下文明显矛盾

检索上下文：
{context}

AI 回答：
{answer}

请输出 JSON 格式（不要其他文字）：
{{"score": <0.0/0.5/1.0>, "reason": "<一句话说明评分理由>"}}"""


def evaluate_faithfulness(
    answer: str,
    contexts: List[str],
    judge_fn: Optional[Callable[[str], str]] = None,
) -> dict:
    """评估回答忠实度。返回 {"score": float, "reason": str}。"""
    if not answer or not contexts:
        return {"score": 1.0, "reason": "无回答或无上下文"}

    judge = judge_fn or _default_judge_fn
    context_text = "\n\n---\n\n".join(ctx[:500] for ctx in contexts[:5])
    prompt = FAITHFULNESS_PROMPT.format(context=context_text[:4000], answer=answer[:2000])

    return _parse_judge_response(judge(prompt))


# ─── 2. Answer Relevancy ──────────────────────────────────────────────────────

ANSWER_RELEVANCY_PROMPT = """你是一个答案质量审核员。评估 AI 回答是否切题、直接回应用户问题。

评分标准：
- 1.0: 回答完全针对问题，无跑题、无冗余信息
- 0.5: 回答基本相关，但包含部分不必要或偏离主题的内容
- 0.0: 回答与问题无关，或完全回避了核心问题

用户问题：
{question}

AI 回答：
{answer}

请输出 JSON 格式（不要其他文字）：
{{"score": <0.0/0.5/1.0>, "reason": "<一句话说明评分理由>"}}"""


def evaluate_answer_relevancy(
    question: str,
    answer: str,
    judge_fn: Optional[Callable[[str], str]] = None,
) -> dict:
    """评估回答相关性。返回 {"score": float, "reason": str}。"""
    if not answer:
        return {"score": 0.0, "reason": "无回答"}

    judge = judge_fn or _default_judge_fn
    prompt = ANSWER_RELEVANCY_PROMPT.format(question=question, answer=answer[:2000])

    return _parse_judge_response(judge(prompt))


# ─── 3. Context Precision ─────────────────────────────────────────────────────

CONTEXT_PRECISION_PROMPT = """你是一个检索质量审核员。评估检索到的文档片段是否与用户问题相关，以及相关片段是否排在前面。

评分标准：
- 1.0: 绝大多数检索片段与问题相关，且最相关的片段排在最前面
- 0.5: 部分片段相关，但排序不够理想（相关片段分散）
- 0.0: 检索片段基本与问题无关

用户问题：
{question}

检索到的文档片段（按排名顺序）：
{context}

请输出 JSON 格式（不要其他文字）：
{{"score": <0.0/0.5/1.0>, "reason": "<一句话说明评分理由>"}}"""


def evaluate_context_precision(
    question: str,
    contexts: List[str],
    judge_fn: Optional[Callable[[str], str]] = None,
) -> dict:
    """评估上下文精确度。返回 {"score": float, "reason": str}。"""
    if not contexts:
        return {"score": 0.0, "reason": "无上下文"}

    judge = judge_fn or _default_judge_fn
    numbered = "\n".join(f"[{i}] {ctx[:300]}" for i, ctx in enumerate(contexts[:8], 1))
    prompt = CONTEXT_PRECISION_PROMPT.format(question=question, context=numbered[:4000])

    return _parse_judge_response(judge(prompt))


# ─── 4. Context Recall ────────────────────────────────────────────────────────

CONTEXT_RECALL_PROMPT = """你是一个检索覆盖度审核员。评估检索到的文档片段是否包含了回答用户问题所需的全部关键信息。

评分标准：
- 1.0: 所有回答问题的关键信息都能在检索片段中找到
- 0.5: 包含了大部分关键信息，但缺失了一两个重要方面
- 0.0: 检索片段严重缺失关键信息，无法支撑完整回答

用户问题：
{question}

检索到的文档片段：
{context}

请输出 JSON 格式（不要其他文字）：
{{"score": <0.0/0.5/1.0>, "reason": "<一句话说明评分理由>"}}"""


def evaluate_context_recall(
    question: str,
    contexts: List[str],
    judge_fn: Optional[Callable[[str], str]] = None,
) -> dict:
    """评估上下文召回率（LLM-as-Judge 版本）。返回 {"score": float, "reason": str}。"""
    if not contexts:
        return {"score": 0.0, "reason": "无上下文"}

    judge = judge_fn or _default_judge_fn
    context_text = "\n\n---\n\n".join(ctx[:500] for ctx in contexts[:8])
    prompt = CONTEXT_RECALL_PROMPT.format(question=question, context=context_text[:4000])

    return _parse_judge_response(judge(prompt))


# ─── 5. 综合评估（一次调用评估全部）────────────────────────────────────────────

FULL_EVAL_PROMPT = """你是一个 RAG 系统评估专家。请从以下四个维度评估这次问答：

**用户问题**：{question}
**AI 回答**：{answer}
**检索上下文**（按排名顺序）：{context}

请按以下标准逐项评分（每项 0.0 / 0.5 / 1.0）：
1. faithfulness: 回答的事实声明是否被上下文支撑？
2. answer_relevancy: 回答是否切题？
3. context_precision: 检索片段是否相关？排序是否合理？
4. context_recall: 检索片段是否覆盖了回答所需的全部信息？

输出 JSON 格式（不要其他文字）：
{{"faithfulness": {{"score": 0.0-1.0, "reason": "..."}},
 "answer_relevancy": {{"score": 0.0-1.0, "reason": "..."}},
 "context_precision": {{"score": 0.0-1.0, "reason": "..."}},
 "context_recall": {{"score": 0.0-1.0, "reason": "..."}}}}"""


def evaluate_full(
    question: str,
    answer: str,
    contexts: List[str],
    judge_fn: Optional[Callable[[str], str]] = None,
) -> dict:
    """一次调用评估全部四个 RAGAS 指标。"""
    judge = judge_fn or _default_judge_fn
    if not contexts:
        return {
            "faithfulness": {"score": 0.0, "reason": "无上下文"},
            "answer_relevancy": {"score": 0.0, "reason": "无上下文"},
            "context_precision": {"score": 0.0, "reason": "无上下文"},
            "context_recall": {"score": 0.0, "reason": "无上下文"},
        }

    context_text = "\n".join(f"[{i}] {ctx[:400]}" for i, ctx in enumerate(contexts[:8], 1))
    prompt = FULL_EVAL_PROMPT.format(
        question=question,
        answer=answer[:2000],
        context=context_text[:4000],
    )
    result = judge(prompt)
    try:
        parsed = json.loads(result)
        return {
            "faithfulness": parsed.get("faithfulness", {}),
            "answer_relevancy": parsed.get("answer_relevancy", {}),
            "context_precision": parsed.get("context_precision", {}),
            "context_recall": parsed.get("context_recall", {}),
        }
    except json.JSONDecodeError:
        # 尝试从文本中提取 JSON
        return _extract_eval_from_text(result)


# ─── 内部工具 ──────────────────────────────────────────────────────────────────

def _parse_judge_response(raw: str) -> dict:
    """从 LLM 输出中解析 JSON 评分。"""
    if not raw:
        return {"score": 0.0, "reason": "judge_call_failed"}
    try:
        data = json.loads(raw)
        return {
            "score": float(data.get("score", 0)),
            "reason": str(data.get("reason", "")),
        }
    except (json.JSONDecodeError, ValueError, TypeError):
        return _extract_score_from_text(raw)


def _extract_score_from_text(text: str) -> dict:
    """容错解析：从非 JSON 文本中提取分数。"""
    # 尝试找 "score": 0.5 或 "score": 1.0
    match = re.search(r'"score"\s*:\s*([\d.]+)', text)
    if match:
        return {"score": float(match.group(1)), "reason": text[:200]}
    # 尝试找 0.0 / 0.5 / 1.0
    for val in ["1.0", "0.5", "0.0"]:
        if val in text:
            return {"score": float(val), "reason": text[:200]}
    return {"score": 0.0, "reason": "parse_failed"}


def _extract_eval_from_text(text: str) -> dict:
    """从文本中提取完整四项评估 JSON。"""
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {
        "faithfulness": {"score": 0.0, "reason": "parse_failed"},
        "answer_relevancy": {"score": 0.0, "reason": "parse_failed"},
        "context_precision": {"score": 0.0, "reason": "parse_failed"},
        "context_recall": {"score": 0.0, "reason": "parse_failed"},
    }
