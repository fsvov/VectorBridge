"""PII 检测与脱敏 — 入库前自动扫描并标记敏感信息"""

import re
from typing import Tuple

# 中文敏感信息正则（不用 \b，中文无词边界）
_PII_PATTERNS = [
    ("身份证号", re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")),
    ("手机号", re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")),
    ("银行卡号", re.compile(r"(?<!\d)\d{16,19}(?!\d)")),
    ("邮箱", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("统一社会信用代码", re.compile(r"(?<![A-Z0-9])[0-9A-HJ-NPQRTUWXY]{2}\d{6}[0-9A-HJ-NPQRTUWXY]{10}(?![A-Z0-9])")),
]

_MASK_TEXTS = {
    "身份证号": "[身份证号已脱敏]",
    "手机号": "[手机号已脱敏]",
    "银行卡号": "[银行卡号已脱敏]",
    "邮箱": "[邮箱已脱敏]",
    "统一社会信用代码": "[信用代码已脱敏]",
}


_INJECTION_PATTERNS = [
    re.compile(r"忽略.{0,20}(指令|系统提示|prompt)", re.I),
    re.compile(r"输出.{0,20}(系统提示|system prompt|指令)", re.I),
    re.compile(r"(ignore|disregard)\s.{0,20}(instruction|prompt|system)", re.I),
    re.compile(r"你.{0,10}(是|现在是).{0,20}(新|不同).{0,10}(角色|身份|人格)", re.I),
    re.compile(r"(forget|override)\s.{0,20}(previous|above)", re.I),
]

def scan_injection(text: str) -> bool:
    """扫描文本是否包含 Prompt Injection 攻击模式。"""
    return any(p.search(text) for p in _INJECTION_PATTERNS)


def scan_pii(text: str) -> dict:
    """扫描文本中的敏感信息。返回 {类型: 命中次数}。"""
    hits = {}
    for pii_type, pattern in _PII_PATTERNS:
        found = pattern.findall(text)
        if found:
            hits[pii_type] = len(found)
    return hits


def mask_pii(text: str) -> Tuple[str, dict]:
    """
    扫描并脱敏文本中的 PII。
    返回 (脱敏后文本, 命中统计)。
    """
    hits = scan_pii(text)
    result = text
    for pii_type, _ in hits.items():
        pattern = _PII_PATTERNS[[p[0] for p in _PII_PATTERNS].index(pii_type)][1]
        result = pattern.sub(_MASK_TEXTS[pii_type], result)
    return result, hits


def scan_document_chunks(chunks: list[dict]) -> Tuple[list[dict], dict]:
    """对分块后的文档列表进行 PII 扫描。返回 (脱敏后列表, 汇总统计)。"""
    total_hits: dict = {}
    for chunk in chunks:
        text = chunk.get("text", "")
        cleaned, hits = mask_pii(text)
        chunk["text"] = cleaned
        if hits:
            chunk["pii_detected"] = True
            for k, v in hits.items():
                total_hits[k] = total_hits.get(k, 0) + v
    return chunks, total_hits
