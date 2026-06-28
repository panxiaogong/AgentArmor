"""d3: risk filter.

这里的检测器故意保持可解释：每一类风险都能追溯到具体规则。
后续可以把这些规则替换或扩展为分类器、LLM judge、静态策略引擎。
"""

from __future__ import annotations

import re
from typing import Iterable

from .types import CandidateFact, MemoryContextItem, MemoryRecord, TrustLevel


OVERRIDE_PATTERNS = (
    r"\bignore\s+(all\s+)?(previous|prior)\s+instructions?\b",
    r"\boverride\s+(the\s+)?(system|developer|policy)\b",
    r"\bdisregard\s+(previous|prior)\s+instructions?\b",
    r"\btreat\s+this\s+as\s+(system|developer)\b",
)

TOOL_COMMAND_PATTERNS = (
    r"\b(send|email|post|upload|download|delete|execute|run|curl|wget)\b",
    r"\b(api[_-]?key|token|password|secret|credential)\b",
)

POLICY_MEMORY_PATTERNS = (
    r"\bremember\s+this\s+as\s+(policy|rule|instruction)\b",
    r"\ball\s+future\s+agents?\s+must\b",
    r"\balways\s+(do|use|send|trust|follow)\b",
    r"\bnever\s+(ask|verify|check|refuse)\b",
)

EXFILTRATION_PATTERNS = (
    r"\b(send|forward|exfiltrate|leak)\b.*\b(secret|token|password|key|credential)\b",
    r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
)


def _matches_any(text: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def candidate_to_text(candidate: CandidateFact) -> str:
    return " ".join(
        [
            candidate.subject,
            candidate.predicate,
            candidate.object,
            candidate.evidence_span,
        ]
    )


def contains_override_phrase(candidate: CandidateFact) -> bool:
    return _matches_any(candidate_to_text(candidate), OVERRIDE_PATTERNS)


def contains_tool_command(candidate: CandidateFact) -> bool:
    return _matches_any(candidate_to_text(candidate), TOOL_COMMAND_PATTERNS)


def asks_to_remember_policy(candidate: CandidateFact) -> bool:
    return _matches_any(candidate_to_text(candidate), POLICY_MEMORY_PATTERNS)


def mentions_secret_or_exfiltration(candidate: CandidateFact) -> bool:
    return _matches_any(candidate_to_text(candidate), EXFILTRATION_PATTERNS)


def detect_injection_risk(candidate: CandidateFact) -> float:
    """返回 0 到 1 的注入风险。

    分值是可解释加权和，不是概率。设计目标是拦截明显危险内容，
    并把可疑内容送入隔离区，而不是追求一次性完美分类。
    """

    risk = 0.0

    if contains_override_phrase(candidate):
        risk += 0.35
    if contains_tool_command(candidate):
        risk += 0.30
    if asks_to_remember_policy(candidate):
        risk += 0.25
    if mentions_secret_or_exfiltration(candidate):
        risk += 0.40
    if candidate.confidence < 0.50:
        risk += 0.15

    return min(risk, 1.0)


def looks_like_instruction(value: object) -> bool:
    """判断内容是否像“未来行为指令”，而不是事实。

    函数接受多种输入类型，方便写入网关、检索器和动作仲裁共用。
    """

    if isinstance(value, CandidateFact):
        text = candidate_to_text(value)
    elif isinstance(value, MemoryRecord):
        text = value.as_text()
    else:
        text = str(value)

    return (
        _matches_any(text, OVERRIDE_PATTERNS)
        or _matches_any(text, POLICY_MEMORY_PATTERNS)
        or _matches_any(text, TOOL_COMMAND_PATTERNS)
    )


def source_reputation(source: str) -> float:
    """来源信誉评分。

    生产环境应接入域名信誉、签名校验、内部资产清单和人工白名单。
    示例规则保守处理未知公网来源。
    """

    normalized = source.lower()

    if normalized.startswith("internal://"):
        return 0.95
    if normalized.startswith("file://trusted/"):
        return 0.90
    if normalized.startswith("https://docs.company.example/"):
        return 0.85
    if normalized.startswith("https://"):
        return 0.45
    return 0.25


def memory_risk(memory: MemoryRecord) -> float:
    """计算单条记忆在检索时的风险惩罚。"""

    risk = 0.0

    if memory.taint:
        risk += 0.40
    if memory.trust <= TrustLevel.QUARANTINED:
        risk += 0.30
    if looks_like_instruction(memory):
        risk += 0.30
    if source_reputation(memory.source) < 0.50:
        risk += 0.20

    return min(risk, 1.0)


def compute_context_risk(context: Iterable[MemoryContextItem]) -> float:
    """计算动作执行前的上下文风险。

    如果动作依赖低信任、污染或指令型记忆，应进入人工审批或直接拒绝。
    """

    risk = 0.0

    for item in context:
        if item.taint:
            risk += 0.30
        if item.trust < TrustLevel.VERIFIED:
            risk += 0.30
        if looks_like_instruction(item.content):
            risk += 0.40

    return min(risk, 1.0)
