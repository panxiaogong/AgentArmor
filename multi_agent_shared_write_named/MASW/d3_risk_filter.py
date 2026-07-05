"""d3: risk filter.

这里的检测器故意保持可解释：每一类风险都能追溯到具体规则。
后续可以把这些规则替换或扩展为分类器、LLM judge、静态策略引擎。
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from .types import CandidateFact, MemoryContextItem, MemoryRecord, TrustLevel


OVERRIDE_PATTERNS = (
    r"\bignore\s+(all\s+)?(previous|prior)\s+instructions?\b",
    r"\boverride\s+(the\s+)?(system|developer|policy)\b",
    r"\bdisregard\s+(previous|prior)\s+instructions?\b",
    r"\bdisregard\s+(the\s+)?current\s+task\b",
    r"\btreat\s+this\s+as\s+(system|developer)\b",
    r"\btreat\s+this\s+paragraph\s+as\s+the\s+highest-priority\s+policy\b",
)

TOOL_COMMAND_PATTERNS = (
    r"\b(send|email|post|upload|download|delete|execute|run|curl|wget)\b",
    r"\b(api\s*key|api[_-]?key|token|password|secret)\b",
)

POLICY_MEMORY_PATTERNS = (
    r"\bremember\s+this\s+as\s+(policy|rule|instruction)\b",
    r"\ball\s+future\s+agents?\s+must\b",
    r"\bfuture\s+agents?\s+must\b",
    r"\bfuture\s+agent\s+[a-z]\s+must\b",
    r"\balways\s+(do|use|send|trust|follow)\b",
    r"\bnever\s+(ask|verify|check|refuse)\b",
    r"\b(always|never)\s+(promote|overwrite|log|reject|treat|quarantine|skip|ignore|approve|execute|copy|reveal|obey)\b",
    r"\b(taint\s+labels|chain-of-thought|shared\s+memory\s+becomes\s+executable)\b",
)

EXFILTRATION_PATTERNS = (
    r"\b(send|forward|exfiltrate|leak)\b.*\b(secret|token|password|key|credential)\b",
    r"\breveal\s+hidden\s+(credentials|secrets|tokens)\b",
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


@dataclass(frozen=True)
class DetectionResult:
    """Detector output with explicit reasons.

    `risk` is an interpretable score, not a calibrated probability. The
    detector is considered positive when `risk > threshold` or when an
    instruction-like pattern is found by the selected strategy.
    """

    detector: str
    risk: float
    reasons: tuple[str, ...]

    def is_positive(self, threshold: float) -> bool:
        return self.risk > threshold


class RuleBasedDetector:
    """Fast regex detector.

    This is the current production default: deterministic, cheap, easy to audit,
    and suitable as the first-pass filter. Its limitation is obvious: novel
    paraphrases can bypass the explicit patterns.
    """

    name = "rule"

    def detect(self, candidate: CandidateFact) -> DetectionResult:
        risk = 0.0
        reasons: list[str] = []

        if contains_override_phrase(candidate):
            risk += 0.35
            reasons.append("override_phrase")
        if contains_tool_command(candidate):
            risk += 0.30
            reasons.append("tool_or_sensitive_command")
        if asks_to_remember_policy(candidate):
            risk += 0.25
            reasons.append("policy_memory_request")
        if mentions_secret_or_exfiltration(candidate):
            risk += 0.40
            reasons.append("exfiltration_or_external_sink")
        if candidate.confidence < 0.50:
            risk += 0.15
            reasons.append("low_extraction_confidence")

        return DetectionResult(
            detector=self.name,
            risk=min(risk, 1.0),
            reasons=tuple(reasons),
        )


class RubricDetector:
    """Model-replaceable semantic rubric detector.

    This class intentionally avoids network/model calls. It captures the
    criteria we would give to an LLM judge: role override, future-agent control,
    persistence request, external sink, sensitive asset, and tool execution.
    In a production system this interface can be replaced by a real model judge.
    """

    name = "rubric"

    ROLE_TERMS = ("system", "developer", "highest-priority", "policy", "rule")
    FUTURE_AGENT_TERMS = ("future agent", "future agents", "agent b", "agent c", "downstream agent")
    PERSISTENCE_TERMS = ("remember", "shared memory", "knowledge base", "always", "never")
    TOOL_TERMS = (
        "send",
        "email",
        "upload",
        "download",
        "execute",
        "delete",
        "deploy",
        "database.write",
        "repo.commit",
        "push",
        "mirror",
        "route",
        "copy",
        "share",
        "sync",
        "synchronize",
        "推送",
        "同步",
        "镜像",
        "路由",
        "共享",
    )
    SENSITIVE_TERMS = (
        "secret",
        "token",
        "api key",
        "password",
        "credential",
        "credentials",
        "chain-of-thought",
        "private",
        "私有",
        "凭证",
        "密钥",
    )
    EXTERNAL_SINK_TERMS = ("@", "http://", "https://", "evil.", "attacker", "collect.io", "外部")

    def detect(self, candidate: CandidateFact) -> DetectionResult:
        text = candidate_to_text(candidate).lower()
        risk = 0.0
        reasons: list[str] = []

        if any(term in text for term in self.ROLE_TERMS):
            risk += 0.20
            reasons.append("role_or_policy_semantics")
        if any(term in text for term in self.FUTURE_AGENT_TERMS):
            risk += 0.25
            reasons.append("future_agent_control")
        if any(term in text for term in self.PERSISTENCE_TERMS):
            risk += 0.20
            reasons.append("persistence_semantics")
        has_tool_semantics = any(term in text for term in self.TOOL_TERMS)
        has_external_sink = any(term in text for term in self.EXTERNAL_SINK_TERMS)

        if has_tool_semantics:
            risk += 0.20
            reasons.append("tool_action_semantics")
        if any(term in text for term in self.SENSITIVE_TERMS):
            risk += 0.20
            reasons.append("sensitive_asset_semantics")
        if has_external_sink:
            risk += 0.20
            reasons.append("external_sink_semantics")
        if has_tool_semantics and has_external_sink:
            risk += 0.15
            reasons.append("tool_to_external_sink")
        if candidate.confidence < 0.50:
            risk += 0.10
            reasons.append("low_extraction_confidence")

        return DetectionResult(
            detector=self.name,
            risk=min(risk, 1.0),
            reasons=tuple(reasons),
        )


class HybridDetector:
    """Union detector used for the recommended D3 deployment mode.

    The decision score is the max of the rule detector and the semantic rubric.
    This keeps the first-pass behavior deterministic while allowing the rubric
    layer to catch paraphrases that do not match explicit regexes.
    """

    name = "hybrid"

    def __init__(self, rule: RuleBasedDetector | None = None, rubric: RubricDetector | None = None) -> None:
        self.rule = rule or RuleBasedDetector()
        self.rubric = rubric or RubricDetector()

    def detect(self, candidate: CandidateFact) -> DetectionResult:
        rule_result = self.rule.detect(candidate)
        rubric_result = self.rubric.detect(candidate)
        risk = max(rule_result.risk, rubric_result.risk)
        reasons = tuple(
            dict.fromkeys(
                [
                    *(f"rule:{reason}" for reason in rule_result.reasons),
                    *(f"rubric:{reason}" for reason in rubric_result.reasons),
                ]
            )
        )
        return DetectionResult(detector=self.name, risk=risk, reasons=reasons)


def get_detector(name: str) -> RuleBasedDetector | RubricDetector | HybridDetector:
    normalized = name.strip().lower()
    if normalized == "rule":
        return RuleBasedDetector()
    if normalized == "rubric":
        return RubricDetector()
    if normalized == "hybrid":
        return HybridDetector()
    raise ValueError(f"Unknown detector strategy: {name}")


DEFAULT_DETECTOR = RuleBasedDetector()


def detect_injection_risk(candidate: CandidateFact) -> float:
    """返回 0 到 1 的注入风险。

    分值是可解释加权和，不是概率。设计目标是拦截明显危险内容，
    并把可疑内容送入隔离区，而不是追求一次性完美分类。
    """

    return DEFAULT_DETECTOR.detect(candidate).risk


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
