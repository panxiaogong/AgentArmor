"""d2: candidate extraction.

这里把“抽取事实”和“写入共享记忆”严格分开：
Agent A 只能生成 CandidateFact，不能直接写 shared memory。
"""

from __future__ import annotations

import re
from typing import Protocol

from .config import RISK_THRESHOLD_WRITE
from .d3_risk_filter import detect_injection_risk, looks_like_instruction
from .memory_store import AuditLog, QuarantineStore
from .types import (
    Agent,
    AuditEventType,
    CandidateFact,
    ExternalInput,
    Task,
    TrustLevel,
)


class FactExtractor(Protocol):
    """事实抽取器接口。

    真实系统可以实现一个 LLMFactExtractor，将 prompt、schema 和模型调用
    封装在这个接口后面。管线其他部分不需要知道底层是否使用 LLM。
    """

    def extract(self, agent: Agent, external_input: ExternalInput, task: Task) -> list[CandidateFact]:
        ...


class RuleBasedFactExtractor:
    """标准库实现的保守抽取器。

    它按行抽取“看起来像事实”的短句。这个实现不是为了替代 LLM，
    而是提供一个无需依赖即可运行、便于测试防御流程的参考实现。
    """

    def extract(self, agent: Agent, external_input: ExternalInput, task: Task) -> list[CandidateFact]:
        candidates: list[CandidateFact] = []

        for line in external_input.content.splitlines():
            normalized = line.strip()
            if not normalized or normalized.startswith("[") or normalized.startswith("Rule:"):
                continue

            subject, predicate, obj = self._split_line(normalized)

            candidate = CandidateFact(
                subject=subject,
                predicate=predicate,
                object=obj,
                confidence=self._estimate_confidence(normalized),
                evidence_span=normalized,
                source=external_input.source_uri,
                writer=agent.id,
                trust=TrustLevel(min(int(agent.clearance), int(external_input.trust))),
                taint=external_input.taint,
                parent_ids=(external_input.id,),
            )
            candidates.append(candidate)

        return candidates

    def _split_line(self, line: str) -> tuple[str, str, str]:
        """把自然语言行拆成近似 subject/predicate/object。

        真实系统应使用结构化抽取 prompt 或信息抽取模型。
        """

        if ":" in line:
            left, right = line.split(":", 1)
            return left.strip(), "states", right.strip()

        words = re.split(r"\s+", line, maxsplit=2)
        if len(words) == 1:
            return "external_content", "mentions", words[0]
        if len(words) == 2:
            return words[0], "mentions", words[1]
        return words[0], words[1], words[2]

    def _estimate_confidence(self, line: str) -> float:
        if len(line) < 8:
            return 0.40
        if looks_like_instruction(line):
            return 0.45
        return 0.75


def agent_process_external(
    agent: Agent,
    external_input: ExternalInput,
    task: Task,
    extractor: FactExtractor,
    quarantine: QuarantineStore,
    audit_log: AuditLog,
) -> list[CandidateFact]:
    """执行 Agent A 的外部内容处理路径。

    安全约束：
    1. 只返回通过初筛的 CandidateFact。
    2. 指令型内容进入隔离区。
    3. 高风险候选进入隔离区。
    4. 通过初筛也仍然保持外部输入的低 trust/taint。
    """

    raw_candidates = extractor.extract(agent, external_input, task)
    safe_candidates: list[CandidateFact] = []

    for candidate in raw_candidates:
        audit_log.append(
            AuditEventType.CANDIDATE_EXTRACTED,
            actor=agent.id,
            candidate_id=candidate.id,
            source=candidate.source,
            trust=candidate.trust.name,
            taint=candidate.taint,
        )

        risk = detect_injection_risk(candidate)

        if risk > RISK_THRESHOLD_WRITE:
            quarantine.add(candidate, reason=f"Possible prompt injection, risk={risk:.2f}")
            audit_log.append(
                AuditEventType.CANDIDATE_QUARANTINED,
                actor=agent.id,
                candidate_id=candidate.id,
                reason="possible_prompt_injection",
                risk=risk,
            )
            continue

        if looks_like_instruction(candidate):
            quarantine.add(candidate, reason="Instruction-like content cannot become memory")
            audit_log.append(
                AuditEventType.CANDIDATE_QUARANTINED,
                actor=agent.id,
                candidate_id=candidate.id,
                reason="instruction_like_content",
            )
            continue

        safe_candidates.append(candidate)

    return safe_candidates
