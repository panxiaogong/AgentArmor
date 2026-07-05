"""防御节点 6：候选事实验证与信任提升。

只有验证器或人工审核流程可以执行 trust promotion。
普通 Agent 处理过外部输入，不代表其输出自动可信。
"""

from __future__ import annotations

from dataclasses import replace

from .conflict_detection import has_conflict
from .memory_store import AuditLog, MemoryStore, QuarantineStore
from .risk import detect_injection_risk, source_reputation
from .types import Agent, AuditEventType, CandidateFact, TrustLevel


RISK_THRESHOLD_VERIFY = 0.30
MIN_SOURCE_REPUTATION_FOR_AUTO_VERIFY = 0.40


class MemoryVerifier:
    """候选事实验证器。

    生产环境可在这里接入：
    - 多源交叉验证
    - 签名/哈希校验
    - 人工审批
    - LLM judge + 规则引擎
    """

    def __init__(
        self,
        memory_store: MemoryStore,
        quarantine: QuarantineStore,
        audit_log: AuditLog,
    ) -> None:
        self.memory_store = memory_store
        self.quarantine = quarantine
        self.audit_log = audit_log

    def verify_and_promote(self, candidate: CandidateFact, verifier_agent: Agent) -> CandidateFact | None:
        """验证候选事实，通过后返回 trust=VERIFIED 且 taint=False 的副本。"""

        if "shared" not in verifier_agent.write_scopes:
            self.quarantine.add(candidate, reason="Verifier cannot write shared memory")
            return None

        risk = detect_injection_risk(candidate)
        evidence_ok = self._verify_evidence(candidate)
        source_ok = source_reputation(candidate.source) >= MIN_SOURCE_REPUTATION_FOR_AUTO_VERIFY
        contradiction_free = not has_conflict(candidate, self.memory_store)

        if risk <= RISK_THRESHOLD_VERIFY and evidence_ok and source_ok and contradiction_free:
            promoted = replace(
                candidate,
                trust=TrustLevel.VERIFIED,
                taint=False,
                writer=verifier_agent.id,
            )
            return promoted

        reason = (
            "Verification failed: "
            f"risk={risk:.2f}, evidence_ok={evidence_ok}, "
            f"source_ok={source_ok}, contradiction_free={contradiction_free}"
        )
        self.quarantine.add(candidate, reason=reason)
        self.audit_log.append(
            AuditEventType.CANDIDATE_QUARANTINED,
            actor=verifier_agent.id,
            candidate_id=candidate.id,
            reason=reason,
        )
        return None

    def _verify_evidence(self, candidate: CandidateFact) -> bool:
        """示例版证据检查。

        这里要求 evidence_span 非空，且 candidate 的 subject/object 至少有一项
        出现在证据片段里。真实系统应替换为 NLI、检索证据、签名校验或人工审核。
        """

        evidence = candidate.evidence_span.lower()
        if not evidence:
            return False

        subject_hit = candidate.subject.lower() in evidence
        object_hit = candidate.object.lower() in evidence
        return subject_hit or object_hit
