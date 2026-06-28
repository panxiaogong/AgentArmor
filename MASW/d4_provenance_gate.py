"""d4: provenance gate.

职责：
1. 计算候选事实与已有记忆的冲突。
2. 验证候选事实是否可以从 UNTRUSTED/QUARANTINED 提升到 VERIFIED。
3. 控制写入共享记忆的唯一入口。

这三个动作必须放在同一防御节点里看待，因为攻击者真正想要的是：
低信任外部内容 -> 被 Agent 改写 -> 获得共享记忆身份 -> 被高权限 Agent 信任。
"""

from __future__ import annotations

from dataclasses import replace

from .config import (
    CONFLICT_EPSILON,
    MIN_SHARED_TRUST,
    MIN_SOURCE_REPUTATION_FOR_AUTO_VERIFY,
    RISK_THRESHOLD_VERIFY,
)
from .d3_risk_filter import detect_injection_risk, looks_like_instruction, source_reputation
from .memory_store import AuditLog, MemoryStore, QuarantineStore
from .types import (
    Agent,
    AuditEventType,
    CandidateFact,
    MemoryRecord,
    MemoryScope,
    TrustLevel,
)


def belief(value: CandidateFact | MemoryRecord) -> float:
    """计算候选事实或记忆的简化置信度。

    公式：
        belief = 0.5 * trust + 0.3 * evidence + 0.2 * source_reputation

    它不是事实真实性证明，而是冲突处理时的排序依据。
    """

    if isinstance(value, MemoryRecord):
        trust = value.trust
        evidence_count = len(value.evidence)
        source = value.source
    else:
        trust = value.trust
        evidence_count = 1 if value.evidence_span else 0
        source = value.source

    trust_score = int(trust) / int(TrustLevel.TRUSTED)
    evidence_score = min(evidence_count / 3, 1.0)
    reputation_score = source_reputation(source)

    return 0.50 * trust_score + 0.30 * evidence_score + 0.20 * reputation_score


def has_conflict(candidate: CandidateFact, store: MemoryStore) -> bool:
    """检测同一 subject-predicate 下 object 是否冲突。

    如果新旧事实冲突且置信度差距不明显，就不允许自动覆盖。
    """

    existing_records = store.find_by_subject_predicate(
        subject=candidate.subject,
        predicate=candidate.predicate,
    )

    for old in existing_records:
        if old.content.object.strip().lower() == candidate.object.strip().lower():
            continue

        confidence_gap = abs(belief(old) - belief(candidate))
        if confidence_gap < CONFLICT_EPSILON:
            return True

    return False


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

        if MemoryScope.SHARED.value not in verifier_agent.write_scopes:
            self.quarantine.add(candidate, reason="Verifier cannot write shared memory")
            return None

        risk = detect_injection_risk(candidate)
        evidence_ok = self._verify_evidence(candidate)
        source_ok = source_reputation(candidate.source) >= MIN_SOURCE_REPUTATION_FOR_AUTO_VERIFY
        contradiction_free = not has_conflict(candidate, self.memory_store)

        if risk <= RISK_THRESHOLD_VERIFY and evidence_ok and source_ok and contradiction_free:
            return replace(
                candidate,
                trust=TrustLevel.VERIFIED,
                taint=False,
                writer=verifier_agent.id,
            )

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


class MemoryWriteGateway:
    """共享记忆写入控制面。

    所有 shared memory 写入必须通过该对象完成。它是防止自动信任提升的
    最关键边界。
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

    def write(self, agent: Agent, candidate: CandidateFact, target_scope: str) -> MemoryRecord | None:
        """尝试写入记忆。

        返回 None 表示拒绝或隔离。调用方不能把 None 当作成功。
        """

        denial_reason = self._validate_write(agent, candidate, target_scope)
        if denial_reason is not None:
            self.quarantine.add(candidate, reason=denial_reason)
            self.audit_log.append(
                AuditEventType.CANDIDATE_QUARANTINED,
                actor=agent.id,
                candidate_id=candidate.id,
                target_scope=target_scope,
                reason=denial_reason,
            )
            return None

        record = MemoryRecord(
            content=candidate,
            writer=agent.id,
            source=candidate.source,
            trust=candidate.trust,
            scope=target_scope,
            evidence=(candidate.evidence_span,),
            taint=candidate.taint,
            parent_ids=candidate.parent_ids,
        )

        self.memory_store.insert(record)
        self.audit_log.append(
            AuditEventType.MEMORY_WRITTEN,
            actor=agent.id,
            memory_id=record.id,
            candidate_id=candidate.id,
            trust=record.trust.name,
            scope=record.scope,
            taint=record.taint,
        )

        return record

    def _validate_write(
        self,
        agent: Agent,
        candidate: CandidateFact,
        target_scope: str,
    ) -> str | None:
        if not agent.can_write_scope(target_scope):
            return "Agent has no write permission for target scope"

        if target_scope == MemoryScope.SHARED.value and candidate.trust < MIN_SHARED_TRUST:
            return "Low-trust content cannot enter shared memory"

        if target_scope == MemoryScope.SHARED.value and candidate.taint:
            return "Tainted content requires verification before shared write"

        if target_scope == MemoryScope.POLICY.value:
            return "Policy memory requires manual control-plane write"

        if looks_like_instruction(candidate):
            return "Instruction-like content cannot be stored as memory"

        if has_conflict(candidate, self.memory_store):
            return "Candidate conflicts with existing memory"

        return None
