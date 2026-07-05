"""防御节点 5：共享记忆写入网关。

所有写入 shared memory 的请求必须经过这里。
Agent 不能绕过网关直接把外部内容或候选事实写入共享记忆。
"""

from __future__ import annotations

from .conflict_detection import has_conflict
from .memory_store import AuditLog, MemoryStore, QuarantineStore
from .risk import looks_like_instruction
from .types import (
    Agent,
    AuditEventType,
    CandidateFact,
    MemoryRecord,
    MemoryScope,
    TrustLevel,
)


MIN_SHARED_TRUST = TrustLevel.VERIFIED


class MemoryWriteGateway:
    """共享记忆写入控制面。"""

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
