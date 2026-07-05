"""d7: revocation.

一旦确认某条记忆被污染，应按 parent_ids 追踪派生记忆并统一撤销。
这比只删除单条记录更安全，因为污染可能已经被二次总结、合并或改写。
"""

from __future__ import annotations

from .memory_store import AuditLog, MemoryStore
from .types import AuditEventType, MemoryRecord


class MemoryRevoker:
    """污染记忆撤销器。"""

    def __init__(self, memory_store: MemoryStore, audit_log: AuditLog) -> None:
        self.memory_store = memory_store
        self.audit_log = audit_log

    def revoke_poisoned_memory(self, memory_id: str, reason: str) -> list[str]:
        """撤销目标记忆及其全部派生记忆。"""

        revoked_ids: list[str] = []
        target = self.memory_store.get(memory_id)
        if target is None:
            return revoked_ids

        for descendant in self._find_descendants(memory_id):
            if self.memory_store.mark_revoked(descendant.id):
                revoked_ids.append(descendant.id)
                self.audit_log.append(
                    AuditEventType.MEMORY_REVOKED,
                    actor="memory_revoker",
                    memory_id=descendant.id,
                    reason=f"Derived from poisoned memory {memory_id}: {reason}",
                )

        if self.memory_store.mark_revoked(memory_id):
            revoked_ids.append(memory_id)
            self.audit_log.append(
                AuditEventType.MEMORY_REVOKED,
                actor="memory_revoker",
                memory_id=memory_id,
                reason=reason,
            )

        return revoked_ids

    def _find_descendants(self, memory_id: str) -> list[MemoryRecord]:
        descendants: list[MemoryRecord] = []

        for record in self.memory_store.all(include_revoked=True):
            if memory_id in record.parent_ids:
                descendants.append(record)
                descendants.extend(self._find_descendants(record.id))

        return descendants
