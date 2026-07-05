"""Unit tests for d5_retrieval_audit.py."""

from __future__ import annotations

import unittest

from MASW.d5_retrieval_audit import MemoryRetriever
from MASW.memory_store import AuditLog, MemoryStore
from MASW.tests.helpers import agent, candidate, memory
from MASW.types import AuditEventType, MemoryScope, TaskContext, TrustLevel


class RetrievalAuditTest(unittest.TestCase):
    def test_retrieval_filters_low_trust_tainted_and_scope_mismatch(self) -> None:
        store = MemoryStore()
        audit = AuditLog()
        safe = memory(content=candidate(obj="Friday night"), trust=TrustLevel.VERIFIED, taint=False)
        tainted = memory(content=candidate(obj="Friday night"), trust=TrustLevel.VERIFIED, taint=True)
        private = memory(
            content=candidate(obj="Friday night"),
            trust=TrustLevel.VERIFIED,
            scope=MemoryScope.PRIVATE.value,
            taint=False,
        )
        store.insert(safe)
        store.insert(tainted)
        store.insert(private)

        context = MemoryRetriever(store, audit).retrieve(
            agent=agent(read_scopes=frozenset({MemoryScope.SHARED.value})),
            query="deployment window Friday",
            task_context=TaskContext(
                min_required_trust=TrustLevel.VERIFIED,
                requires_clean_context=True,
                allowed_scopes=frozenset({MemoryScope.SHARED.value}),
            ),
        )

        self.assertEqual([item.memory_id for item in context], [safe.id])
        self.assertEqual(len(audit.by_type(AuditEventType.MEMORY_RETRIEVED)), 1)


if __name__ == "__main__":
    unittest.main()
