"""Unit tests for d7_revocation.py."""

from __future__ import annotations

import unittest

from MASW.d7_revocation import MemoryRevoker
from MASW.memory_store import AuditLog, MemoryStore
from MASW.tests.helpers import candidate, memory
from MASW.types import AuditEventType


class RevocationTest(unittest.TestCase):
    def test_revokes_target_and_descendants(self) -> None:
        store = MemoryStore()
        audit = AuditLog()

        parent = memory(content=candidate(parent_ids=("input_1",)))
        child = memory(content=candidate(parent_ids=(parent.id,)), parent_ids=(parent.id,))
        unrelated = memory(content=candidate(subject="Service status", obj="healthy", parent_ids=("input_2",)))

        store.insert(parent)
        store.insert(child)
        store.insert(unrelated)

        revoked = MemoryRevoker(store, audit).revoke_poisoned_memory(parent.id, "confirmed poison")

        self.assertEqual(set(revoked), {parent.id, child.id})
        self.assertTrue(store.get(parent.id).revoked)
        self.assertTrue(store.get(child.id).revoked)
        self.assertEqual([record.id for record in store.all()], [unrelated.id])
        self.assertEqual(len(audit.by_type(AuditEventType.MEMORY_REVOKED)), 2)

    def test_unknown_memory_is_noop(self) -> None:
        revoked = MemoryRevoker(MemoryStore(), AuditLog()).revoke_poisoned_memory("missing", "test")

        self.assertEqual(revoked, [])


if __name__ == "__main__":
    unittest.main()
