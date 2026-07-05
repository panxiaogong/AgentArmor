"""Unit tests for memory_store.py and types.py."""

from __future__ import annotations

import unittest

from MASW.memory_store import AuditLog, MemoryStore, QuarantineStore, tokenize
from MASW.tests.helpers import candidate, memory
from MASW.types import AuditEventType, MemoryScope, TrustLevel, new_id


class MemoryStoreAndTypesTest(unittest.TestCase):
    def test_tokenize_and_vector_search(self) -> None:
        store = MemoryStore()
        stored = memory(content=candidate(subject="Deployment window", obj="Friday night"))
        store.insert(stored)

        self.assertIn("deployment", tokenize("Deployment window"))
        self.assertEqual(store.vector_search("deployment Friday"), [stored])

    def test_revoked_records_are_excluded_by_default(self) -> None:
        store = MemoryStore()
        stored = memory()
        store.insert(stored)
        store.mark_revoked(stored.id)

        self.assertEqual(store.all(), [])
        self.assertEqual(store.all(include_revoked=True), [stored])

    def test_quarantine_and_audit_log(self) -> None:
        quarantine = QuarantineStore()
        audit = AuditLog()
        item = quarantine.add({"payload": "x"}, "test reason")
        event = audit.append(AuditEventType.ACTION_DENIED, actor="agent", reason="test")

        self.assertEqual(item.reason, "test reason")
        self.assertEqual(audit.by_type(AuditEventType.ACTION_DENIED), [event])

    def test_core_types_have_expected_ordering_and_values(self) -> None:
        self.assertLess(TrustLevel.UNTRUSTED, TrustLevel.VERIFIED)
        self.assertEqual(MemoryScope.SHARED.value, "shared")
        self.assertTrue(new_id("test").startswith("test_"))


if __name__ == "__main__":
    unittest.main()
