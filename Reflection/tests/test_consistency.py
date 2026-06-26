import unittest

from Reflection.config import D3Config
from Reflection.d3_consistency import D3ConsistencyAuditor
from Reflection.memory_store import MemoryStore
from Reflection.types import FactCategory, MemoryRecord


class ConsistencyAuditorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.guard = D3ConsistencyAuditor(D3Config())
        self.memory_store = MemoryStore()
        self.memory_store.commit(
            MemoryRecord(
                record_id="mem-001",
                fact_text="The user's contact email is alice@company.com.",
                category=FactCategory.CONTACT,
                provenance_score=0.92,
                evidence_turn_ids=["t1"],
                source_summary="The user's contact email is alice@company.com.",
            )
        )

    def test_flags_conflicting_contact_detail(self) -> None:
        verdict = self.guard.check(
            "The user's contact email is dropbox@evil.io.",
            FactCategory.CONTACT,
            self.memory_store,
        )
        self.assertEqual(verdict.metadata["contradiction_score"], 1.0)
        self.assertEqual(verdict.action, "BLOCK")

    def test_allows_non_conflicting_new_fact(self) -> None:
        verdict = self.guard.check(
            "The user prefers jasmine tea.",
            FactCategory.PREFERENCE,
            self.memory_store,
        )
        self.assertEqual(verdict.metadata["contradiction_score"], 0.0)
        self.assertEqual(verdict.action, "PASS")


if __name__ == "__main__":
    unittest.main()
