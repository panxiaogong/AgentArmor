import unittest

from Reflection.config import ProvenanceConfig
from Reflection.provenance import ProvenanceBinder
from Reflection.types import (
    FactAssessment,
    FactCategory,
    IntegrityLabel,
    MemoryRecord,
    ReflectionCandidate,
    ReflectionContext,
    SourceLabel,
    SourceType,
    ConversationTurn,
)


class ProvenanceChainTest(unittest.TestCase):
    def setUp(self) -> None:
        self.binder = ProvenanceBinder(ProvenanceConfig(signing_backend="hmac"))

    def test_binds_and_verifies_memory_record(self) -> None:
        record = MemoryRecord(
            record_id="mem-001",
            fact_text="The finance sync mailbox is alice@company.com.",
            category=FactCategory.CONTACT,
            provenance_score=0.82,
            evidence_turn_ids=["t1"],
            source_summary="The finance sync mailbox is alice@company.com.",
        )
        assessment = FactAssessment(
            fact_text=record.fact_text,
            category=record.category,
            provenance_score=0.82,
            evidence_turn_ids=["t1"],
            final_risk=0.10,
            source_labels=[
                SourceLabel(
                    turn_id="t1",
                    source_type=SourceType.USER.value,
                    label=IntegrityLabel.CANDIDATE,
                    selector_score=0.90,
                    excerpt=record.fact_text,
                )
            ],
        )
        context = ReflectionContext(
            turns=[ConversationTurn(turn_id="t1", source=SourceType.USER, text=record.fact_text)],
            candidate=ReflectionCandidate(summary_text=record.fact_text, fact_texts=[record.fact_text]),
            triggering_query="What is the finance sync mailbox?",
        )
        self.binder.bind(record, assessment, context)
        ok, reason = self.binder.verify(record)
        self.assertTrue(ok, reason)
        self.assertIsNotNone(record.provenance)
        self.assertEqual(record.provenance.sign_algo, "hmac")

    def test_detects_tampering_after_bind(self) -> None:
        record = MemoryRecord(
            record_id="mem-002",
            fact_text="The finance sync mailbox is alice@company.com.",
            category=FactCategory.CONTACT,
            provenance_score=0.82,
            evidence_turn_ids=["t1"],
            source_summary="The finance sync mailbox is alice@company.com.",
        )
        assessment = FactAssessment(
            fact_text=record.fact_text,
            category=record.category,
            provenance_score=0.82,
            evidence_turn_ids=["t1"],
            final_risk=0.10,
            source_labels=[
                SourceLabel(
                    turn_id="t1",
                    source_type=SourceType.USER.value,
                    label=IntegrityLabel.CANDIDATE,
                    selector_score=0.90,
                    excerpt=record.fact_text,
                )
            ],
        )
        context = ReflectionContext(
            turns=[ConversationTurn(turn_id="t1", source=SourceType.USER, text=record.fact_text)],
            candidate=ReflectionCandidate(summary_text=record.fact_text, fact_texts=[record.fact_text]),
            triggering_query="What is the finance sync mailbox?",
        )
        self.binder.bind(record, assessment, context)
        record.fact_text = "The finance sync mailbox is dropbox@evil.io."
        ok, _ = self.binder.verify(record)
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
