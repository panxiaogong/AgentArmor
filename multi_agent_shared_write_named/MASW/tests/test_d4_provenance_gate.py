"""Unit tests for d4_provenance_gate.py."""

from __future__ import annotations

import unittest

from MASW.d4_provenance_gate import MemoryVerifier, MemoryWriteGateway, has_conflict
from MASW.memory_store import AuditLog, MemoryStore, QuarantineStore
from MASW.tests.helpers import agent, candidate, memory
from MASW.types import MemoryScope, TrustLevel


class ProvenanceGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.store = MemoryStore()
        self.quarantine = QuarantineStore()
        self.audit = AuditLog()
        self.verifier = MemoryVerifier(self.store, self.quarantine, self.audit)
        self.gateway = MemoryWriteGateway(self.store, self.quarantine, self.audit)
        self.security_agent = agent(agent_id="verifier", clearance=TrustLevel.TRUSTED)

    def test_verifier_promotes_clean_candidate_only_after_checks(self) -> None:
        promoted = self.verifier.verify_and_promote(
            candidate(trust=TrustLevel.UNTRUSTED, taint=True),
            self.security_agent,
        )

        self.assertIsNotNone(promoted)
        self.assertEqual(promoted.trust, TrustLevel.VERIFIED)
        self.assertFalse(promoted.taint)

    def test_gateway_rejects_low_trust_shared_write(self) -> None:
        record = self.gateway.write(
            agent=self.security_agent,
            candidate=candidate(trust=TrustLevel.UNTRUSTED, taint=True),
            target_scope=MemoryScope.SHARED.value,
        )

        self.assertIsNone(record)
        self.assertEqual(len(self.quarantine.items), 1)

    def test_gateway_writes_verified_clean_candidate(self) -> None:
        record = self.gateway.write(
            agent=self.security_agent,
            candidate=candidate(trust=TrustLevel.VERIFIED, taint=False),
            target_scope=MemoryScope.SHARED.value,
        )

        self.assertIsNotNone(record)
        self.assertEqual(len(self.store.all()), 1)

    def test_conflict_detection_blocks_ambiguous_overwrite(self) -> None:
        existing = memory(content=candidate(obj="Friday night"), trust=TrustLevel.VERIFIED)
        self.store.insert(existing)

        conflicting = candidate(obj="Saturday morning", trust=TrustLevel.VERIFIED, taint=False)

        self.assertTrue(has_conflict(conflicting, self.store))


if __name__ == "__main__":
    unittest.main()
