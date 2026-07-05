"""Regression tests that run the whole dataset through the MASW pipeline."""

from __future__ import annotations

import unittest

from MASW.pipeline import build_default_pipeline
from MASW.tests.build_dataset import build_samples
from MASW.tests.eval_masw import _agents
from MASW.types import Task, TaskContext, TrustLevel


class DatasetAgainstPipelineTest(unittest.TestCase):
    def test_expected_memory_and_quarantine_outcomes(self) -> None:
        for sample in build_samples():
            with self.subTest(sample_id=sample["id"], category=sample["category"]):
                pipeline = build_default_pipeline()
                agent_a, verifier, agent_b = _agents()
                task = Task(
                    id=str(sample["id"]),
                    query="deployment window incident status",
                    source_uri=str(sample["source_uri"]),
                    source_type=str(sample["source_type"]),
                    context=TaskContext(min_required_trust=TrustLevel.VERIFIED),
                    requested_action="ticket.create",
                    target_resource="deployment-board",
                )

                result = pipeline.full_secure_path(
                    agent_a=agent_a,
                    agent_b=agent_b,
                    verifier_agent=verifier,
                    raw_external_content=str(sample["content"]),
                    task=task,
                )

                expected = sample["expected"]
                self.assertEqual(bool(result.written_memories), expected["memory_write"])
                self.assertEqual(bool(pipeline.quarantine.items), expected["quarantine"])


if __name__ == "__main__":
    unittest.main()
