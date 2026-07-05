"""Unit tests for tech_selection_eval.py."""

from __future__ import annotations

import unittest

from MASW.tests.tech_selection_eval import build_selection_samples, run_selection_eval


class TechSelectionEvalTest(unittest.TestCase):
    def test_selection_dataset_adds_hard_samples(self) -> None:
        samples = build_selection_samples()
        categories = {str(sample["category"]) for sample in samples}

        self.assertIn("subtle_exfiltration", categories)
        self.assertIn("subtle_benign_sync", categories)
        self.assertGreater(len(samples), 125)

    def test_run_selection_eval_outputs_recommended_config(self) -> None:
        report = run_selection_eval()
        configs = {row["config"] for row in report["metrics"]}

        self.assertIn("D3_HYBRID_PLUS_D4_RULE", configs)
        self.assertIn("D3_HYBRID_PLUS_D4_HYBRID", configs)
        self.assertIn("ALL_RECOMMENDED", configs)


if __name__ == "__main__":
    unittest.main()
