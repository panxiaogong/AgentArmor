"""Unit tests for ablation_eval.py."""

from __future__ import annotations

import unittest

from MASW.tests.ablation_eval import _metrics, run_ablation


class AblationEvalTest(unittest.TestCase):
    def test_metric_formulae(self) -> None:
        rows = [
            {"label": "attack", "predicted_attack": True, "elapsed_ms": 1.0},
            {"label": "attack", "predicted_attack": False, "elapsed_ms": 2.0},
            {"label": "benign", "predicted_attack": True, "elapsed_ms": 3.0},
            {"label": "benign", "predicted_attack": False, "elapsed_ms": 4.0},
        ]

        metrics = _metrics(rows, include_latency=True)

        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["fn"], 1)
        self.assertEqual(metrics["tn"], 1)
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["f1"], 0.5)
        self.assertEqual(metrics["fpr"], 0.5)
        self.assertIsNotNone(metrics["p50_ms"])
        self.assertIsNotNone(metrics["p95_ms"])
        self.assertIsNotNone(metrics["p99_ms"])

    def test_run_ablation_produces_all_and_synergy_answer(self) -> None:
        report = run_ablation()
        configs = {row["config"] for row in report["metrics"]}

        self.assertIn("ALL", configs)
        self.assertIn("D3_RISK_FILTER_ONLY", configs)
        self.assertIn("strict_f1_synergy", report["synergy"])


if __name__ == "__main__":
    unittest.main()
