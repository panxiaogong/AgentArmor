import unittest

from Reflection.evaluation import compute_metrics


class EvaluationMetricsTest(unittest.TestCase):
    def test_metrics_are_computed_correctly(self) -> None:
        metrics = compute_metrics(
            gold_attack=[True, True, False, False],
            predicted_attack=[True, False, True, False],
            latencies_ms=[10.0, 20.0, 30.0, 40.0],
        )
        self.assertAlmostEqual(metrics["Prec"], 0.5)
        self.assertAlmostEqual(metrics["Rec"], 0.5)
        self.assertAlmostEqual(metrics["F1"], 0.5)
        self.assertAlmostEqual(metrics["FPR"], 0.5)
        self.assertEqual(metrics["P50ms"], 30.0)


if __name__ == "__main__":
    unittest.main()
