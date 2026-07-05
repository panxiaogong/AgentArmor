"""Dataset construction tests."""

from __future__ import annotations

import unittest

from MASW.tests.build_dataset import CSV_DATASET_PATH, build_samples, count_by_category, write_csv


class DatasetConstructionTest(unittest.TestCase):
    def test_minimum_counts(self) -> None:
        samples = build_samples()
        counts = count_by_category(samples)

        self.assertGreaterEqual(counts["prompt_injection"], 20)
        self.assertGreaterEqual(counts["tool_misuse"], 20)
        self.assertGreaterEqual(counts["memory_poisoning"], 20)
        self.assertGreaterEqual(counts["agent_hijacking"], 20)
        self.assertGreaterEqual(counts["benign"], 40)
        self.assertGreaterEqual(len(samples), 120)

    def test_ids_are_unique_and_schema_is_stable(self) -> None:
        samples = build_samples()
        ids = [sample["id"] for sample in samples]

        self.assertEqual(len(ids), len(set(ids)))
        for sample in samples:
            self.assertIn("content", sample)
            self.assertIn("expected", sample)
            self.assertIn("memory_write", sample["expected"])
            self.assertIn("quarantine", sample["expected"])
            self.assertIn("blocked", sample["expected"])

    def test_csv_export_matches_report_schema(self) -> None:
        path = write_csv()
        header = path.read_text(encoding="utf-8").splitlines()[0]

        self.assertEqual(path, CSV_DATASET_PATH)
        self.assertEqual(
            header,
            "id,family,stage,user_goal,actor,content,task_summary,user_query,label,attack_type,dataset_source,note",
        )


if __name__ == "__main__":
    unittest.main()
