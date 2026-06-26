import unittest
from pathlib import Path

from Reflection.evaluation import load_dataset


class DatasetLoaderTest(unittest.TestCase):
    def test_seed_dataset_loads(self) -> None:
        dataset_path = Path(__file__).resolve().parents[1] / "datasets" / "reflection_type3_seed.csv"
        samples = load_dataset(dataset_path)
        self.assertGreaterEqual(len(samples), 40)
        self.assertEqual(samples[0].sample_id, "R3A-001")
        self.assertEqual(samples[-1].sample_id, "R3B-020")


if __name__ == "__main__":
    unittest.main()
