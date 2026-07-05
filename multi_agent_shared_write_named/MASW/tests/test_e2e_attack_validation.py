"""Tests for the Type-5 end-to-end attack validation harness."""

from __future__ import annotations

import unittest

from MASW.e2e_vulnerable_agent import build_e2e_attack_scenarios, run_masw_defended_path, run_vulnerable_path
from MASW.tests.e2e_attack_validation import run_e2e_attack_validation


class EndToEndAttackValidationTest(unittest.TestCase):
    def test_scenario_count_is_small_but_not_trivial(self) -> None:
        scenarios = build_e2e_attack_scenarios()

        self.assertGreaterEqual(len(scenarios), 5)
        self.assertLessEqual(len(scenarios), 10)

    def test_vulnerable_path_executes_dangerous_tool(self) -> None:
        scenario = build_e2e_attack_scenarios()[0]
        store, tool_log, decision = run_vulnerable_path(scenario)

        self.assertEqual(len(store.all()), 1)
        self.assertTrue(decision.executed)
        self.assertEqual(len(tool_log.executions), 1)
        self.assertEqual(tool_log.executions[0].tool, scenario.expected_tool)
        self.assertEqual(tool_log.executions[0].resource, scenario.expected_resource)

    def test_masw_defended_path_blocks_same_chain(self) -> None:
        scenario = build_e2e_attack_scenarios()[0]
        written_count, quarantine_count, decision = run_masw_defended_path(scenario)

        self.assertEqual(written_count, 0)
        self.assertGreaterEqual(quarantine_count, 1)
        self.assertFalse(decision.executed)

    def test_full_validation_summary(self) -> None:
        report = run_e2e_attack_validation()
        summary = report["summary"]

        self.assertEqual(summary["total"], len(build_e2e_attack_scenarios()))
        self.assertEqual(summary["vulnerable_dangerous_executions"], summary["total"])
        self.assertEqual(summary["masw_poisoned_memory_writes"], 0)
        self.assertEqual(summary["masw_dangerous_executions"], 0)


if __name__ == "__main__":
    unittest.main()
