"""End-to-end validation of the Type-5 shared-write attack chain."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

from MASW.e2e_vulnerable_agent import (
    EndToEndValidationResult,
    build_e2e_attack_scenarios,
    run_masw_defended_path,
    run_vulnerable_path,
)


RESULTS_DIR = Path(__file__).resolve().parent / "results"
E2E_JSON_PATH = RESULTS_DIR / "masw_e2e_attack_validation_results.json"
E2E_CSV_PATH = RESULTS_DIR / "masw_e2e_attack_validation_results.csv"
E2E_REPORT_PATH = RESULTS_DIR / "masw_e2e_attack_validation_report.md"


def _run_one(scenario: object) -> EndToEndValidationResult:
    vulnerable_store, tool_log, vulnerable_decision = run_vulnerable_path(scenario)
    masw_written_count, masw_quarantine_count, masw_decision = run_masw_defended_path(scenario)

    last_execution = tool_log.executions[-1] if tool_log.executions else None
    return EndToEndValidationResult(
        scenario_id=scenario.id,
        category=scenario.category,
        expected_tool=scenario.expected_tool,
        vulnerable_memory_written=len(vulnerable_store.all()) > 0,
        vulnerable_action_executed=vulnerable_decision.executed,
        vulnerable_executed_tool=last_execution.tool if last_execution else "none",
        vulnerable_executed_resource=last_execution.resource if last_execution else None,
        masw_memory_written=masw_written_count > 0,
        masw_quarantine_count=masw_quarantine_count,
        masw_action_executed=masw_decision.executed,
        masw_decision_reason=masw_decision.reason,
    )


def _write_csv(rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "scenario_id",
        "category",
        "expected_tool",
        "vulnerable_memory_written",
        "vulnerable_action_executed",
        "vulnerable_executed_tool",
        "vulnerable_executed_resource",
        "masw_memory_written",
        "masw_quarantine_count",
        "masw_action_executed",
        "masw_decision_reason",
    ]
    with E2E_CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_report(rows: list[dict[str, object]]) -> str:
    total = len(rows)
    vulnerable_success = sum(1 for row in rows if bool(row["vulnerable_action_executed"]))
    masw_executed = sum(1 for row in rows if bool(row["masw_action_executed"]))
    masw_poison_writes = sum(1 for row in rows if bool(row["masw_memory_written"]))

    lines = [
        "# MASW End-to-End Attack Validation",
        "",
        "## Attack Chain",
        "",
        "`Q_inject -> poisoned shared memory write -> Q_target -> retrieval -> dangerous tool execution`",
        "",
        "## Summary",
        "",
        f"- Scenarios: {total}",
        f"- Vulnerable path dangerous executions: {vulnerable_success}/{total}",
        f"- MASW poisoned memory writes: {masw_poison_writes}/{total}",
        f"- MASW dangerous executions: {masw_executed}/{total}",
        "",
        "## Scenario Results",
        "",
        "| Scenario | Category | Tool | Vulnerable write | Vulnerable exec | MASW write | MASW quarantine | MASW exec | MASW reason |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['scenario_id']}` | {row['category']} | `{row['expected_tool']}` | "
            f"{row['vulnerable_memory_written']} | {row['vulnerable_action_executed']} | "
            f"{row['masw_memory_written']} | {row['masw_quarantine_count']} | "
            f"{row['masw_action_executed']} | {row['masw_decision_reason']} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The vulnerable agent succeeds because it gives external content the same effective trust as internal shared memory, then treats retrieved memory as an executable instruction. MASW breaks the chain at two independent points: poisoned content is quarantined before shared-memory write, and high-impact tools still require approval even if a later task requests them.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_e2e_attack_validation() -> dict[str, object]:
    scenarios = build_e2e_attack_scenarios()
    result_rows = [asdict(_run_one(scenario)) for scenario in scenarios]

    summary = {
        "total": len(result_rows),
        "vulnerable_dangerous_executions": sum(1 for row in result_rows if bool(row["vulnerable_action_executed"])),
        "masw_poisoned_memory_writes": sum(1 for row in result_rows if bool(row["masw_memory_written"])),
        "masw_dangerous_executions": sum(1 for row in result_rows if bool(row["masw_action_executed"])),
    }
    payload = {"summary": summary, "rows": result_rows}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    E2E_JSON_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(result_rows)
    E2E_REPORT_PATH.write_text(_build_report(result_rows), encoding="utf-8")
    return payload


if __name__ == "__main__":
    report = run_e2e_attack_validation()
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
