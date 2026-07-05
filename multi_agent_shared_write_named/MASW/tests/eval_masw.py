"""Minimal evaluation harness for MASW.

Metrics:
- ASR proxy: attack samples that still entered shared memory.
- Quarantine rate: attack samples sent to quarantine.
- Benign write rate: benign samples that remained useful.

The evaluator also writes a JSON result log under tests/results for manual
inspection of tool-misuse and memory-poisoning cases.
"""

from __future__ import annotations

import json
from pathlib import Path

from MASW.pipeline import build_default_pipeline
from MASW.tests.build_dataset import build_samples, count_by_category, write_jsonl
from MASW.types import Agent, Task, TaskContext, TrustLevel


RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_PATH = RESULTS_DIR / "masw_eval_results.json"


def _agents() -> tuple[Agent, Agent, Agent]:
    agent_a = Agent(
        id="agent-a",
        clearance=TrustLevel.UNTRUSTED,
        read_scopes=frozenset({"private", "shared"}),
        write_scopes=frozenset({"private"}),
        tools=frozenset({"browser.read"}),
    )
    verifier = Agent(
        id="verifier",
        clearance=TrustLevel.TRUSTED,
        read_scopes=frozenset({"shared", "quarantine"}),
        write_scopes=frozenset({"shared", "quarantine"}),
        tools=frozenset(),
    )
    agent_b = Agent(
        id="agent-b",
        clearance=TrustLevel.TRUSTED,
        read_scopes=frozenset({"shared"}),
        write_scopes=frozenset({"private"}),
        tools=frozenset({"ticket.create"}),
    )
    return agent_a, verifier, agent_b


def run_eval() -> dict[str, object]:
    write_jsonl()
    rows: list[dict[str, object]] = []

    for sample in build_samples():
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

        rows.append(
            {
                "id": sample["id"],
                "label": sample["label"],
                "category": sample["category"],
                "memory_written": len(result.written_memories),
                "quarantine_count": len(pipeline.quarantine.items),
                "action_executed": result.action_decision.executed,
                "decision_reason": result.action_decision.reason,
            }
        )

    attack_rows = [row for row in rows if row["label"] == "attack"]
    benign_rows = [row for row in rows if row["label"] == "benign"]
    attack_writes = sum(1 for row in attack_rows if int(row["memory_written"]) > 0)
    attack_quarantines = sum(1 for row in attack_rows if int(row["quarantine_count"]) > 0)
    benign_writes = sum(1 for row in benign_rows if int(row["memory_written"]) > 0)

    summary = {
        "total": len(rows),
        "by_category": count_by_category(build_samples()),
        "attack_shared_memory_writes": attack_writes,
        "attack_quarantine_rate": attack_quarantines / len(attack_rows),
        "benign_write_rate": benign_writes / len(benign_rows),
        "rows": rows,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    report = run_eval()
    printable = {key: value for key, value in report.items() if key != "rows"}
    print(json.dumps(printable, ensure_ascii=False, indent=2))
