"""Minimal evaluation harness for MASW.

指标含义：
- memory_written: 写入共享记忆的数量。
- quarantine_count: 进入隔离区的数量。
- action_executed: 动作是否通过仲裁。
"""

from __future__ import annotations

from MASW.pipeline import build_default_pipeline
from MASW.tests.build_dataset import build_samples
from MASW.types import Agent, Task, TaskContext, TrustLevel


def run_eval() -> list[dict[str, object]]:
    results: list[dict[str, object]] = []

    for index, sample in enumerate(build_samples(), start=1):
        pipeline = build_default_pipeline()
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
        task = Task(
            id=f"task-{index}",
            query="deployment window",
            source_uri="https://example.invalid/page",
            source_type="webpage",
            context=TaskContext(min_required_trust=TrustLevel.VERIFIED),
            requested_action="ticket.create",
            target_resource="deployment-board",
        )

        result = pipeline.full_secure_path(
            agent_a=agent_a,
            agent_b=agent_b,
            verifier_agent=verifier,
            raw_external_content=sample["content"],
            task=task,
        )

        results.append(
            {
                "label": sample["label"],
                "memory_written": len(result.written_memories),
                "quarantine_count": len(pipeline.quarantine.items),
                "action_executed": result.action_decision.executed,
            }
        )

    return results


if __name__ == "__main__":
    for row in run_eval():
        print(row)
