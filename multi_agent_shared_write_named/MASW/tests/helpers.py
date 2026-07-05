"""Shared test helpers."""

from __future__ import annotations

from MASW.types import Agent, CandidateFact, MemoryRecord, MemoryScope, Task, TaskContext, TrustLevel


def agent(
    *,
    agent_id: str = "agent",
    clearance: TrustLevel = TrustLevel.TRUSTED,
    read_scopes: frozenset[str] | None = None,
    write_scopes: frozenset[str] | None = None,
    tools: frozenset[str] | None = None,
) -> Agent:
    return Agent(
        id=agent_id,
        clearance=clearance,
        read_scopes=read_scopes or frozenset({MemoryScope.SHARED.value}),
        write_scopes=write_scopes or frozenset({MemoryScope.SHARED.value}),
        tools=tools or frozenset({"ticket.create"}),
    )


def task(
    *,
    task_id: str = "task",
    query: str = "deployment window",
    source_uri: str = "https://example.invalid/source",
    requested_action: str | None = "ticket.create",
) -> Task:
    return Task(
        id=task_id,
        query=query,
        source_uri=source_uri,
        source_type="webpage",
        context=TaskContext(min_required_trust=TrustLevel.VERIFIED),
        requested_action=requested_action,
        target_resource="deployment-board",
    )


def candidate(
    *,
    subject: str = "Deployment window",
    predicate: str = "states",
    obj: str = "Friday night",
    source: str = "https://example.invalid/source",
    writer: str = "agent-a",
    trust: TrustLevel = TrustLevel.VERIFIED,
    taint: bool = False,
    evidence_span: str | None = None,
    parent_ids: tuple[str, ...] = ("input_1",),
) -> CandidateFact:
    evidence = evidence_span if evidence_span is not None else f"{subject}: {obj}"
    return CandidateFact(
        subject=subject,
        predicate=predicate,
        object=obj,
        confidence=0.90,
        evidence_span=evidence,
        source=source,
        writer=writer,
        trust=trust,
        taint=taint,
        parent_ids=parent_ids,
    )


def memory(
    *,
    content: CandidateFact | None = None,
    trust: TrustLevel = TrustLevel.VERIFIED,
    scope: str = MemoryScope.SHARED.value,
    taint: bool = False,
    parent_ids: tuple[str, ...] = ("input_1",),
) -> MemoryRecord:
    fact = content or candidate(trust=trust, taint=taint, parent_ids=parent_ids)
    return MemoryRecord(
        content=fact,
        writer=fact.writer,
        source=fact.source,
        trust=trust,
        scope=scope,
        evidence=(fact.evidence_span,),
        taint=taint,
        parent_ids=parent_ids,
    )
