"""d5: retrieval audit.

普通 RAG/记忆检索常只看语义相似度、时间和重要性。
这里额外加入 trust、scope、taint 和 risk，避免低信任记忆被高权限 Agent 当成指令。
"""

from __future__ import annotations

from datetime import datetime, timezone
from math import exp

from .d3_risk_filter import memory_risk
from .memory_store import AuditLog, MemoryStore, tokenize
from .types import (
    Agent,
    AuditEventType,
    MemoryContextItem,
    MemoryRecord,
    TaskContext,
    TrustLevel,
)


class MemoryRetriever:
    """信任感知检索器。"""

    def __init__(self, memory_store: MemoryStore, audit_log: AuditLog) -> None:
        self.memory_store = memory_store
        self.audit_log = audit_log

    def retrieve(self, agent: Agent, query: str, task_context: TaskContext) -> list[MemoryContextItem]:
        raw_candidates = self.memory_store.vector_search(query)
        scored: list[tuple[MemoryRecord, float]] = []

        for memory in raw_candidates:
            if not self._can_read(agent, memory, task_context):
                continue

            score = self._trust_aware_score(memory, query)
            if score <= 0:
                continue

            scored.append((memory, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        selected = scored[: task_context.top_k]

        context = [
            MemoryContextItem(
                memory_id=memory.id,
                content=memory.as_text(),
                source=memory.source,
                trust=memory.trust,
                taint=memory.taint,
                score=score,
            )
            for memory, score in selected
        ]

        for item in context:
            self.audit_log.append(
                AuditEventType.MEMORY_RETRIEVED,
                actor=agent.id,
                memory_id=item.memory_id,
                score=item.score,
                trust=item.trust.name,
                taint=item.taint,
            )

        return context

    def _can_read(self, agent: Agent, memory: MemoryRecord, task_context: TaskContext) -> bool:
        if memory.scope not in task_context.allowed_scopes:
            return False
        if not agent.can_read_scope(memory.scope):
            return False
        if memory.trust < task_context.min_required_trust:
            return False
        if task_context.requires_clean_context and memory.taint:
            return False
        return True

    def _trust_aware_score(self, memory: MemoryRecord, query: str) -> float:
        semantic = self._lexical_similarity(memory.as_text(), query)
        recency = self._time_decay(memory)
        importance = self._estimate_importance(memory)
        trust_bonus = int(memory.trust) / int(TrustLevel.TRUSTED)
        risk_penalty = memory_risk(memory)

        return (
            0.45 * semantic
            + 0.15 * recency
            + 0.15 * importance
            + 0.25 * trust_bonus
            - 0.50 * risk_penalty
        )

    def _lexical_similarity(self, memory_text: str, query: str) -> float:
        memory_tokens = tokenize(memory_text)
        query_tokens = tokenize(query)
        if not memory_tokens or not query_tokens:
            return 0.0
        return len(memory_tokens & query_tokens) / len(memory_tokens | query_tokens)

    def _time_decay(self, memory: MemoryRecord) -> float:
        age_seconds = (datetime.now(timezone.utc) - memory.created_at).total_seconds()
        half_life_seconds = 7 * 24 * 60 * 60
        return exp(-age_seconds / half_life_seconds)

    def _estimate_importance(self, memory: MemoryRecord) -> float:
        if memory.trust >= TrustLevel.VERIFIED:
            return 0.75
        return 0.25
