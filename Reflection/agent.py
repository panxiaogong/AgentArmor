from __future__ import annotations

from typing import List

from Reflection.config import PipelineConfig
from Reflection.memory_store import MemoryStore
from Reflection.pipeline import ReflectionDefensePipeline
from Reflection.types import ConversationTurn, PipelineResult, SourceType


class ReflectionAgent:
    """Small end-to-end agent that exposes the reflection-write attack surface."""

    def __init__(
        self,
        pipeline: ReflectionDefensePipeline | None = None,
        memory_store: MemoryStore | None = None,
    ) -> None:
        self.pipeline = pipeline or ReflectionDefensePipeline.from_config()
        self.memory_store = memory_store or MemoryStore()
        self.history: List[ConversationTurn] = []

    def observe(self, text: str, source: SourceType = SourceType.USER) -> ConversationTurn:
        turn = ConversationTurn(
            turn_id=f"turn-{len(self.history) + 1:03d}",
            source=source,
            text=text,
        )
        # 显式记录原始 turn，而不是只保留 summary，
        # 是因为 D2 接地性和 D1 反思注入检测都依赖原始证据。
        self.history.append(turn)
        return turn

    def reflect(self) -> PipelineResult:
        return self.pipeline.process(self.history, self.memory_store)

    def answer(self, query: str) -> str:
        retrieval = self.pipeline.on_retrieval(query, self.memory_store)
        if not retrieval.entries:
            return "I do not have trusted long-term memory for that request."
        # 这个最小 Agent 直接复述检索后排名最高的安全记忆，
        # 便于端到端验证“检索防御是否真的改变最终上下文注入结果”。
        return f"Trusted memory says: {retrieval.entries[0].record.fact_text}"


class UnsafeReflectionAgent(ReflectionAgent):
    """Baseline agent that blindly persists reflection outputs."""

    def __init__(self) -> None:
        super().__init__(pipeline=ReflectionDefensePipeline.from_config(PipelineConfig.unsafe()))
