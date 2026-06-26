"""
D3：一致性核查——候选事实是否与已信任记忆冲突。

反思型注入经常伪装成“你之前记住的事实”。
因此对长时记忆中已有稳定槽位做冲突检测，是类型三的重要一道闸。
"""
from __future__ import annotations

from Reflection.config import D3Config
from Reflection.memory_store import MemoryStore
from Reflection.types import DefenseVerdict, FactCategory
from Reflection.utils import extract_canonical_value, extract_slot_key


class D3ConsistencyAuditor:
    """对候选事实执行槽位级冲突检查。"""

    def __init__(self, config: D3Config) -> None:
        self.cfg = config

    def check(
        self,
        fact_text: str,
        category: FactCategory,
        memory_store: MemoryStore | None,
    ) -> DefenseVerdict:
        if not self.cfg.enabled:
            return DefenseVerdict("D3", passed=True, score=0.0, reason="D3 已禁用", action="PASS")
        if memory_store is None:
            return DefenseVerdict("D3", passed=True, score=0.0, reason="无既有记忆，跳过冲突检查", action="PASS")
        if self.cfg.strategy == "slot_conflict":
            return self._slot_conflict_check(fact_text, category, memory_store)
        if self.cfg.strategy == "consensus":
            return self._consensus_check(fact_text, category, memory_store)
        raise ValueError(f"未知 D3 strategy: {self.cfg.strategy}")

    def _slot_conflict_check(
        self,
        fact_text: str,
        category: FactCategory,
        memory_store: MemoryStore,
    ) -> DefenseVerdict:
        slot_key = extract_slot_key(fact_text, category)
        value = extract_canonical_value(fact_text, category)
        for record in memory_store.by_category(category):
            # 先比较槽位，再比较该槽位的规范化值。
            # 这样可以避免“完全不同字段的事实”被误报为冲突。
            record_slot = extract_slot_key(record.fact_text, record.category)
            record_value = extract_canonical_value(record.fact_text, record.category)
            if slot_key == record_slot and value and record_value and value != record_value:
                return DefenseVerdict(
                    node="D3",
                    passed=False,
                    score=1.0,
                    reason=f"候选事实与已信任记忆冲突：{record.fact_text}",
                    action="BLOCK",
                    metadata={"contradiction_score": 1.0, "conflict_record_id": record.record_id},
                )

        return DefenseVerdict(
            node="D3",
            passed=True,
            score=0.0,
            reason="未发现与现有信任记忆冲突的槽位",
            action="PASS",
            metadata={"contradiction_score": 0.0},
        )

    def _consensus_check(
        self,
        fact_text: str,
        category: FactCategory,
        memory_store: MemoryStore,
    ) -> DefenseVerdict:
        slot_key = extract_slot_key(fact_text, category)
        value = extract_canonical_value(fact_text, category)
        slot_matches = []
        for record in memory_store.by_category(category):
            record_slot = extract_slot_key(record.fact_text, record.category)
            record_value = extract_canonical_value(record.fact_text, record.category)
            if slot_key == record_slot and value and record_value:
                slot_matches.append(record)

        if not slot_matches:
            return DefenseVerdict(
                node="D3",
                passed=True,
                score=0.0,
                reason="无既有共识槽位，允许继续后续核查",
                action="PASS",
                metadata={"contradiction_score": 0.0},
            )

        same_value_records = [
            record
            for record in slot_matches
            if extract_canonical_value(record.fact_text, record.category) == value
        ]
        if same_value_records:
            return DefenseVerdict(
                node="D3",
                passed=True,
                score=0.0,
                reason="候选事实与既有高可信记忆共识一致",
                action="PASS",
                metadata={"contradiction_score": 0.0, "consensus_size": len(same_value_records)},
            )

        return DefenseVerdict(
            node="D3",
            passed=False,
            score=1.0,
            reason=f"候选事实偏离既有记忆共识：{slot_matches[0].fact_text}",
            action="BLOCK",
            metadata={"contradiction_score": 1.0, "consensus_size": len(slot_matches)},
        )
