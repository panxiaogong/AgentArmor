from __future__ import annotations

from typing import List

from Reflection.types import FactAssessment, FactCategory, MemoryRecord
from Reflection.utils import lexical_support


class MemoryStore:
    """Simple in-repo memory backend used for unit tests and end-to-end demos."""

    def __init__(self) -> None:
        self.records: List[MemoryRecord] = []
        self.quarantine: List[FactAssessment] = []

    def commit(self, record: MemoryRecord) -> None:
        self.records.append(record)

    def quarantine_fact(self, assessment: FactAssessment) -> None:
        self.quarantine.append(assessment)

    def by_category(self, category: FactCategory) -> List[MemoryRecord]:
        return [record for record in self.records if record.category == category]

    def search(self, query: str) -> List[MemoryRecord]:
        """Return likely-relevant memories ranked by token overlap."""

        ranked = []
        for record in self.records:
            score = lexical_support(query, record.fact_text)
            if score > 0:
                ranked.append((score, record))
        # 检索阶段只做轻量排序，不改写底层 records 顺序，
        # 这样可以把“存储顺序”和“查询排序”两个职责分开。
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [record for _, record in ranked]
