"""内存版共享记忆、隔离区和审计日志。

生产环境中，这个文件可以替换为数据库、向量库或图数据库适配器。
为了让防御逻辑可读可测，示例实现只使用 Python 标准库。
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable

from .types import (
    AuditEvent,
    AuditEventType,
    MemoryRecord,
    QuarantineItem,
)


def tokenize(text: str) -> set[str]:
    """非常轻量的词元化函数。

    这里只用于演示 trust-aware retrieval 的接口形态。
    真实系统应替换为 embedding/vector search，但保留后续 trust/risk 重排。
    """

    return set(re.findall(r"[a-zA-Z0-9_@\-.]+", text.lower()))


class MemoryStore:
    """持久记忆仓库。

    重要约束：
    1. store 不负责判断是否能写入；写入决策在 d4_provenance_gate.py。
    2. store 只保存 provenance，不自动提升 trust。
    3. revoked 记录默认不参与检索，但保留审计痕迹。
    """

    def __init__(self) -> None:
        self._records: dict[str, MemoryRecord] = {}
        self._by_subject_predicate: dict[tuple[str, str], set[str]] = defaultdict(set)

    def insert(self, record: MemoryRecord) -> MemoryRecord:
        self._records[record.id] = record
        key = (
            record.content.subject.lower().strip(),
            record.content.predicate.lower().strip(),
        )
        self._by_subject_predicate[key].add(record.id)
        return record

    def get(self, memory_id: str) -> MemoryRecord | None:
        return self._records.get(memory_id)

    def all(self, include_revoked: bool = False) -> list[MemoryRecord]:
        records = list(self._records.values())
        if include_revoked:
            return records
        return [record for record in records if not record.revoked]

    def mark_revoked(self, memory_id: str) -> bool:
        record = self._records.get(memory_id)
        if record is None:
            return False
        record.revoked = True
        return True

    def find_by_subject_predicate(self, subject: str, predicate: str) -> list[MemoryRecord]:
        key = (subject.lower().strip(), predicate.lower().strip())
        ids = self._by_subject_predicate.get(key, set())
        return [self._records[memory_id] for memory_id in ids if not self._records[memory_id].revoked]

    def vector_search(self, query: str, limit: int = 20) -> list[MemoryRecord]:
        """占位版向量检索。

        使用 token overlap 模拟语义召回。真实系统可以在这里接入
        Chroma、FAISS、pgvector、Milvus、Mem0 或图记忆检索。
        """

        query_tokens = tokenize(query)
        scored: list[tuple[MemoryRecord, float]] = []

        for record in self.all():
            memory_tokens = tokenize(record.as_text())
            if not memory_tokens:
                continue
            overlap = len(query_tokens & memory_tokens)
            union = len(query_tokens | memory_tokens)
            score = overlap / union if union else 0.0
            if score > 0:
                scored.append((record, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return [record for record, _score in scored[:limit]]


class QuarantineStore:
    """隔离区。

    被隔离的对象不进入普通检索路径，只供安全审计、人工复核或离线分析。
    """

    def __init__(self) -> None:
        self.items: list[QuarantineItem] = []

    def add(self, payload: object, reason: str) -> QuarantineItem:
        item = QuarantineItem(payload=payload, reason=reason)
        self.items.append(item)
        return item


class AuditLog:
    """内存版审计日志。"""

    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    def append(self, event_type: AuditEventType, actor: str, **details: object) -> AuditEvent:
        event = AuditEvent(event_type=event_type, actor=actor, details=details)
        self.events.append(event)
        return event

    def by_type(self, event_type: AuditEventType) -> list[AuditEvent]:
        return [event for event in self.events if event.event_type == event_type]
