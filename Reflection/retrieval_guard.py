"""
Reflection 检索阶段防御。

类型三攻击不止发生在写入那一刻，真正危险的是：
  “毒化后的长期记忆在未来被当作可信知识再次取回。”

因此检索阶段至少要做三件事：
1. 验签，确认条目没有被静态篡改；
2. 基于 provenance 重排，低完整性/低接地性内容不应轻易排到最前；
3. 检测协调检索异常，例如同一 poisoned page 蒸馏出的多条事实集体命中。
"""
from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Dict, List

from Reflection.config import RetrievalConfig
from Reflection.provenance import ProvenanceBinder
from Reflection.types import (
    DefenseVerdict,
    IntegrityLabel,
    MemoryRecord,
    RetrievedMemory,
    RetrievalResult,
)
from Reflection.utils import clamp, lexical_support, unique_tokens


class RetrievalHubnessTracker:
    """记录长期记忆条目在历史检索中被命中的频次。"""

    def __init__(self) -> None:
        self._counts: Dict[str, int] = defaultdict(int)
        self._query_hashes: Dict[str, set[str]] = defaultdict(set)

    def record(self, query: str, record_ids: List[str]) -> None:
        query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()
        for record_id in record_ids:
            self._counts[record_id] += 1
            self._query_hashes[record_id].add(query_hash)

    def get_count(self, record_id: str) -> int:
        return self._counts.get(record_id, 0)

    def get_query_variety(self, record_id: str) -> int:
        return len(self._query_hashes.get(record_id, set()))

    def stats(self) -> tuple[float, float]:
        if not self._counts:
            return 0.0, 1.0
        counts = list(self._counts.values())
        mean = sum(counts) / len(counts)
        var = sum((count - mean) ** 2 for count in counts) / len(counts)
        return mean, math.sqrt(var) if var > 0 else 1.0


class RetrievalDefenseGuard:
    """执行检索路径的验签、重排和协调异常检测。"""

    def __init__(self, config: RetrievalConfig, binder: ProvenanceBinder) -> None:
        self.cfg = config
        self.binder = binder
        self.hubness = RetrievalHubnessTracker()

    def check(self, query: str, ranked_records: List[tuple[float, MemoryRecord]]) -> RetrievalResult:
        if not self.cfg.enabled:
            return RetrievalResult(
                entries=[
                    RetrievedMemory(record=record, query_score=score, weight=score)
                    for score, record in ranked_records[: self.cfg.max_results]
                ],
                verdicts=[DefenseVerdict(node="R1", passed=True, score=1.0, reason="retrieval defense 已禁用", action="PASS")],
            )

        top_records = ranked_records[: self.cfg.max_results]
        entries: List[RetrievedMemory] = []
        verdicts: List[DefenseVerdict] = []
        tampered_count = 0

        for query_score, record in top_records:
            verified, reason = self.binder.verify(record)
            if not verified:
                tampered_count += 1
                verdicts.append(
                    DefenseVerdict(
                        node="R1",
                        passed=False,
                        score=0.0,
                        reason=reason,
                        action="BLOCK",
                        metadata={"record_id": record.record_id},
                    )
                )
                continue

            weight = self._base_weight(query_score, record)
            entries.append(RetrievedMemory(record=record, query_score=query_score, weight=weight, verified=True))

        if self.cfg.strategy == "verify_only":
            entries.sort(key=lambda item: item.weight, reverse=True)
            self.hubness.record(query, [entry.record.record_id for entry in entries])
            if not verdicts:
                verdicts.append(DefenseVerdict(node="R1", passed=True, score=1.0, reason="验签通过，未做额外检索审计", action="PASS"))
            return RetrievalResult(entries=entries, verdicts=verdicts, tampered_count=tampered_count)

        self._apply_trust_ranking(entries, verdicts)
        if self.cfg.strategy == "hubness_cluster":
            self._apply_hubness_and_cluster(query, entries, verdicts)
        elif self.cfg.strategy != "trust_rank":
            raise ValueError(f"未知 retrieval strategy: {self.cfg.strategy}")

        entries.sort(key=lambda item: item.weight, reverse=True)
        self.hubness.record(query, [entry.record.record_id for entry in entries])
        if not verdicts:
            verdicts.append(DefenseVerdict(node="R1", passed=True, score=1.0, reason="未检测到异常检索模式", action="PASS"))
        return RetrievalResult(entries=entries, verdicts=verdicts, tampered_count=tampered_count)

    def _base_weight(self, query_score: float, record: MemoryRecord) -> float:
        provenance_score = record.provenance_score
        label_value = record.provenance.label if record.provenance else IntegrityLabel.UNTRUSTED
        label_boost = int(label_value) / int(IntegrityLabel.TRUSTED)
        return clamp((0.60 * query_score) + (0.25 * provenance_score) + (0.15 * label_boost))

    def _apply_trust_ranking(self, entries: List[RetrievedMemory], verdicts: List[DefenseVerdict]) -> None:
        for entry in entries:
            provenance = entry.record.provenance
            if provenance is None:
                entry.flagged = True
                entry.flag_reason = "缺少 provenance，检索阶段降权"
                entry.weight *= (1.0 - self.cfg.downweight_factor)
                verdicts.append(
                    DefenseVerdict(
                        node="R1",
                        passed=False,
                        score=entry.weight,
                        reason=entry.flag_reason,
                        action="FLAG",
                        metadata={"record_id": entry.record.record_id},
                    )
                )
                continue

            if provenance.label == IntegrityLabel.UNTRUSTED:
                entry.flagged = True
                entry.flag_reason = "低完整性 provenance 条目在检索阶段降权"
                entry.weight *= (1.0 - self.cfg.low_integrity_penalty)
                verdicts.append(
                    DefenseVerdict(
                        node="R1",
                        passed=False,
                        score=entry.weight,
                        reason=entry.flag_reason,
                        action="FLAG",
                        metadata={"record_id": entry.record.record_id, "label": provenance.label.name},
                    )
                )

    def _apply_hubness_and_cluster(
        self,
        query: str,
        entries: List[RetrievedMemory],
        verdicts: List[DefenseVerdict],
    ) -> None:
        mean, sigma = self.hubness.stats()
        threshold = mean + self.cfg.hubness_alpha * sigma

        for entry in entries:
            count = self.hubness.get_count(entry.record.record_id)
            variety = self.hubness.get_query_variety(entry.record.record_id)
            provenance = entry.record.provenance
            if provenance is None:
                continue
            if count > threshold and provenance.label != IntegrityLabel.TRUSTED and variety >= 2:
                entry.flagged = True
                entry.flag_reason = f"Hubness 异常：count={count}, variety={variety}"
                entry.weight *= (1.0 - self.cfg.downweight_factor)
                verdicts.append(
                    DefenseVerdict(
                        node="R1",
                        passed=False,
                        score=float(count),
                        reason=entry.flag_reason,
                        action="FLAG",
                        metadata={"record_id": entry.record.record_id, "count": count, "threshold": round(threshold, 2)},
                    )
                )

        # 对同一 triggering_query_hash 产出的多条相似事实做协调聚类检测。
        clusters: Dict[str, List[RetrievedMemory]] = defaultdict(list)
        for entry in entries:
            provenance = entry.record.provenance
            if provenance is None:
                continue
            clusters[provenance.triggering_query_hash].append(entry)

        query_tokens = unique_tokens(query)
        for trigger_hash, cluster_entries in clusters.items():
            if len(cluster_entries) < self.cfg.cluster_min_size:
                continue
            avg_similarity = self._average_cluster_similarity(cluster_entries)
            alignment = self._cluster_alignment(query_tokens, cluster_entries)
            low_integrity_count = sum(
                1
                for entry in cluster_entries
                if entry.record.provenance and entry.record.provenance.label == IntegrityLabel.UNTRUSTED
            )
            if avg_similarity >= self.cfg.similarity_threshold and low_integrity_count > 0:
                for entry in cluster_entries:
                    entry.flagged = True
                    entry.flag_reason = "同一来源反思簇在检索阶段被识别为协调命中"
                    entry.weight *= (1.0 - self.cfg.downweight_factor)
                verdicts.append(
                    DefenseVerdict(
                        node="R1",
                        passed=False,
                        score=avg_similarity,
                        reason=(
                            f"检测到来自同一 triggering_query 的协调检索簇，"
                            f"size={len(cluster_entries)}, sim={avg_similarity:.3f}, align={alignment:.3f}"
                        ),
                        action="FLAG",
                        metadata={
                            "triggering_query_hash": trigger_hash,
                            "cluster_size": len(cluster_entries),
                            "avg_similarity": round(avg_similarity, 3),
                            "alignment": round(alignment, 3),
                        },
                    )
                )

    @staticmethod
    def _average_cluster_similarity(entries: List[RetrievedMemory]) -> float:
        if len(entries) < 2:
            return 0.0
        scores = []
        for left_index in range(len(entries)):
            for right_index in range(left_index + 1, len(entries)):
                left_tokens = unique_tokens(entries[left_index].record.fact_text)
                right_tokens = unique_tokens(entries[right_index].record.fact_text)
                denom = len(left_tokens | right_tokens) or 1
                scores.append(len(left_tokens & right_tokens) / denom)
        return sum(scores) / len(scores) if scores else 0.0

    @staticmethod
    def _cluster_alignment(query_tokens: set[str], entries: List[RetrievedMemory]) -> float:
        if not entries:
            return 0.0
        scores = []
        for entry in entries:
            fact_tokens = unique_tokens(entry.record.fact_text)
            denom = len(query_tokens | fact_tokens) or 1
            scores.append(len(query_tokens & fact_tokens) / denom)
        return sum(scores) / len(scores)

