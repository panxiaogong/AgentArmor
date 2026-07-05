"""防御节点 4：冲突检测。

共享记忆中最危险的情况之一是攻击者写入与既有事实冲突的内容，
并通过 recency/importance/similarity 让毒化内容覆盖原事实。
本模块只负责判断“是否冲突”和“置信度如何计算”。
"""

from __future__ import annotations

from .memory_store import MemoryStore
from .risk import source_reputation
from .types import CandidateFact, MemoryRecord, TrustLevel


CONFLICT_EPSILON = 0.15


def belief(value: CandidateFact | MemoryRecord) -> float:
    """计算候选事实或记忆的简化置信度。

    公式：
        belief = 0.5 * trust + 0.3 * evidence + 0.2 * source_reputation

    它不是事实真实性证明，而是冲突处理时的排序依据。
    """

    if isinstance(value, MemoryRecord):
        trust = value.trust
        evidence_count = len(value.evidence)
        source = value.source
    else:
        trust = value.trust
        evidence_count = 1 if value.evidence_span else 0
        source = value.source

    trust_score = int(trust) / int(TrustLevel.TRUSTED)
    evidence_score = min(evidence_count / 3, 1.0)
    reputation_score = source_reputation(source)

    return 0.50 * trust_score + 0.30 * evidence_score + 0.20 * reputation_score


def has_conflict(candidate: CandidateFact, store: MemoryStore) -> bool:
    """检测同一 subject-predicate 下 object 是否冲突。

    如果新旧事实冲突且置信度差距不明显，就不允许自动覆盖。
    """

    existing_records = store.find_by_subject_predicate(
        subject=candidate.subject,
        predicate=candidate.predicate,
    )

    for old in existing_records:
        if old.content.object.strip().lower() == candidate.object.strip().lower():
            continue

        confidence_gap = abs(belief(old) - belief(candidate))
        if confidence_gap < CONFLICT_EPSILON:
            return True

    return False
