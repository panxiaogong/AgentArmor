"""Reflection 防御体系共享数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from Reflection.config import PipelineConfig


class SourceType(str, Enum):
    """Origin of a piece of text before reflection happens."""

    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    WEB = "web"
    SYSTEM = "system"
    MEMORY = "memory"


class FactCategory(str, Enum):
    """Coarse fact type used by policy and consistency checks."""

    IDENTITY = "identity"
    PREFERENCE = "preference"
    CONTACT = "contact"
    TASK = "task"
    INSTRUCTION = "instruction"
    CREDENTIAL = "credential"
    OTHER = "other"


class DecisionAction(str, Enum):
    """Final write decision for a synthesized fact."""

    ACCEPT = "accept"
    QUARANTINE = "quarantine"
    REJECT = "reject"


@dataclass
class DefenseVerdict:
    """
    单个防御节点的判决结果。

    与 MINJA 保持相同语义：
      PASS  = 通过，继续后续节点
      FLAG  = 软告警，升级到更精细的后续判断
      BLOCK = 硬拦截，终止当前事实写入
      ASK   = 预留给需人工确认的高风险模式
    """

    node: str
    passed: bool
    score: float
    reason: str
    action: Literal["PASS", "FLAG", "BLOCK", "ASK"]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationTurn:
    """One raw turn that may later be summarized into long-term memory."""

    turn_id: str
    source: SourceType
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReflectionCandidate:
    """Output produced by the background reflection/summarization step."""

    summary_text: str
    fact_texts: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReflectionContext:
    """
    一次反思写入尝试的上下文封装。

    反思型记忆的关键不是“谁直接调用 save”，而是：
      原始 turns -> 后台反思候选 -> 长时记忆写入
    因此这里把原始对话与反思候选绑定在一起，供 D1-D5 共用。
    """

    turns: List[ConversationTurn]
    candidate: ReflectionCandidate
    triggering_query: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def joined_turn_text(self) -> str:
        """将原始 turns 合并成单段文本，供快速规则检测使用。"""

        return "\n".join(turn.text for turn in self.turns)


@dataclass
class InjectionAssessment:
    """Global signal collected before the pipeline inspects each fact."""

    score: float = 0.0
    hard_block: bool = False
    matched_rules: List[str] = field(default_factory=list)


@dataclass
class FactAssessment:
    """Per-fact view of the evidence accumulated by each defense node."""

    fact_text: str
    category: FactCategory
    evidence_turn_ids: List[str] = field(default_factory=list)
    support_score: float = 0.0
    provenance_score: float = 0.0
    contradiction_score: float = 0.0
    injection_score: float = 0.0
    policy_score: float = 0.0
    final_risk: float = 0.0
    action: DecisionAction = DecisionAction.QUARANTINE
    reasons: List[str] = field(default_factory=list)
    verdicts: List[DefenseVerdict] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryRecord:
    """Trusted memory that survived the defense pipeline."""

    record_id: str
    fact_text: str
    category: FactCategory
    provenance_score: float
    evidence_turn_ids: List[str]
    source_summary: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Full result of one reflection-write attempt."""

    candidate: ReflectionCandidate
    injection: InjectionAssessment = field(default_factory=InjectionAssessment)
    verdicts: List[DefenseVerdict] = field(default_factory=list)
    facts: List[FactAssessment] = field(default_factory=list)
    accepted_records: List[MemoryRecord] = field(default_factory=list)
    quarantined_facts: List[FactAssessment] = field(default_factory=list)
    rejected_facts: List[FactAssessment] = field(default_factory=list)
    blocked_by: Optional[str] = None

    @property
    def blocked(self) -> bool:
        return bool(self.blocked_by or self.rejected_facts or self.quarantined_facts)
