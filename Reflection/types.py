"""Reflection 防御体系共享数据结构。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum, IntEnum
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


class IntegrityLabel(IntEnum):
    """
    反思型记忆的完整性标签。

    语义与 MINJA 对齐：
      TRUSTED   > CANDIDATE > UNTRUSTED
    一条事实如果主要来自 web/tool 等低可信来源，就不应在检索阶段
    与用户明确陈述的高可信长期偏好享有同等信任级别。
    """

    UNTRUSTED = 1
    CANDIDATE = 2
    TRUSTED = 3


def lattice_join(labels: List[IntegrityLabel]) -> IntegrityLabel:
    """完整性格的下确界：任一低完整性来源都会拉低最终标签。"""

    if not labels:
        return IntegrityLabel.UNTRUSTED
    return IntegrityLabel(min(int(label) for label in labels))


def default_integrity_for_source(source: "SourceType") -> IntegrityLabel:
    """给原始来源一个默认完整性先验，供 provenance 绑定和检索重排使用。"""

    if source == SourceType.SYSTEM:
        return IntegrityLabel.TRUSTED
    if source in {SourceType.USER, SourceType.ASSISTANT, SourceType.MEMORY}:
        return IntegrityLabel.CANDIDATE
    return IntegrityLabel.UNTRUSTED


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
class SourceLabel:
    """一条被选中证据的来源标签。"""

    turn_id: str
    source_type: str
    label: IntegrityLabel
    selector_score: float
    excerpt: str = ""


@dataclass
class ProvenanceTag:
    """
    写入成功后绑定到 MemoryRecord 的可追溯 provenance 信息。

    这里既记录“事实从哪里来”，也记录“它为什么能被写入”。
    比赛里这类审计链很重要，因为类型三攻击的隐蔽点就在于：
    毒化内容会被二次 LLM 处理后伪装成“看起来像系统自己记住的事实”。
    """

    label: IntegrityLabel
    source_types: List[str]
    evidence_turn_ids: List[str]
    triggering_query_hash: str
    summary_hash: str
    write_time: float
    signature: str
    sign_algo: str
    risk_at_write: float
    verdict_trace: List[str] = field(default_factory=list)


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
    source_labels: List[SourceLabel] = field(default_factory=list)


@dataclass
class MemoryRecord:
    """Trusted memory that survived the defense pipeline."""

    record_id: str
    fact_text: str
    category: FactCategory
    provenance_score: float
    evidence_turn_ids: List[str]
    source_summary: str
    provenance: Optional[ProvenanceTag] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def content_hash(self) -> str:
        """对事实正文做稳定哈希，供 provenance 签名与检索验签使用。"""

        return hashlib.sha256(self.fact_text.encode("utf-8")).hexdigest()

    def summary_hash(self) -> str:
        return hashlib.sha256(self.source_summary.encode("utf-8")).hexdigest()


@dataclass
class RetrievedMemory:
    """检索路径中的记录视图：在原始 MemoryRecord 上叠加验签/降权信息。"""

    record: MemoryRecord
    query_score: float
    weight: float
    verified: bool = True
    flagged: bool = False
    flag_reason: str = ""


@dataclass
class RetrievalResult:
    """检索路径执行结果。"""

    entries: List[RetrievedMemory] = field(default_factory=list)
    verdicts: List[DefenseVerdict] = field(default_factory=list)
    tampered_count: int = 0


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
