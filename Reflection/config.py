"""
Reflection（类型三：反思/合成写入）防御体系全局配置。

整体风格与 MINJA 对齐：
  - 每个防御节点拥有独立配置段
  - 主管线只负责编排，不内嵌魔法数字
  - 支持通过配置做消融实验和策略替换
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class D1Config:
    # "keyword"   : 关键词/模式快速筛查，轻量、可解释
    # "hybrid"    : 关键词 + 工作流模式 + 来源上下文加权，默认推荐
    # "pattern_graph" : 将多类可疑模式做图式组合，适合更隐蔽的 summary steering
    strategy: Literal["keyword", "hybrid", "pattern_graph"] = "hybrid"

    flag_threshold: float = 0.25
    block_threshold: float = 0.60
    graph_bonus: float = 0.15
    enabled: bool = True


@dataclass
class D2Config:
    # "lexical" : 基于词面证据的接地性审计
    # "hybrid"  : 词面覆盖 + 句级证据选择 + 来源可信度
    # "evidence_graph" : top-k 证据句图聚合，适合摘要接地性核查
    strategy: Literal["lexical", "hybrid", "evidence_graph"] = "hybrid"

    min_support_score: float = 0.35
    min_provenance_score: float = 0.55
    min_sentence_score: float = 0.22
    max_evidence_sentences: int = 3
    independent_witness_bonus: float = 0.08
    enabled: bool = True


@dataclass
class D3Config:
    # "slot_conflict" : 以槽位冲突为核心的一致性核查
    # "consensus"     : 冲突 + 既有高可信记忆共识联合判断
    strategy: Literal["slot_conflict", "consensus"] = "slot_conflict"

    enabled: bool = True


@dataclass
class D4Config:
    # "rule_policy"   : 基于规则的长时记忆存储策略
    # "strict_privacy": 更严格地限制 contact/task 类事实进入长期记忆
    strategy: Literal["rule_policy", "strict_privacy"] = "rule_policy"

    block_instruction: bool = True
    block_credential: bool = True
    flag_ephemeral_task: bool = True
    enabled: bool = True


@dataclass
class D5Config:
    # "weighted_gate" : 汇总多节点风险分数的加权决策门
    # "strict_gate"   : 对低 provenance 或高注入信号更保守
    strategy: Literal["weighted_gate", "strict_gate"] = "weighted_gate"

    quarantine_threshold: float = 0.35
    rejection_threshold: float = 0.60
    injection_weight: float = 0.35
    provenance_weight: float = 0.35
    contradiction_weight: float = 0.20
    policy_weight: float = 0.10
    min_accept_witnesses: int = 1
    enabled: bool = True


@dataclass
class ProvenanceConfig:
    # "hmac"    : 对称签名，部署简单，默认推荐
    # "ed25519" : 非对称签名，可公开验证，需 cryptography 依赖
    signing_backend: Literal["hmac", "ed25519"] = "hmac"

    # "source_lattice" : 仅基于证据来源的完整性格
    # "risk_aware"     : 在来源格上叠加写入时风险信号
    integrity_mode: Literal["source_lattice", "risk_aware"] = "risk_aware"

    hmac_secret: Optional[bytes] = None
    private_key_b64: Optional[str] = None
    enabled: bool = True


@dataclass
class RetrievalConfig:
    # "verify_only"    : 仅做签名验证和基础重排
    # "trust_rank"     : 验签 + 基于 provenance 的信任重排
    # "hubness_cluster": trust_rank + 协调检索/高频异常检测
    strategy: Literal["verify_only", "trust_rank", "hubness_cluster"] = "hubness_cluster"

    max_results: int = 5
    similarity_threshold: float = 0.55
    cluster_min_size: int = 2
    hubness_alpha: float = 2.0
    low_integrity_penalty: float = 0.35
    downweight_factor: float = 0.40
    enabled: bool = True


@dataclass
class PipelineConfig:
    """Reflection 主管线汇总配置。"""

    d1: D1Config = field(default_factory=D1Config)
    d2: D2Config = field(default_factory=D2Config)
    d3: D3Config = field(default_factory=D3Config)
    d4: D4Config = field(default_factory=D4Config)
    d5: D5Config = field(default_factory=D5Config)
    provenance: ProvenanceConfig = field(default_factory=ProvenanceConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)

    audit_log_path: Optional[str] = "reflection_audit.jsonl"

    @classmethod
    def unsafe(cls) -> "PipelineConfig":
        """构造无防护基线配置，用于消融与端到端对照。"""

        return cls(
            d1=D1Config(enabled=False),
            d2=D2Config(enabled=False, min_support_score=0.0, min_provenance_score=0.0),
            d3=D3Config(enabled=False),
            d4=D4Config(enabled=False),
            d5=D5Config(
                enabled=False,
                quarantine_threshold=1.0,
                rejection_threshold=1.0,
                injection_weight=0.0,
                provenance_weight=0.0,
                contradiction_weight=0.0,
                policy_weight=0.0,
            ),
            provenance=ProvenanceConfig(enabled=False),
            retrieval=RetrievalConfig(enabled=False),
            audit_log_path=None,
        )
