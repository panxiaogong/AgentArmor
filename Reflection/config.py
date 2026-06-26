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
    # "hybrid"    : 关键词 + 风险模式加权，默认推荐
    # "llm_judge" : 预留给真实 LLM 二分类器
    strategy: Literal["keyword", "hybrid", "llm_judge"] = "hybrid"

    flag_threshold: float = 0.25
    block_threshold: float = 0.60
    enabled: bool = True


@dataclass
class D2Config:
    # "lexical" : 基于词面证据的接地性审计
    # "hybrid"  : 当前与 lexical 等价，预留给 embedding / NLI 扩展
    strategy: Literal["lexical", "hybrid"] = "lexical"

    min_support_score: float = 0.35
    min_provenance_score: float = 0.55
    enabled: bool = True


@dataclass
class D3Config:
    # "slot_conflict" : 以槽位冲突为核心的一致性核查
    strategy: Literal["slot_conflict"] = "slot_conflict"

    enabled: bool = True


@dataclass
class D4Config:
    # "rule_policy" : 基于规则的长时记忆存储策略
    strategy: Literal["rule_policy"] = "rule_policy"

    block_instruction: bool = True
    block_credential: bool = True
    flag_ephemeral_task: bool = True
    enabled: bool = True


@dataclass
class D5Config:
    # "weighted_gate" : 汇总多节点风险分数的加权决策门
    strategy: Literal["weighted_gate"] = "weighted_gate"

    quarantine_threshold: float = 0.35
    rejection_threshold: float = 0.60
    injection_weight: float = 0.35
    provenance_weight: float = 0.35
    contradiction_weight: float = 0.20
    policy_weight: float = 0.10
    enabled: bool = True


@dataclass
class PipelineConfig:
    """Reflection 主管线汇总配置。"""

    d1: D1Config = field(default_factory=D1Config)
    d2: D2Config = field(default_factory=D2Config)
    d3: D3Config = field(default_factory=D3Config)
    d4: D4Config = field(default_factory=D4Config)
    d5: D5Config = field(default_factory=D5Config)

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
            audit_log_path=None,
        )
