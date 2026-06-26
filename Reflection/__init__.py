"""
Reflection（类型三：反思/合成写入）防御体系包初始化。

目录结构风格与 MINJA 对齐：
  config.py               — 全局配置
  types.py                — 共享数据结构
  d1_reflection_intent.py — D1：反思注入意图筛查
  d2_grounding.py         — D2：接地性审计
  d3_consistency.py       — D3：一致性核查
  d4_policy.py            — D4：存储策略核查
  d5_write_gate.py        — D5：最终写入闸门
  pipeline.py             — 五节点主管线
"""

from Reflection.agent import ReflectionAgent, UnsafeReflectionAgent
from Reflection.config import D1Config, D2Config, D3Config, D4Config, D5Config, PipelineConfig
from Reflection.d1_reflection_intent import D1ReflectionIntentDetector
from Reflection.d2_grounding import D2GroundingAuditor
from Reflection.d3_consistency import D3ConsistencyAuditor
from Reflection.d4_policy import D4PolicyAuditor
from Reflection.d5_write_gate import D5WriteGate
from Reflection.memory_store import MemoryStore
from Reflection.pipeline import ReflectionDefensePipeline, WriteResult
from Reflection.types import (
    ConversationTurn,
    DefenseVerdict,
    DecisionAction,
    FactAssessment,
    FactCategory,
    MemoryRecord,
    PipelineResult,
    ReflectionCandidate,
    ReflectionContext,
    SourceType,
)

__all__ = [
    "PipelineConfig",
    "D1Config",
    "D2Config",
    "D3Config",
    "D4Config",
    "D5Config",
    "ConversationTurn",
    "DefenseVerdict",
    "DecisionAction",
    "FactAssessment",
    "FactCategory",
    "MemoryRecord",
    "MemoryStore",
    "PipelineResult",
    "ReflectionAgent",
    "ReflectionCandidate",
    "ReflectionContext",
    "ReflectionDefensePipeline",
    "WriteResult",
    "D1ReflectionIntentDetector",
    "D2GroundingAuditor",
    "D3ConsistencyAuditor",
    "D4PolicyAuditor",
    "D5WriteGate",
    "SourceType",
    "UnsafeReflectionAgent",
]
