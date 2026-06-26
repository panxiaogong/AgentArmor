"""
MINJA 防御体系包初始化。

目录结构：
  config.py            — 全局配置（所有可调参数）
  types.py             — 共享数据结构
  d1_query_intent.py   — D1：查询意图检测
  d2_causal_write.py   — D2：写入决策因果归因
  d3_prospective_sim.py — D3：前瞻行为仿真
  d4_provenance.py     — D4：IFC 溯源标签绑定
  d5_retrieval_audit.py — D5：检索集合 Hubness + 一致性图
  d6_execution_align.py — D6：执行前任务对齐核查
  pipeline.py          — 六节点纵深防御主管线
"""
from .config import (
    D1Config, D2Config, D3Config, D4Config, D5Config, D6Config,
    PipelineConfig,
)
from .types import (
    IntegrityLabel, lattice_join,
    DefenseVerdict,
    SourceLabel, ProvenanceTag,
    CandidateEntry, WriteContext,
    ToolCallRequest, RetrievedEntry,
)
from .d1_query_intent import D1QueryIntentDetector
from .d2_causal_write import D2CausalWriteAuditor
from .d3_prospective_sim import D3ProspectiveSimulator
from .d4_provenance import D4ProvenanceBinder
from .d5_retrieval_audit import D5RetrievalSetAuditor, HubnessTracker
from .d6_execution_align import D6ExecutionAlignmentGuard
from .pipeline import MINJADefensePipeline, WriteResult, RetrievalResult, ToolCallResult

__all__ = [
    # 配置
    "PipelineConfig", "D1Config", "D2Config", "D3Config",
    "D4Config", "D5Config", "D6Config",
    # 类型
    "IntegrityLabel", "lattice_join", "DefenseVerdict",
    "SourceLabel", "ProvenanceTag", "CandidateEntry",
    "WriteContext", "ToolCallRequest", "RetrievedEntry",
    # 节点
    "D1QueryIntentDetector", "D2CausalWriteAuditor",
    "D3ProspectiveSimulator", "D4ProvenanceBinder",
    "D5RetrievalSetAuditor", "HubnessTracker",
    "D6ExecutionAlignmentGuard",
    # 管线
    "MINJADefensePipeline", "WriteResult", "RetrievalResult", "ToolCallResult",
]
