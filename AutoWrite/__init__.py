"""
AutoWrite 防御包公共接口。

外部代码只需从本包导入，无需关心内部模块结构：

    from AgentArmor.AutoWrite import (
        AutoWriteDefensePipeline,
        AutoWriteConfig,
    )

节点类也可直接导入，用于单独测试或自定义组合：

    from AgentArmor.AutoWrite import (
        DAPreWriteSanitizer,
        DBSelectiveWritePolicy,
        DCStorageIntegrityChain,
        DDTemporalDecayReranker,
        DERetrievalAlignmentVerifier,
        DFMemoryDistributionMonitor,
    )
"""
from .config import (
    AutoWriteConfig,
    DAConfig,
    DBConfig,
    DCConfig,
    DDConfig,
    DEConfig,
    DFConfig,
)
from .da_token_sanitizer import DAPreWriteSanitizer
from .db_selective_write import DBSelectiveWritePolicy, MemoryIndex, MockMemoryIndex
from .dc_integrity_chain import DCStorageIntegrityChain
from .dd_temporal_decay import DDTemporalDecayReranker
from .de_retrieval_align import DERetrievalAlignmentVerifier
from .df_distribution_monitor import DFMemoryDistributionMonitor
from .pipeline import AutoWriteDefensePipeline
from .types import (
    CandidateEntry,
    ChainedCandidateEntry,
    ChainedRetrievedEntry,
    DefenseVerdict,
    RetrievedEntry,
    WriteContext,
)

__all__ = [
    # 管线
    "AutoWriteDefensePipeline",
    # 汇总配置
    "AutoWriteConfig",
    # 节点配置
    "DAConfig",
    "DBConfig",
    "DCConfig",
    "DDConfig",
    "DEConfig",
    "DFConfig",
    # 节点类
    "DAPreWriteSanitizer",
    "DBSelectiveWritePolicy",
    "DCStorageIntegrityChain",
    "DDTemporalDecayReranker",
    "DERetrievalAlignmentVerifier",
    "DFMemoryDistributionMonitor",
    # 辅助类
    "MemoryIndex",
    "MockMemoryIndex",
    # 类型
    "CandidateEntry",
    "ChainedCandidateEntry",
    "ChainedRetrievedEntry",
    "DefenseVerdict",
    "RetrievedEntry",
    "WriteContext",
]
