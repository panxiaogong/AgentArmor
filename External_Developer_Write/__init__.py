# External_Developer_Write — Type 4 Defense Pipeline
#
# 防御体系针对"外部批量写入"（External/Developer Write）类型的记忆注入攻击。
# 覆盖 3 个攻击节点: 供应链投毒 (PoisonedRAG)、语义混淆注入、触发词检索劫持 (AgentPoison)
#
# 包含 7 个子防御节点 (SP1-SP7) + 管线编排 (Pipeline) + 共享数据结构 (types) + 工具函数 (utils)

from .types import (
    Document, DocumentEmbedding, CleanStats, DetectionResult,
    ChunkInfo, RetrievedDoc, RetrievalGroup, NLIResult, SemanticTriple,
    PipelineConfig, DefenseAlert
)

from .sp1_embedding_anomaly import EmbeddingAnomalyDetector
from .sp2_content_perplexity import ContentPerplexityAnalyzer
from .sp3_cross_chunk_coherence import CrossChunkCoherenceVerifier
from .sp4_trigger_region import TriggerRegionDetector
from .sp5_robust_aggregation import RobustAggregationRetriever
from .sp6_post_retrieval_verifier import PostRetrievalVerifier
from .sp7_semantic_graph import SemanticDependencyGraphAnalyzer
from .pipeline import DefensePipeline

__all__ = [
    "Document", "DocumentEmbedding", "CleanStats", "DetectionResult",
    "ChunkInfo", "RetrievedDoc", "RetrievalGroup", "NLIResult", "SemanticTriple",
    "PipelineConfig", "DefenseAlert",
    "EmbeddingAnomalyDetector",
    "ContentPerplexityAnalyzer",
    "CrossChunkCoherenceVerifier",
    "TriggerRegionDetector",
    "RobustAggregationRetriever",
    "PostRetrievalVerifier",
    "SemanticDependencyGraphAnalyzer",
    "DefensePipeline",
]
