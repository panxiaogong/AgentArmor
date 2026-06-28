"""
types.py — 防御体系共享的数据结构

所有防御节点 (SP1–SP7) 共用的数据类、枚举和类型别名。
每个类都附有详细的字段说明，确保模块间的接口清晰一致。
"""

from dataclasses import dataclass, field
from typing import List, Optional, Set, Dict, Any, Tuple
from enum import Enum


# ============================================================
# 基础文档类
# ============================================================

@dataclass
class Document:
    """文档原始信息 —— 上传时传入的最小单位。

    Fields:
        doc_id:    全局唯一标识（如 UUID 或业务主键）
        content:   原始文本内容（未分块、未嵌入）
        source:    来源标签 ('upload' / 'api' / 'sync' / 'scrape')
        timestamp: Unix 时间戳（秒）
        metadata:  额外元数据（来源 URL、作者、格式等）
    """
    doc_id: str
    content: str
    source: str = "upload"
    timestamp: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentEmbedding:
    """文档嵌入信息 —— 嵌入后的向量化表示。

    Fields:
        doc_id:       关联的文档 ID
        vector:       n 维嵌入向量（如 768/1024 维）
        model_name:   使用的嵌入模型名称（用于溯源和跨模型校验）
        chunk_index:  如果该嵌入来自某个 chunk，记录其在源文档中的序号
    """
    doc_id: str
    vector: List[float]
    model_name: str = "default"
    chunk_index: Optional[int] = None


@dataclass
class ChunkInfo:
    """分块信息 —— 文档分块后的单个 chunk。

    Fields:
        doc_id:      所属文档 ID
        chunk_id:    块唯一标识（如 'doc_xxx_chunk_003'）
        chunk_index: 在文档中的序号（从 0 开始）
        content:     块文本内容
        embedding:   块嵌入向量（可选，未嵌入时为 None）
    """
    doc_id: str
    chunk_id: str
    chunk_index: int
    content: str
    embedding: Optional[List[float]] = None


# ============================================================
# 检索相关类
# ============================================================

@dataclass
class RetrievedDoc:
    """单条检索结果 —— 向量数据库返回的匹配文档。

    Fields:
        doc_id:      文档 ID
        content:     文档内容
        embedding:   文档嵌入向量（用于计算一致性）
        score:       原始检索相似度（如余弦距离，[0,1] 越大越相关）
        trust_score: SP6 处理后填充的信任得分（None 表示未处理）
    """
    doc_id: str
    content: str
    embedding: List[float]
    score: float = 0.0
    trust_score: Optional[float] = None


@dataclass
class RetrievalGroup:
    """检索结果分组 —— SP5 隔离-聚合框架中的基本单位。

    Fields:
        group_id:    组编号（0 .. m-1）
        docs:        该组包含的检索文档列表
        response:    LLM 基于该组文档的生成结果（隔离推理后填充）
        keywords:    从 response 中提取的关键词集合
        is_abstained:该组是否弃权（拒绝回答 / 信息不足）
    """
    group_id: int
    docs: List[RetrievedDoc] = field(default_factory=list)
    response: Optional[str] = None
    keywords: Optional[Set[str]] = None
    is_abstained: bool = False


@dataclass
class NLIResult:
    """自然语言推理结果 —— 用于 SP6 模型一致性检查。

    Fields:
        entailment:     蕴涵概率（p 支持）
        neutral:        中性概率
        contradiction:  矛盾概率
    """
    entailment: float = 0.0
    neutral: float = 0.0
    contradiction: float = 0.0


@dataclass
class SemanticTriple:
    """语义三元组 —— 用于 SP7 语义依赖图分析（EIRE 格式）。

    Fields:
        head:     主体实体（如 "RAG 系统"）
        relation: 关系（如 "使用"、"依赖"、"包含"）
        tail:     客体实体（如 "向量数据库"）
    """
    head: str
    relation: str
    tail: str

    def __hash__(self) -> int:
        return hash((self.head, self.relation, self.tail))


# ============================================================
# 统计与检测结果类
# ============================================================

@dataclass
class CleanStats:
    """正常文档统计信息 —— SP1 预处理阶段从干净文档集计算得出。

    Fields:
        mu:                   均值向量，shape=(dim,)
        sigma_inv:            协方差矩阵的逆，shape=(dim, dim)
        sigma:                协方差矩阵（用于 Cholesky 求解）
        max_train_mahal:      训练集最大马氏距离（用于归一化）
        max_train_lof_norm:   训练集最大 LOF 归一化值
        reference_embeddings: 参考嵌入集（用于在线 LOF 计算）
    """
    mu: List[float]
    sigma_inv: List[List[float]]
    sigma: List[List[float]]
    max_train_mahal: float = 0.0
    max_train_lof_norm: float = 1.0
    reference_embeddings: List[List[float]] = field(default_factory=list)


@dataclass
class DetectionResult:
    """单一检测结果 —— 所有防御节点的统一输出格式。

    Fields:
        doc_id:       被检测的文档 ID（空字符串表示不适用）
        is_anomaly:   是否判定为异常（True = 可疑/攻击）
        anomaly_score:连续异常得分（[0, 1] 或原始值，越高越可疑）
        reason:       人类可读的判定原因
        details:      各子指标的详细数值（用于可解释性分析）
    """
    doc_id: str = ""
    is_anomaly: bool = False
    anomaly_score: float = 0.0
    reason: str = "normal"
    details: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# 枚举
# ============================================================

class PipelineConfig(Enum):
    """管线配置模式 —— 控制哪些防御节点被启用。

    从低延迟/基础安全到高延迟/最高安全共 5 个级别：
        CONFIG_1_FAST:       SP1 + SP2，快速基础保护
        CONFIG_2_STANDARD:   SP1 + SP2 + SP3，标准安全
        CONFIG_3_FULL_UPLOAD:SP1 + SP2 + SP3 + SP4，全面上传防护
        CONFIG_4_RETRIEVAL:  SP5 + SP6，仅检索阶段保护
        CONFIG_5_MAX:        SP1~SP7 全开，最高安全等级
    """
    CONFIG_1_FAST = "config_1"
    CONFIG_2_STANDARD = "config_2"
    CONFIG_3_FULL_UPLOAD = "config_3"
    CONFIG_4_RETRIEVAL = "config_4"
    CONFIG_5_MAX = "config_5"


@dataclass
class DefenseAlert:
    """防御告警 —— 管线检测到异常时生成的告警记录。

    Fields:
        node:      触发告警的防御节点名称（如 'SP1', 'SP2'）
        reason:    触发原因描述
        score:     异常得分
        doc_id:    关联的文档 ID
        timestamp: Unix 时间戳
        details:   额外上下文信息
    """
    node: str
    reason: str
    score: float = 0.0
    doc_id: str = ""
    timestamp: int = 0
    details: Dict[str, Any] = field(default_factory=dict)
