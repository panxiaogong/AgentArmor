"""
sp1_embedding_anomaly.py — 上传文档的嵌入空间异常检测（SP1）

对应攻击: 节点1 — PoisonedRAG 供应链投毒
对应分析报告: 5.1 节
核心方法: 马氏距离 + LOF 加权融合

攻击者通过上传与目标查询在嵌入空间中高度对齐的恶意文档，使得检索阶段
这些恶意文档总能进入 Top-k。SP1 的核心洞察是: 投毒文档在嵌入空间中表现为
统计异常值——它们与目标查询的"人为对齐"偏离了正常文档的分布流形。

数学基础:
    1. 马氏距离: D_M(x) = sqrt((x-μ)^T Σ^{-1} (x-μ))
       — 考虑了特征间的相关性，捕捉方向性偏离
    2. LOF: LOF(d) = (1/k) · Σ lrd(neighbor) / lrd(d)
       — 检测局部密度异常，不受全局分布形状约束
    3. 收缩估计: Σ̂ = (1-ρ)·Σ_empirical + ρ·diag(Σ_empirical)
       — 处理高维嵌入时协方差估计的数值稳定性
"""

import math
from typing import List, Optional, Tuple

from .types import DocumentEmbedding, CleanStats, DetectionResult
from .utils import (
    EPSILON, mean_vector, mahalanobis_distance, euclidean_distance,
    mean, log2, cholesky_decomposition, dot_product
)


class EmbeddingAnomalyDetector:
    """嵌入空间异常检测器。

    使用马氏距离和 LOF 的双指标融合，检测上传文档是否在嵌入空间中
    表现为统计异常值（潜在投毒文档）。

    使用方式:
        detector = EmbeddingAnomalyDetector()
        clean_stats = detector.train(clean_embeddings)
        result = detector.detect(doc_embedding, clean_stats)
    """

    def __init__(
        self,
        knn_k: int = 20,
        weight_mahal: float = 0.6,
        weight_lof: float = 0.4,
        threshold: float = 0.5,
        shrinkage_rho: float = 0.3,
    ):
        """初始化检测器。

        Args:
            knn_k: LOF 的邻居数 k（默认 20）
            weight_mahal: 马氏距离在融合得分中的权重（默认 0.6）
            weight_lof: LOF 在融合得分中的权重（默认 0.4）
            threshold: 融合得分阈值，超过此值判定为异常
            shrinkage_rho: 协方差收缩估计系数 ρ（默认 0.3）
                           ρ→0: 完全依赖经验协方差
                           ρ→1: 完全退化为对角矩阵
        """
        self.knn_k = knn_k
        self.weight_mahal = weight_mahal
        self.weight_lof = weight_lof
        self.threshold = threshold
        self.shrinkage_rho = shrinkage_rho

    # ---- 公有接口 ----

    def train(self, clean_docs: List[DocumentEmbedding]) -> CleanStats:
        """在已知干净的文档集上训练统计量。

        计算均值、收缩协方差矩阵及其逆，以及归一化常数。
        应在系统初始化时调用一次，后续复用 CleanStats。

        Args:
            clean_docs: 已知干净的文档嵌入列表（至少 > dim 个样本）

        Returns:
            CleanStats 包含均值、协方差逆、归一化常数和参考嵌入集
        """
        if len(clean_docs) < 2:
            raise ValueError(f"训练集至少需要 2 个文档，当前 {len(clean_docs)}")

        vectors = [doc.vector for doc in clean_docs]
        dim = len(vectors[0])
        N = len(vectors)

        # ---- Step 1: 均值向量 ----
        mu = mean_vector(vectors)

        # ---- Step 2: 中心化 ----
        centered = []
        for vec in vectors:
            centered.append([vec[i] - mu[i] for i in range(dim)])

        # ---- Step 3: 经验协方差矩阵 ----
        # Σ_empirical = (1/(N-1)) · X^T · X
        sigma_empirical = [[0.0] * dim for _ in range(dim)]
        for c in centered:
            for i in range(dim):
                for j in range(dim):
                    sigma_empirical[i][j] += c[i] * c[j]
        for i in range(dim):
            for j in range(dim):
                sigma_empirical[i][j] /= (N - 1)

        # ---- Step 4: 收缩估计 ----
        # Σ̂ = (1-ρ) · Σ_empirical + ρ · diag(Σ_empirical)
        # 目的: 当 dim >> N 时，经验协方差的条件数极大，
        # 收缩估计将其拉向对角矩阵，保证正定性和数值稳定性
        sigma_shrunk = [[0.0] * dim for _ in range(dim)]
        for i in range(dim):
            for j in range(dim):
                sigma_shrunk[i][j] = (1 - self.shrinkage_rho) * sigma_empirical[i][j]
        for i in range(dim):
            sigma_shrunk[i][i] += self.shrinkage_rho * sigma_empirical[i][i]

        # ---- Step 5: 计算训练集上的统计量 ----
        max_mahal = 0.0
        max_lof = 0.0

        # 预计算 LOF 用于训练集
        lof_values = self._compute_all_lof(vectors)

        for i, vec in enumerate(vectors):
            # 马氏距离
            d_m = mahalanobis_distance(vec, mu, sigma_shrunk)
            max_mahal = max(max_mahal, d_m)

            # LOF 归一化值
            lof_norm = abs(log2(lof_values[i])) if lof_values[i] != 0 else 0.0
            max_lof = max(max_lof, lof_norm)

        # 防止除零
        max_mahal = max(max_mahal, EPSILON)
        max_lof = max(max_lof, EPSILON)

        return CleanStats(
            mu=mu,
            sigma_inv=[],  # 不直接存储逆，用 sigma + Cholesky 求解
            sigma=sigma_shrunk,
            max_train_mahal=max_mahal,
            max_train_lof_norm=max_lof,
            reference_embeddings=vectors,
        )

    def detect(self, doc_embedding: List[float],
               clean_stats: CleanStats) -> DetectionResult:
        """对单个文档嵌入执行异常检测。

        Args:
            doc_embedding: 待检测文档的嵌入向量
            clean_stats:   train() 返回的正常文档统计量

        Returns:
            DetectionResult 包含异常判定和各子指标值
        """
        # ---- 阶段 1: 马氏距离 ----
        mahal_dist = mahalanobis_distance(
            doc_embedding, clean_stats.mu, clean_stats.sigma
        )
        norm_mahal = mahal_dist / (clean_stats.max_train_mahal + EPSILON)

        # ---- 阶段 2: LOF ----
        lof_score = self._compute_lof(
            doc_embedding, clean_stats.reference_embeddings
        )
        # LOF > 1: 比邻居稀疏（异常离群点）
        # LOF < 1: 比邻居密集（正常聚类内部）
        # 对 AgentPoison 的紧凑聚类，LOF < 1 也是异常信号
        norm_lof = abs(log2(lof_score)) / (clean_stats.max_train_lof_norm + EPSILON)

        # ---- 阶段 3: 融合决策 ----
        anomaly_score = (
            self.weight_mahal * norm_mahal +
            self.weight_lof * norm_lof
        )
        is_anomaly = anomaly_score > self.threshold

        # ---- 构造结果 ----
        reason_parts = []
        if is_anomaly:
            reason_parts.append(
                f"Mahalanobis={mahal_dist:.3f}(norm={norm_mahal:.3f})"
            )
            reason_parts.append(
                f"LOF={lof_score:.3f}(norm={norm_lof:.3f})"
            )
            reason_parts.append(
                f"fused={anomaly_score:.3f}>={self.threshold}"
            )

        return DetectionResult(
            doc_id="",
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,
            reason=" | ".join(reason_parts) if reason_parts else "normal",
            details={
                "mahalanobis_raw": round(mahal_dist, 4),
                "mahalanobis_norm": round(norm_mahal, 4),
                "lof_raw": round(lof_score, 4),
                "lof_norm": round(norm_lof, 4),
                "fused_score": round(anomaly_score, 4),
                "threshold": self.threshold,
                "weight_mahal": self.weight_mahal,
                "weight_lof": self.weight_lof,
            },
        )

    # ---- 内部方法 ----

    def _compute_lof(self, target: List[float],
                     reference_set: List[List[float]]) -> float:
        """计算目标点的 LOF（局部异常因子）。

        LOF(d) = (1/|N_k(d)|) · Σ_{d'∈N_k(d)} lrd(d') / lrd(d)

        其中 lrd(d)（局部可达密度）= 1 / ((1/k) · Σ ||d - d'||₂ + ε)

        Args:
            target:       目标嵌入向量
            reference_set: 参考嵌入集（正常文档）

        Returns:
            LOF 得分
        """
        # 找到 k 个最近邻
        distances = [
            euclidean_distance(target, ref) for ref in reference_set
        ]
        sorted_indices = sorted(
            range(len(distances)), key=lambda i: distances[i]
        )
        # 排除自身（索引 0），取前 k 个邻居
        neighbor_indices = sorted_indices[1:self.knn_k + 1]
        neighbor_dists = [distances[i] for i in neighbor_indices]

        # 目标点的 lrd
        lrd_target = 1.0 / (mean(neighbor_dists) + EPSILON)

        # 每个邻居的 lrd
        lrd_neighbors = []
        for idx in neighbor_indices:
            n_dists = [
                euclidean_distance(reference_set[idx], ref)
                for ref in reference_set
            ]
            n_idx = sorted(
                range(len(n_dists)), key=lambda i: n_dists[i]
            )[1:self.knn_k + 1]
            n_k_dist = [n_dists[i] for i in n_idx]
            lrd_n = 1.0 / (mean(n_k_dist) + EPSILON)
            lrd_neighbors.append(lrd_n)

        return mean(lrd_neighbors) / (lrd_target + EPSILON)

    def _compute_all_lof(self, vectors: List[List[float]]) -> List[float]:
        """批量计算所有向量的 LOF（用于训练阶段的归一化估计）。

        与 _compute_lof 逻辑相同，但使用向量化方式计算。
        """
        N = len(vectors)
        lof_values = []

        for i in range(N):
            distances = [
                euclidean_distance(vectors[i], vectors[j])
                for j in range(N) if j != i
            ]
            # 选最小的 k 个距离
            sorted_d = sorted(distances)[:self.knn_k]

            lrd_i = 1.0 / (mean(sorted_d) + EPSILON)

            # 邻居的 lrd（简化: 用每个邻居到其他点的距离）
            all_indices = sorted(
                range(N),
                key=lambda j: euclidean_distance(vectors[i], vectors[j])
            )[1:self.knn_k + 1]

            lrd_neighbors = []
            for ni in all_indices:
                n_dists = sorted(
                    euclidean_distance(vectors[ni], vectors[k])
                    for k in range(N) if k != ni
                )[:self.knn_k]
                lrd_n = 1.0 / (mean(n_dists) + EPSILON)
                lrd_neighbors.append(lrd_n)

            lof_i = mean(lrd_neighbors) / (lrd_i + EPSILON)
            lof_values.append(lof_i)

        return lof_values
