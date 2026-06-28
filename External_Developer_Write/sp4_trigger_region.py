"""
sp4_trigger_region.py — 嵌入空间触发词区域检测（SP4）

对应攻击: 节点3 — AgentPoison 触发词检索劫持
对应分析报告: 5.4 节
核心方法: LOF + NNR + 聚类紧凑度 三指标融合

AgentPoison 攻击通过优化触发词，将在嵌入空间中"压缩"恶意文档到
一个异常紧凑的区域（由 L_compactness 损失驱动）。SP4 逆向利用
这个攻击特征——越紧凑越可疑。

数学基础:
    1. LOF: LOF(d) = (1/k) · Σ lrd(neighbor) / lrd(d)
       — AgentPoison 文档的 LOF < 1（比邻居更密集）
    2. NNR: NNR_k(d) = avg_knn_dist(d) / global_avg_knn_dist
       — AgentPoison 文档的 NNR << 1（邻域异常密集）
    3. 紧凑度: Compactness(C) = (1/|C|²) · ΣΣ ||e_i - e_j||₂
       — 攻击聚类的类内距离远小于正常聚类

设计决策:
    - 共享 kNN 预计算: 全库只用一次 KD-Tree/Ball-Tree
    - 多指标融合弥补单一指标的盲区
    - 支持周期性全库扫描模式（后台离线任务）
"""

import math
from typing import List, Optional, Tuple, Dict, Any

from .types import DocumentEmbedding, DetectionResult
from .utils import (
    EPSILON, euclidean_distance, cosine_similarity, mean, std, log2
)


class TriggerRegionDetector:
    """嵌入空间触发词区域检测器。

    检测嵌入空间中是否存在 AgentPoison 攻击造成的人为紧凑聚类。
    支持在线单文档检测和离线全库扫描两种模式。

    使用方式:
        detector = TriggerRegionDetector()
        results = detector.scan(all_docs)  # 全库扫描
        # 或逐文档:
        result = detector.detect_single(doc_embedding, all_embeddings)
    """

    def __init__(
        self,
        knn_k: int = 20,
        nnr_threshold: float = 0.6,
        lof_threshold: float = 2.0,
        anomaly_score_threshold: float = 0.7,
        use_clustering: bool = True,
        dbscan_eps: float = 0.5,
        dbscan_min_samples: int = 3,
        weight_lof: float = 0.4,
        weight_nnr: float = 0.4,
        weight_compactness: float = 0.2,
    ):
        """初始化检测器。

        Args:
            knn_k: kNN 参数 k
            nnr_threshold: NNR 异常阈值（低于此值标记紧凑异常）
            lof_threshold: LOF 阈值（|log2(LOF)| > |log2(threshold)| 标记异常）
            anomaly_score_threshold: 融合得分阈值
            use_clustering: 是否使用聚类分析（DBSCAN）
            dbscan_eps: DBSCAN 半径参数
            dbscan_min_samples: DBSCAN 最小样本数
            weight_lof: LOF 权重
            weight_nnr: NNR 权重
            weight_compactness: 紧凑度权重
        """
        self.knn_k = knn_k
        self.nnr_threshold = nnr_threshold
        self.lof_threshold = lof_threshold
        self.anomaly_score_threshold = anomaly_score_threshold
        self.use_clustering = use_clustering
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.weight_lof = weight_lof
        self.weight_nnr = weight_nnr
        self.weight_compactness = weight_compactness

        # 缓存（scan 后保留，供后续增量检测使用）
        self._cached_vectors: List[List[float]] = []
        self._cached_tree: Optional["KDTree"] = None

    # ---- 公有接口 ----

    def scan(self, all_docs: List[DocumentEmbedding]) -> List[DetectionResult]:
        """对全量文档执行触发词区域检测。

        这是 SP4 的主要入口，建议作为后台周期任务调用。

        Args:
            all_docs: 向量数据库中的全量文档嵌入

        Returns:
            每个文档一个 DetectionResult
        """
        N = len(all_docs)
        if N == 0:
            return []
        if N < self.knn_k + 1:
            self.knn_k = max(2, N // 2)

        vectors = [doc.vector for doc in all_docs]
        self._cached_vectors = vectors

        # ---- 阶段 1: 预计算 kNN 距离（共享计算）----
        # 使用 KD-Tree 加速批量邻居查询
        self._build_tree(vectors)
        all_distances, all_indices = self._batch_knn(vectors)

        # ---- 阶段 2: 计算 LOF ----
        lof_scores = self._compute_batch_lof(all_distances, all_indices)

        # ---- 阶段 3: 计算 NNR ----
        nnr_scores = self._compute_batch_nnr(all_distances)

        # ---- 阶段 4: 聚类分析 ----
        cluster_labels = None
        cluster_compactness_map = None
        if self.use_clustering and N >= self.dbscan_min_samples:
            cluster_labels = self._dbscan_cluster(vectors)
            cluster_compactness_map = self._compute_cluster_compactness(
                vectors, cluster_labels
            )

        # ---- 阶段 5: 综合决策 ----
        results = []
        for i in range(N):
            is_anomaly, anomaly_score, reasons, extra = self._fuse_scores(
                doc_id=all_docs[i].doc_id,
                lof=lof_scores[i],
                nnr=nnr_scores[i],
                cluster_label=(
                    cluster_labels[i] if cluster_labels is not None else None
                ),
                cluster_compactness_map=cluster_compactness_map,
            )

            details = {
                "lof_score": round(lof_scores[i], 4),
                "nnr_score": round(nnr_scores[i], 4),
                "anomaly_score": round(anomaly_score, 4),
                "threshold": self.anomaly_score_threshold,
            }
            if cluster_labels is not None:
                details["cluster_label"] = int(cluster_labels[i])
            if extra:
                details.update(extra)

            results.append(DetectionResult(
                doc_id=all_docs[i].doc_id,
                is_anomaly=is_anomaly,
                anomaly_score=anomaly_score,
                reason=" | ".join(reasons) if reasons else "normal",
                details=details,
            ))

        return results

    def detect_single(
        self,
        doc_embedding: List[float],
        reference_embeddings: List[List[float]],
    ) -> DetectionResult:
        """对单个文档执行检测（基于已缓存的参考集）。

        适用于在线增量场景（新文档入库前检查）。

        Args:
            doc_embedding: 待检测文档嵌入
            reference_embeddings: 参考文档嵌入集

        Returns:
            单个 DetectionResult
        """
        N = len(reference_embeddings)
        if N < 2:
            return DetectionResult(
                is_anomaly=False,
                reason="参考集过小",
            )

        # 临时构建 kNN
        all_vectors = reference_embeddings + [doc_embedding]
        query_idx = len(all_vectors) - 1

        # 计算到所有参考点的距离
        distances = [
            euclidean_distance(doc_embedding, ref)
            for ref in reference_embeddings
        ]
        sorted_indices = sorted(
            range(len(distances)), key=lambda i: distances[i]
        )

        # LOF
        neighbor_indices = sorted_indices[:self.knn_k]
        neighbor_dists = [distances[i] for i in neighbor_indices]
        lrd_target = 1.0 / (mean(neighbor_dists) + EPSILON)

        lrd_neighbors = []
        for ni in neighbor_indices:
            n_dists = sorted([
                euclidean_distance(reference_embeddings[ni], ref)
                for ref in reference_embeddings
            ])[:self.knn_k]
            lrd_n = 1.0 / (mean(n_dists) + EPSILON)
            lrd_neighbors.append(lrd_n)

        lof_score = mean(lrd_neighbors) / (lrd_target + EPSILON)

        # NNR
        avg_knn = mean([distances[i] for i in sorted_indices[:self.knn_k]])
        global_avg_knn = mean([
            mean(sorted([
                euclidean_distance(reference_embeddings[j], ref)
                for ref in reference_embeddings if ref != reference_embeddings[j]
            ])[:self.knn_k])
            for j in range(min(100, N))  # 采样估计
        ]) if N > 1 else 1.0
        nnr_score = avg_knn / (global_avg_knn + EPSILON)

        # 融合
        is_anomaly, anomaly_score, reasons, _ = self._fuse_scores(
            doc_id="",
            lof=lof_score,
            nnr=nnr_score,
            cluster_label=None,
            cluster_compactness_map=None,
        )

        return DetectionResult(
            doc_id="",
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,
            reason=" | ".join(reasons) if reasons else "normal",
            details={
                "lof_score": round(lof_score, 4),
                "nnr_score": round(nnr_score, 4),
                "anomaly_score": round(anomaly_score, 4),
            },
        )

    # ---- 内部: kNN 批量计算 ----

    def _build_tree(self, vectors: List[List[float]]) -> None:
        """构建 KD-Tree（简化版：直接存储向量，线性搜索）。

        生产环境可替换为 sklearn.neighbors.KDTree 或 faiss.IndexFlatIP。
        """
        # 简单实现: 直接存储向量
        # 对 N < 10000 的场景，线性搜索足够快
        self._cached_tree = None  # 预留接口
        self._cached_vectors = vectors

    def _batch_knn(
        self, vectors: List[List[float]]
    ) -> Tuple[List[List[float]], List[List[int]]]:
        """批量计算所有点的 k 近邻距离和索引。

        Returns:
            (distances, indices)
            distances[i][j]: 点 i 到第 j 个最近邻的距离
            indices[i][j]:   点 i 的第 j 个最近邻的索引
        """
        N = len(vectors)
        k = min(self.knn_k + 1, N)

        all_distances = []
        all_indices = []

        for i in range(N):
            dists = [
                euclidean_distance(vectors[i], vectors[j])
                for j in range(N)
            ]
            # -1 表示自身
            idx_sorted = sorted(
                range(N), key=lambda j: dists[j]
            )[:k]
            all_distances.append([dists[j] for j in idx_sorted])
            all_indices.append(idx_sorted)

        return all_distances, all_indices

    # ---- 内部: LOF ----

    def _compute_batch_lof(
        self,
        all_distances: List[List[float]],
        all_indices: List[List[int]],
    ) -> List[float]:
        """批量计算 LOF。"""
        N = len(all_distances)
        k = min(self.knn_k, N - 1)

        # 先算所有点的 lrd
        lrd_values = []
        for i in range(N):
            k_dist = all_distances[i][1:k + 1]  # 排除自身
            lrd_i = 1.0 / (mean(k_dist) + EPSILON)
            lrd_values.append(lrd_i)

        # 再算 LOF
        lof_scores = []
        for i in range(N):
            neighbor_lrd = [
                lrd_values[idx]
                for idx in all_indices[i][1:k + 1]
            ]
            lof_i = mean(neighbor_lrd) / (lrd_values[i] + EPSILON)
            lof_scores.append(lof_i)

        return lof_scores

    # ---- 内部: NNR ----

    def _compute_batch_nnr(
        self,
        all_distances: List[List[float]],
    ) -> List[float]:
        """批量计算 NNR（最近邻距离异常比）。

        NNR_k(d) = avg_knn_dist(d) / global_avg_knn_dist
        """
        N = len(all_distances)
        k = min(self.knn_k, N - 1)

        avg_knn_dists = [mean(all_distances[i][1:k + 1]) for i in range(N)]
        global_avg = mean(avg_knn_dists)

        return [d / (global_avg + EPSILON) for d in avg_knn_dists]

    # ---- 内部: 聚类分析 ----

    def _dbscan_cluster(
        self, vectors: List[List[float]]
    ) -> List[int]:
        """简化 DBSCAN 聚类。

        基于距离矩阵的 DBSCAN 实现。
        生产环境可替换为 sklearn.cluster.DBSCAN。

        Returns:
            每个点的聚类标签（-1 表示噪声点）
        """
        N = len(vectors)
        labels = [-1] * N
        cluster_id = 0

        for i in range(N):
            if labels[i] != -1:
                continue

            # 找到 eps 半径内的邻居
            neighbors = []
            for j in range(N):
                if i == j:
                    continue
                dist = euclidean_distance(vectors[i], vectors[j])
                if dist <= self.dbscan_eps:
                    neighbors.append(j)

            if len(neighbors) < self.dbscan_min_samples:
                labels[i] = -1  # 噪声点
                continue

            # 新聚类
            labels[i] = cluster_id
            seed_set = list(neighbors)

            while seed_set:
                j = seed_set.pop()
                if labels[j] == -1:
                    labels[j] = cluster_id  # 噪声点变成边界点
                elif labels[j] != -1:
                    continue  # 已分配

                # 扩展
                j_neighbors = []
                for k in range(N):
                    if k == j:
                        continue
                    dist = euclidean_distance(vectors[j], vectors[k])
                    if dist <= self.dbscan_eps:
                        j_neighbors.append(k)

                if len(j_neighbors) >= self.dbscan_min_samples:
                    for n in j_neighbors:
                        if labels[n] == -1 or labels[n] == -1:
                            if n not in seed_set:
                                seed_set.append(n)

            cluster_id += 1

        return labels

    def _compute_cluster_compactness(
        self,
        vectors: List[List[float]],
        labels: List[int],
    ) -> Dict[int, float]:
        """计算每个聚类的紧凑度。

        Compactness(C) = mean_{i,j∈C, i<j} ||e_i - e_j||₂

        Returns:
            {cluster_label: compactness_score}
        """
        unique_labels = set(labels) - {-1}
        compactness_map = {}

        for label in unique_labels:
            indices = [i for i, lbl in enumerate(labels) if lbl == label]
            if len(indices) < 2:
                continue

            sum_dist = 0.0
            count = 0
            for a_idx in range(len(indices)):
                for b_idx in range(a_idx + 1, len(indices)):
                    sum_dist += euclidean_distance(
                        vectors[indices[a_idx]], vectors[indices[b_idx]]
                    )
                    count += 1

            compactness_map[label] = sum_dist / count if count > 0 else 0.0

        return compactness_map

    # ---- 内部: 融合 ----

    def _fuse_scores(
        self,
        doc_id: str,
        lof: float,
        nnr: float,
        cluster_label: Optional[int],
        cluster_compactness_map: Optional[Dict[int, float]],
    ) -> Tuple[bool, float, List[str], Dict[str, Any]]:
        """融合多指标得分，生成最终判定。

        Score_AP = w_lof · |log(LOF)| + w_nnr · max(0, 1 - NNR/τ) + w_comp · compactness_signal

        Returns:
            (is_anomaly, anomaly_score, reasons, extra_details)
        """
        reasons = []
        extra = {}

        # LOF 信号
        lof_signal = abs(log2(lof))
        lof_thresh_log = abs(log2(self.lof_threshold))

        # NNR 信号
        nnr_signal = max(0.0, 1.0 - nnr / (self.nnr_threshold + EPSILON))

        # 紧凑度信号
        compactness_signal = 0.0
        if cluster_label is not None and cluster_label >= 0 \
                and cluster_compactness_map is not None:
            label = int(cluster_label)
            if label in cluster_compactness_map:
                comp = cluster_compactness_map[label]
                # 紧凑度越低（值越小）越可疑
                # 将所有紧凑度归一化后取反
                all_comp = list(cluster_compactness_map.values())
                max_comp = max(all_comp) if all_comp else 1.0
                min_comp = min(all_comp) if all_comp else 0.0
                comp_range = max_comp - min_comp if max_comp > min_comp else 1.0
                # 归一化: 0=最紧凑(最可疑), 1=最松散(最正常)
                comp_norm = (comp - min_comp) / comp_range
                # 取反: 1=最可疑, 0=最正常
                compactness_signal = 1.0 - comp_norm
                extra["compactness_raw"] = round(comp, 4)
                extra["compactness_norm"] = round(comp_norm, 4)

        # 融合
        anomaly_score = (
            self.weight_lof * min(lof_signal, 5.0) +          # |log2(LOF)| 上限 5
            self.weight_nnr * nnr_signal +
            self.weight_compactness * compactness_signal
        )

        # 判定: 任一强信号 或 融合得分超过阈值
        strong_lof = lof_signal > lof_thresh_log
        strong_nnr = nnr < self.nnr_threshold
        strong_compact = compactness_signal > 0.8

        is_anomaly = (strong_lof or strong_nnr or strong_compact) and anomaly_score >= 0.3

        if lof_signal > lof_thresh_log:
            reasons.append(f"LOF异常(lof={lof:.3f})")
        if nnr < self.nnr_threshold:
            reasons.append(f"NNR异常(nnr={nnr:.3f})")
        if compactness_signal > 0.8:
            reasons.append(f"紧凑度异常(compact={compactness_signal:.3f})")

        return is_anomaly, anomaly_score, reasons, extra
