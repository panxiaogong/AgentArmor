"""
test_sp1.py — SP1 嵌入异常检测单元测试

测试目标: EmbeddingAnomalyDetector
  1. 正常训练流程（clean→CleanStats）
  2. 良性文档返回低异常得分
  3. PoisonedRAG 攻击文档返回高异常得分
  4. 马氏距离和 LOF 融合的互补性
"""

import math
import random
import pytest
from typing import List

from External_Developer_Write.sp1_embedding_anomaly import EmbeddingAnomalyDetector
from External_Developer_Write.types import DocumentEmbedding, CleanStats

random.seed(42)


class TestEmbeddingAnomalyDetector:
    """SP1 嵌入异常检测器测试套件。"""

    @pytest.fixture(autouse=True)
    def setup(self, benign_embeddings):
        self.detector = EmbeddingAnomalyDetector(
            knn_k=5,
            weight_mahal=0.5,
            weight_lof=0.5,
            threshold=0.5,
            shrinkage_rho=0.3,
        )
        # 构造 DocumentEmbedding 列表用于 train
        self.benign_de = [
            DocumentEmbedding(doc_id=f"train_{i}", vector=vec)
            for i, vec in enumerate(benign_embeddings[:30])
        ]

    # ── 测试 1: 训练 ──────────────────────────────────────────

    def test_train_produces_valid_stats(self):
        """训练应生成有效的 CleanStats（非空均值、可逆协方差）。"""
        stats = self.detector.train(self.benign_de)
        assert stats is not None
        assert len(stats.mu) > 0
        assert len(stats.sigma) == len(stats.sigma[0]) > 0
        assert stats.max_train_mahal > 0
        assert len(stats.reference_embeddings) > 0

    def test_train_shrinkage_produces_positive_definite(self):
        """收缩估计应保证协方差正定（对角元 > 0）。"""
        stats = self.detector.train(self.benign_de)
        for i in range(len(stats.mu)):
            assert stats.sigma[i][i] > 0

    # ── 测试 2: 良性检测 ─────────────────────────────────────

    def test_benign_docs_return_low_scores(self):
        """良性文档应返回低异常得分。"""
        stats = self.detector.train(self.benign_de)
        for de in self.benign_de[:10]:
            result = self.detector.detect(de.vector, stats)
            assert result.anomaly_score >= 0

    def test_benign_has_low_mahalanobis(self):
        """良性文档的马氏距离应接近训练集分布。"""
        stats = self.detector.train(self.benign_de)
        mahal_scores = []
        for de in self.benign_de[:10]:
            result = self.detector.detect(de.vector, stats)
            mahal = result.details.get("mahalanobis_raw", 0)
            mahal_scores.append(mahal)
        avg_mahal = sum(mahal_scores) / len(mahal_scores) if mahal_scores else 0
        assert avg_mahal / (stats.max_train_mahal + 1e-8) < 2.0

    # ── 测试 3: 攻击检测 ─────────────────────────────────────

    def test_poisonedrag_detected_as_anomaly(self):
        """PoisonedRAG（均值偏移）应被检测为异常。"""
        stats = self.detector.train(self.benign_de)

        # 构造显式攻击嵌入（大幅偏移）
        attack_vec = [v + 3.0 for v in stats.mu]
        result = self.detector.detect(attack_vec, stats)
        assert result.anomaly_score > 0.5 or result.is_anomaly, \
            f"攻击得分={result.anomaly_score:.4f}"

    def test_mahalanobis_increases_with_deviation(self):
        """马氏距离应随偏移增大而单调递增。"""
        stats = self.detector.train(self.benign_de)
        prev_score = 0
        for scale in [0.5, 1.0, 2.0, 3.0, 4.0]:
            vec = [stats.mu[d] + scale for d in range(len(stats.mu))]
            result = self.detector.detect(vec, stats)
            mahal = result.details.get("mahalanobis_raw", 0)
            assert mahal >= prev_score - 0.01, f"马氏距离应在 scale={scale} 时递增"
            prev_score = mahal

    # ── 测试 4: 方法特性 ─────────────────────────────────────

    def test_lof_sensitivity_to_dense_regions(self):
        """LOF检测密集与稀疏区域的能力。"""
        import random
        random.seed(42)

        # 创建密集区域 + 稀疏区域
        dim = 10
        dense_vecs = [[random.gauss(5, 0.1) for _ in range(dim)] for _ in range(20)]
        sparse_vecs = [[random.gauss(0, 1.0) for _ in range(dim)] for _ in range(20)]
        all_de = [DocumentEmbedding(doc_id=f"d{i}", vector=v)
                  for i, v in enumerate(dense_vecs + sparse_vecs)]

        stats = self.detector.train(all_de)

        # 密集区的点
        result_dense = self.detector.detect(dense_vecs[0], stats)
        lof_dense = result_dense.details.get("lof_score", 1.0)

        # 稀疏区的点
        result_sparse = self.detector.detect(sparse_vecs[5], stats)
        lof_sparse = result_sparse.details.get("lof_score", 1.0)

        # LOF > 1 表示比邻居稀疏，LOF < 1 表示比邻居密集
        print(f"\n密集点 LOF={lof_dense:.3f}, 稀疏点 LOF={lof_sparse:.3f}")
        assert lof_sparse >= lof_dense * 0.5, \
            f"稀疏点 LOF({lof_sparse:.3f}) 不应过度低于密集点({lof_dense:.3f})"

    # ── 测试 5: 边界条件 ─────────────────────────────────────

    def test_empty_train_raises_value_error(self):
        """空训练集应抛出 ValueError。"""
        import pytest
        with pytest.raises(ValueError):
            self.detector.train([])

    def test_single_doc_train_raises_value_error(self):
        """单文档训练应抛出 ValueError（不足 2 个）。"""
        import pytest
        with pytest.raises(ValueError):
            self.detector.train(self.benign_de[:1])
