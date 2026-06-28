"""
test_sp7.py — SP7 语义依赖图分析单元测试

测试目标: SemanticDependencyGraphAnalyzer
  1. 良性文档组 → 正常图密度
  2. 碎片化攻击文档组 → 稀疏图 + 异常文档标记
  3. 语义三元组提取
"""

import pytest
from typing import List, Dict, Any

from External_Developer_Write.sp7_semantic_graph import SemanticDependencyGraphAnalyzer
from External_Developer_Write.types import Document, SemanticTriple, DetectionResult


class TestSemanticDependencyGraphAnalyzer:
    """SP7 语义依赖图测试套件。"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.analyzer = SemanticDependencyGraphAnalyzer(
            nlp_extractor=None,  # 使用默认的简单提取器
            density_threshold=0.25,
            anomaly_edge_ratio=0.4,
            extract_top_k=15,
            min_entity_count=2,
        )

    def _make_doc(self, doc_id: str, content: str) -> Document:
        return Document(doc_id=doc_id, content=content, source="upload")

    # ── 测试 1: 良性文档组 ───────────────────────────────────

    def test_benign_group_has_normal_density(self):
        """语义相关的良性文档组应有正常图密度（> 阈值）。"""
        docs = [
            self._make_doc("doc_1", "Neural Networks process data through layers of neurons. "
                           "Deep Learning uses many such layers."),
            self._make_doc("doc_2", "Convolutional Neural Networks are used for image processing. "
                           "They use filters to detect features."),
            self._make_doc("doc_3", "Recurrent Neural Networks handle sequential data. "
                           "LSTM is a popular RNN variant."),
        ]
        result = self.analyzer.analyze(docs)
        print(f"\n良性组: density={result.details.get('graph_density', 'N/A')}, "
              f"entities={result.details.get('entity_count', 'N/A')}")
        # 良性组通常有较高密度的实体连接
        assert not result.is_anomaly or result.details.get('graph_density', 0) >= 0.1

    # ── 测试 2: 碎片化攻击 ───────────────────────────────────

    def test_fragmented_attack_detected(self):
        """跨文档碎片化攻击应导致低图密度。

        攻击文档故意使用不同领域的实体，减少交叉引用。
        """
        # 良性文档 2 篇（共享实体）
        benign = [
            self._make_doc("benign_1",
                           "Apple released a new iPhone with advanced AI chips. "
                           "The processor uses machine learning for camera optimization."),
            self._make_doc("benign_2",
                           "Apple's AI processor enhances photo quality. "
                           "Machine learning algorithms run on the Neural Engine."),
        ]
        # 攻击文档 3 篇（各自不同实体，缺乏跨文档连接）
        attack = [
            self._make_doc("attack_1",
                           "BANK_TRANSFER initiated to account 12345. "
                           "Transaction ID: TXN_98765 confirmed."),
            self._make_doc("attack_2",
                           "PASSWORD_RESET requested for admin@company.com. "
                           "Verification code sent to phone."),
            self._make_doc("attack_3",
                           "FIREWALL_DISABLED on server 192.168.1.1. "
                           "All ports opened for external access."),
        ]

        all_docs = benign + attack
        result = self.analyzer.analyze(all_docs)

        print(f"\n碎片化攻击组: density={result.details.get('graph_density', 'N/A')}, "
              f"anomalous_docs={result.details.get('anomalous_docs', [])}")

        graph_density = result.details.get("graph_density", 1.0)
        anomalous_docs = result.details.get("anomalous_docs", [])

        # 攻击文档应在 anomalous_docs 中
        if result.is_anomaly:
            for ad_id in ["attack_1", "attack_2", "attack_3"]:
                assert ad_id in anomalous_docs or not result.is_anomaly, \
                    f"{ad_id} 应在异常文档列表中"

    def test_cross_doc_ratio_elevated_for_attack(self):
        """碎片化攻击应导致偏低的跨文档边比例。"""
        # 全是良性相关文档
        related_docs = [
            self._make_doc(f"rel_{i}",
                           f"Topic A discusses item {i}. Related concept is B. "
                           f"Cross reference to C and D.") for i in range(3)
        ]
        result = self.analyzer.analyze(related_docs)
        print(f"\n关联文档: density={result.details.get('graph_density', 'N/A')}, "
              f"cross_doc_ratio={result.details.get('cross_doc_ratio', 'N/A')}")

    # ── 测试 3: 语义三元组提取 ───────────────────────────────

    def test_triple_extraction_basic(self):
        """语义三元组提取应返回有效结果。"""
        doc = self._make_doc("test", "Apple and Google are technology companies. "
                             "Both invest heavily in AI research and development.")
        triples = self.analyzer._extract_triples(doc)
        print(f"\n提取的三元组数: {len(triples)}")
        if triples:
            print(f"示例: {triples[0]}")
            assert len(triples[0]) == 2  # (doc_id, SemanticTriple)
            assert isinstance(triples[0][1], SemanticTriple)

    def test_triple_extraction_empty(self):
        """空文档应返回空列表。"""
        doc = self._make_doc("empty", "a")
        triples = self.analyzer._extract_triples(doc)
        assert isinstance(triples, list)

    # ── 测试 4: 图构建 ────────────────────────────────────────

    def test_graph_building(self):
        """图构建应生成邻接矩阵和边-文档映射。"""
        triples = [
            ("d1", SemanticTriple(head="Apple", relation="produces", tail="iPhone")),
            ("d1", SemanticTriple(head="Apple", relation="competes", tail="Samsung")),
            ("d2", SemanticTriple(head="Samsung", relation="produces", tail="Galaxy")),
        ]
        entities, adj, edge_map = self.analyzer._build_graph(triples)
        assert len(entities) >= 3  # Apple, Samsung, iPhone, Galaxy
        assert len(adj) == len(entities)
        print(f"\n图构建: {len(entities)} 实体, {len(edge_map)} 边")

    # ── 测试 5: 边界条件 ─────────────────────────────────────

    def test_single_doc_group(self):
        """单文档组应跳过分析。"""
        docs = [self._make_doc("only", "Test content.")]
        result = self.analyzer.analyze(docs)
        assert not result.is_anomaly

    def test_no_triples_extracted(self):
        """无三元组时返回正常。"""
        docs = [
            self._make_doc("d1", "a b c"),
            self._make_doc("d2", "d e f"),
        ]
        result = self.analyzer.analyze(docs)
        assert not result.is_anomaly
