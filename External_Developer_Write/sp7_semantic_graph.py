"""
sp7_semantic_graph.py — 语义依赖图分析（SP7）

对应攻击: 节点2 — 跨文档语义混淆注入（跨文档碎片化）
对应分析报告: 5.3.2 节（方法3: 语义依赖图）
核心方法: EIRE 语义三元组 → 跨文档依赖图 → 图异常检测

攻击者将恶意指令碎片化分布在多个文档中，每个文档单独无害，
但组合后形成完整攻击。SP7 通过构建文档间的语义依赖图来检测
这种"人为拼接"的特征——碎片化攻击产生的跨文档边密度显著低于正常。

数学基础:
    1. 语义三元组: (E, I, R) — 实体, 意图, 关系
    2. 图边密度: ρ(G) = 2|E| / (|V|(|V|-1))
    3. 跨文档边比例: cross_doc_edges / total_edges

设计决策:
    - 假设同一批次的文档组应为语义相关的自然集合
    - 攻击文档的实体集通常偏小（只关注攻击目标）
    - 攻击文档的内部边密度接近 0（无自然语义连接）
"""

from typing import List, Optional, Set, Dict, Any, Tuple, Callable
from collections import defaultdict

from .types import Document, SemanticTriple, DetectionResult
from .utils import EPSILON, cosine_similarity, mean, simple_tokenize


class SemanticDependencyGraphAnalyzer:
    """语义依赖图分析器。

    通过构建文档组内实体级别的语义依赖图，检测跨文档碎片化攻击。
    建议在文档批量入库后运行（批次级别检测）。

    使用方式:
        analyzer = SemanticDependencyGraphAnalyzer(nlp_extractor)
        result = analyzer.analyze(doc_group)
    """

    def __init__(
        self,
        nlp_extractor: Optional[Callable] = None,
        density_threshold: float = 0.25,
        anomaly_edge_ratio: float = 0.4,
        extract_top_k: int = 20,
        min_entity_count: int = 3,
    ):
        """初始化分析器。

        Args:
            nlp_extractor: NLP 语义三元组提取器。
                           签名: nlp_extractor(text, top_k) -> List[SemanticTriple]
                           为 None 时使用简单的抽取式替代。
            density_threshold: 图密度异常阈值（低于此值标记可疑）
            anomaly_edge_ratio: 异常文档的跨文档边比例阈值
            extract_top_k: 每文档提取的最大三元组数
            min_entity_count: 最小实体数（少于该数的文档跳过）
        """
        self.nlp_extractor = nlp_extractor
        self.density_threshold = density_threshold
        self.anomaly_edge_ratio = anomaly_edge_ratio
        self.extract_top_k = extract_top_k
        self.min_entity_count = min_entity_count

    # ---- 公有接口 ----

    def analyze(self, doc_group: List[Document]) -> DetectionResult:
        """对文档组执行语义依赖图分析。

        Args:
            doc_group: 同一批次/来源的文档列表（建议 >= 3 篇）

        Returns:
            组级别的 DetectionResult
        """
        if len(doc_group) < 2:
            return DetectionResult(
                is_anomaly=False,
                reason=f"文档数 {len(doc_group)} < 2，不适合图分析",
            )

        # ---- 阶段 1: 提取语义三元组 ----
        all_triples: List[Tuple[str, SemanticTriple]] = []
        doc_entity_counts: Dict[str, int] = {}

        for doc in doc_group:
            triples = self._extract_triples(doc)
            for triple in triples:
                all_triples.append((doc.doc_id, triple))
            # 统计该文档的唯一实体数
            entities = set()
            for _, t in triples:
                entities.add(t.head)
                entities.add(t.tail)
            doc_entity_counts[doc.doc_id] = len(entities)

        if not all_triples:
            return DetectionResult(
                is_anomaly=False,
                reason="未提取到语义三元组",
            )

        # ---- 阶段 2: 构建语义依赖图 ----
        entity_list, adj_matrix, edge_doc_map = self._build_graph(
            all_triples
        )

        if len(entity_list) < 3:
            return DetectionResult(
                is_anomaly=False,
                reason=f"实体数 {len(entity_list)} < 3，图过于稀疏",
            )

        # ---- 阶段 3: 图度量计算 ----
        M = len(entity_list)

        # 3a. 整体边密度 ρ(G) = 2|E| / (|V|(|V|-1))
        edge_count = sum(
            1 for i in range(M) for j in range(i + 1, M)
            if adj_matrix[i][j] > 0
        )
        total_possible = M * (M - 1) / 2
        graph_density = edge_count / total_possible if total_possible > 0 else 0.0

        # 3b. 跨文档边统计
        cross_doc_edges = 0
        total_edges_with_docs = len(edge_doc_map)
        for doc_set in edge_doc_map.values():
            if len(doc_set) > 1:  # 多个文档涉及同一实体对
                cross_doc_edges += 1

        cross_doc_ratio = (
            cross_doc_edges / total_edges_with_docs
            if total_edges_with_docs > 0 else 0.0
        )

        # 3c. 单文档子图特性
        doc_metrics = self._compute_doc_metrics(
            doc_group, all_triples, entity_list, adj_matrix
        )

        # ---- 阶段 4: 异常判定 ----
        is_global_sparse = graph_density < self.density_threshold

        # 检测异常文档: 内部密度低 + 跨文档连接异常 + 实体集小
        anomalous_docs = []
        for doc_id, metrics in doc_metrics.items():
            is_suspicious = (
                metrics["internal_density"] < self.density_threshold and
                metrics["cross_edges"] > 0 and
                metrics["entity_count"] <= self.min_entity_count
            )
            if is_suspicious:
                anomalous_docs.append(doc_id)

        is_anomaly = is_global_sparse and len(anomalous_docs) >= 2

        # 构造原因
        reasons = []
        if is_global_sparse:
            reasons.append(
                f"整体图稀疏(密度={graph_density:.3f}<{self.density_threshold})"
            )
        if anomalous_docs:
            reasons.append(
                f"异常文档({len(anomalous_docs)}篇): {', '.join(anomalous_docs[:3])}"
            )
        if cross_doc_ratio > self.anomaly_edge_ratio:
            reasons.append(
                f"跨文档边比例过高({cross_doc_ratio:.2f})"
            )

        return DetectionResult(
            doc_id="|".join(d.doc_id for d in doc_group),
            is_anomaly=is_anomaly,
            anomaly_score=1.0 - graph_density if is_anomaly else graph_density,
            reason=" | ".join(reasons) if reasons else f"正常，图密度={graph_density:.3f}",
            details={
                "graph_density": round(graph_density, 4),
                "cross_doc_ratio": round(cross_doc_ratio, 4),
                "entity_count": M,
                "edge_count": edge_count,
                "total_possible_edges": int(total_possible),
                "anomalous_docs": anomalous_docs,
                "doc_metrics": {
                    doc_id: {
                        "entity_count": m["entity_count"],
                        "internal_density": round(m["internal_density"], 4),
                        "cross_edges": m["cross_edges"],
                    }
                    for doc_id, m in doc_metrics.items()
                },
                "n_docs": len(doc_group),
            },
        )

    # ---- 内部方法 ----

    def _extract_triples(
        self, doc: Document
    ) -> List[Tuple[str, SemanticTriple]]:
        """从文档中提取语义三元组列表。

        使用 nlp_extractor 或默认的简单模式匹配。

        Returns:
            List of (entity_type, triple)
            简化实现: 每个 (head, relation, tail) 三元组
        """
        if self.nlp_extractor is not None:
            try:
                result = self.nlp_extractor(doc.content, self.extract_top_k)
                return [(doc.doc_id, t) for t in result]
            except Exception:
                pass

        # 默认的简单提取: 基于命名实体 + 共现关系
        triples: List[Tuple[str, SemanticTriple]] = []
        tokens = simple_tokenize(doc.content)

        # 简单关键词作为实体候选
        candidates = []
        for word in tokens:
            if (word[0].isupper() if word else False) and len(word) > 2:
                candidates.append(word)

        # 去重并限制数量
        seen = set()
        unique_candidates = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                unique_candidates.append(c)

        candidates = unique_candidates[:self.extract_top_k]

        # 基于共现构建三元组（窗口内共现 = 语义关系）
        window_size = 10
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                # 检查是否在文档中共同出现（同一窗口内）
                text_lower = doc.content.lower()
                c_i_lower = candidates[i].lower()
                c_j_lower = candidates[j].lower()

                if c_i_lower in text_lower and c_j_lower in text_lower:
                    # 检查窗口共现
                    words = text_lower.split()
                    try:
                        pos_i = words.index(c_i_lower) if c_i_lower in words else -1
                        pos_j = words.index(c_j_lower) if c_j_lower in words else -1
                        if (pos_i >= 0 and pos_j >= 0
                                and abs(pos_i - pos_j) < window_size):
                            triple = SemanticTriple(
                                head=candidates[i],
                                relation="co_occur",
                                tail=candidates[j],
                            )
                            triples.append((doc.doc_id, triple))
                    except ValueError:
                        continue

        return triples[:self.extract_top_k]

    def _build_graph(
        self, all_triples: List[Tuple[str, SemanticTriple]]
    ) -> Tuple[List[str], List[List[float]], Dict[Tuple[int, int], Set[str]]]:
        """从语义三元组构建图。

        Returns:
            (entity_list, adj_matrix, edge_doc_map)
            entity_list: 实体名称列表（索引映射）
            adj_matrix:   M×M 邻接矩阵（边权重 = 出现次数）
            edge_doc_map: {(i,j): set_of_doc_ids} 记录每条边涉及的文档
        """
        # 收集所有实体
        entity_set: Set[str] = set()
        for _, triple in all_triples:
            entity_set.add(triple.head)
            entity_set.add(triple.tail)

        entity_list = list(entity_set)
        entity_to_idx = {e: i for i, e in enumerate(entity_list)}
        M = len(entity_list)

        # 邻接矩阵
        adj = [[0.0] * M for _ in range(M)]
        edge_doc_map: Dict[Tuple[int, int], Set[str]] = defaultdict(set)

        for doc_id, triple in all_triples:
            i = entity_to_idx.get(triple.head)
            j = entity_to_idx.get(triple.tail)
            if i is None or j is None:
                continue
            if i == j:
                continue

            # 无向图
            adj[i][j] += 1.0
            adj[j][i] += 1.0

            key = (i, j) if i < j else (j, i)
            edge_doc_map[key].add(doc_id)

        return entity_list, adj, dict(edge_doc_map)

    def _compute_doc_metrics(
        self,
        doc_group: List[Document],
        all_triples: List[Tuple[str, SemanticTriple]],
        entity_list: List[str],
        adj_matrix: List[List[float]],
    ) -> Dict[str, Dict[str, Any]]:
        """计算每个文档的子图特性。

        对每个文档:
        - entity_count: 该文档涉及的实体数
        - internal_density: 该文档实体之间的边密度
        - cross_edges: 该文档实体与其他文档实体的连接数
        """
        entity_to_idx = {e: i for i, e in enumerate(entity_list)}
        doc_metrics: Dict[str, Dict[str, Any]] = {}

        for doc in doc_group:
            doc_entities: Set[str] = set()
            for d_id, triple in all_triples:
                if d_id == doc.doc_id:
                    doc_entities.add(triple.head)
                    doc_entities.add(triple.tail)

            doc_entities_list = list(doc_entities)
            n_entities = len(doc_entities_list)
            if n_entities < 2:
                doc_metrics[doc.doc_id] = {
                    "entity_count": n_entities,
                    "internal_density": 0.0,
                    "cross_edges": 0,
                }
                continue

            # 内部边密度
            internal_edges = 0
            internal_possible = n_entities * (n_entities - 1) / 2
            for a in range(n_entities):
                for b in range(a + 1, n_entities):
                    idx_a = entity_to_idx.get(doc_entities_list[a])
                    idx_b = entity_to_idx.get(doc_entities_list[b])
                    if (idx_a is not None and idx_b is not None
                            and adj_matrix[idx_a][idx_b] > 0):
                        internal_edges += 1

            internal_density = (
                internal_edges / internal_possible
                if internal_possible > 0 else 0.0
            )

            # 跨文档边数（该文档实体与其他文档实体的连接）
            cross_edges = 0
            for ent in doc_entities_list:
                ei = entity_to_idx.get(ent)
                if ei is None:
                    continue
                for j in range(len(entity_list)):
                    if j != ei and adj_matrix[ei][j] > 0:
                        # 检查 j 实体是否属于其他文档
                        entity_j = entity_list[j]
                        for d_id2, triple2 in all_triples:
                            if (d_id2 != doc.doc_id
                                    and (triple2.head == entity_j
                                         or triple2.tail == entity_j)):
                                cross_edges += 1
                                break

            doc_metrics[doc.doc_id] = {
                "entity_count": n_entities,
                "internal_density": round(internal_density, 4),
                "cross_edges": cross_edges,
            }

        return doc_metrics
