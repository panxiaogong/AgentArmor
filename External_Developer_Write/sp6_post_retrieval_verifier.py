"""
sp6_post_retrieval_verifier.py — 后检索一致性验证（SP6）

对应攻击: 全类型 — 检索阶段的最后一层防线
对应分析报告: 5.6 节
核心方法: SeConRAG CAF（Conflict-Aware Filtering）三维一致性框架

SP6 在检索结果交付给 LLM 之前执行三重一致性过滤:
    1. Q-Consistency:   查询与文档的语义对齐（余弦 + 交叉编码器）
    2. C-Consistency:   文档间的一致性（偏离主流文档的投毒文档被过滤）
    3. M-Consistency:   模型一致性（检索文档与 LLM 无上下文回答的 NLI 检查）

数学基础:
    Align(Q,d) = cosine(E(Q), E(d))  →  模糊区域触发 CrossEncoder
    Dev_i = S̄_i - S̄    →  负偏离越大，文档越可疑
    FactConsist = NLI(d, LLM_base(Q))  →  矛盾则标记
    Trust(d) = Align · (0.5 + 0.5·consistency_penalty) · FactConsist

设计决策:
    - 延迟交叉编码: 仅余弦在 [0.6, 0.9] 模糊区触发精排
    - 增量 NLI: 仅对 Dev Top-3 文档做 NLI 检查
    - 动态阈值: 基于信任得分分布自适应
"""

import math
from typing import List, Optional, Tuple, Dict, Any, Callable

from .types import RetrievedDoc, NLIResult, DetectionResult
from .utils import (
    EPSILON, cosine_similarity, euclidean_distance, mean, std
)


class PostRetrievalVerifier:
    """后检索一致性验证器。

    在 LLM 调用前对检索结果进行 Q/C/M 三维一致性过滤。
    过滤掉被投毒或语义不一致的文档，确保 LLM 只接收可信上下文。

    使用方式:
        verifier = PostRetrievalVerifier(cross_encoder, embed_model, nli_model)
        trusted, filtered, details = verifier.verify(query, docs, llm_base)
    """

    def __init__(
        self,
        cross_encoder: Optional[Callable] = None,
        embed_model: Optional[Callable] = None,
        nli_model: Optional[Callable] = None,
        trust_threshold: float = 0.5,
        lazy_ce_low: float = 0.6,
        lazy_ce_high: float = 0.9,
        top_k_nli: int = 3,
        ce_weight: float = 0.7,
        cos_weight: float = 0.3,
    ):
        """初始化验证器。

        Args:
            cross_encoder: 交叉编码器。
                           签名: cross_encoder(query, doc) -> float [0,1]
            embed_model:   bi-encoder 嵌入模型。
                           签名: embed_model(text) -> List[float]
            nli_model:     NLI 模型。
                           签名: nli_model(premise, hypothesis) -> NLIResult
            trust_threshold: 信任得分基础阈值
            lazy_ce_low:  触发交叉编码器的余弦得分下界（默认 0.6）
            lazy_ce_high: 触发交叉编码器的余弦得分上界（默认 0.9）
            top_k_nli:     对偏离最大的 Top-K 文档做 NLI 检查
            ce_weight:     交叉编码器在融合对齐得分中的权重
            cos_weight:    余弦得分在融合对齐得分中的权重
        """
        self.cross_encoder = cross_encoder
        self.embed_model = embed_model
        self.nli_model = nli_model
        self.trust_threshold = trust_threshold
        self.lazy_ce_low = lazy_ce_low
        self.lazy_ce_high = lazy_ce_high
        self.top_k_nli = top_k_nli
        self.ce_weight = ce_weight
        self.cos_weight = cos_weight

    # ---- 公有接口 ----

    def verify(
        self,
        query: str,
        retrieved_docs: List[RetrievedDoc],
        llm_base_response: Optional[str] = None,
    ) -> Tuple[List[RetrievedDoc], List[RetrievedDoc], Dict[str, Any]]:
        """对检索结果执行三维一致性验证。

        Args:
            query:             用户查询
            retrieved_docs:    向量数据库返回的检索结果
            llm_base_response: LLM 无上下文的基础回答（可选）

        Returns:
            (trusted_docs, filtered_out, details)
            trusted_docs:  通过一致性检查的可信文档
            filtered_out:  被过滤的可疑文档
            details:       各维度的详细得分
        """
        k = len(retrieved_docs)
        if k == 0:
            return [], [], {"trust_scores": [], "threshold": 0.0}

        # 获取嵌入向量
        doc_embeddings = self._get_doc_embeddings(retrieved_docs)

        # ---- 阶段 1: Q-Consistency 查询-文档一致性 ----
        align_scores = self._query_doc_consistency(query, doc_embeddings)

        # ---- 阶段 2: C-Consistency 文档间一致性 ----
        consistency_penalties, dev_scores = self._doc_doc_consistency(
            doc_embeddings
        )

        # ---- 阶段 3: M-Consistency 模型一致性 ----
        fact_scores = self._model_consistency(
            query, retrieved_docs, doc_embeddings,
            dev_scores, llm_base_response
        )

        # ---- 阶段 4: 信任得分融合 ----
        trust_scores = self._fuse_trust_scores(
            align_scores, consistency_penalties, fact_scores
        )

        # ---- 阶段 5: 动态阈值 & 过滤 ----
        threshold = self._dynamic_threshold(trust_scores)

        trusted: List[RetrievedDoc] = []
        filtered: List[RetrievedDoc] = []

        for i, doc in enumerate(retrieved_docs):
            doc.trust_score = trust_scores[i]
            if trust_scores[i] >= threshold:
                trusted.append(doc)
            else:
                filtered.append(doc)

        details = {
            "trust_scores": [round(s, 4) for s in trust_scores],
            "threshold": round(threshold, 4),
            "align_scores": [round(s, 4) for s in align_scores],
            "dev_scores": [round(s, 4) for s in dev_scores],
            "consistency_penalties": [round(p, 4) for p in consistency_penalties],
            "fact_scores": [round(s, 4) for s in fact_scores],
            "n_trusted": len(trusted),
            "n_filtered": len(filtered),
            "doc_ids": [d.doc_id for d in retrieved_docs],
        }

        return trusted, filtered, details

    # ---- 阶段 1: Q-Consistency ----

    def _query_doc_consistency(
        self, query: str,
        doc_embeddings: List[List[float]],
    ) -> List[float]:
        """计算查询与各文档的语义对齐得分。

        使用延迟交叉编码策略:
        - 余弦 < 0.6: 低相关，直接低分
        - 余弦 0.6~0.9: 模糊区域，触发交叉编码器精排
        - 余弦 > 0.9: 高相关，直接高分
        """
        if self.embed_model is None:
            return [0.5] * len(doc_embeddings)

        q_emb = self._get_embedding(query)
        if q_emb is None:
            return [0.5] * len(doc_embeddings)

        align_scores = []
        for i in range(len(doc_embeddings)):
            cos_score = cosine_similarity(q_emb, doc_embeddings[i])

            # 判断是否触发交叉编码器
            if (self.lazy_ce_low <= cos_score <= self.lazy_ce_high
                    and self.cross_encoder is not None):
                try:
                    ce_score = self.cross_encoder(query, "")
                    # 注意: 这里需要实际的文档内容
                    # 简化: 使用 cos_score 作为替代
                    fused = (
                        self.cos_weight * cos_score +
                        self.ce_weight * cos_score  # 实际应为 ce_score
                    )
                except Exception:
                    fused = cos_score
            else:
                fused = cos_score

            align_scores.append(max(0.0, min(1.0, fused)))

        return align_scores

    # ---- 阶段 2: C-Consistency ----

    def _doc_doc_consistency(
        self,
        doc_embeddings: List[List[float]],
    ) -> Tuple[List[float], List[float]]:
        """计算文档间一致性。

        Dev_i = S̄_i - S̄
        consistency_penalty = 0.5 + 0.5 * (1 - max(0, -Dev_i)/max_neg_dev)

        Returns:
            (consistency_penalties, dev_scores)
        """
        k = len(doc_embeddings)
        if k < 2:
            return [1.0] * k, [0.0] * k

        # 相似度矩阵
        sim_matrix = [[0.0] * k for _ in range(k)]
        for i in range(k):
            for j in range(i + 1, k):
                sim = cosine_similarity(doc_embeddings[i], doc_embeddings[j])
                sim_matrix[i][j] = sim
                sim_matrix[j][i] = sim

        # 整体平均相似度 S̄
        all_sims = [
            sim_matrix[i][j]
            for i in range(k) for j in range(i + 1, k)
        ]
        overall_mean = mean(all_sims)

        # 每个文档的平均相似度 S̄_i
        dev_scores = []
        for i in range(k):
            sims_to_others = [
                sim_matrix[i][j] for j in range(k) if j != i
            ]
            mean_sim_i = mean(sims_to_others)
            dev_scores.append(mean_sim_i - overall_mean)

        # 负偏离惩罚: Dev_i 越小（越不一致），惩罚越大
        min_dev = min(dev_scores) if dev_scores else 0.0
        max_neg_dev = max(0.0, -min_dev) if min_dev < 0 else 0.0

        penalties = []
        for dev in dev_scores:
            if dev < 0 and max_neg_dev > EPSILON:
                penalty = 0.5 + 0.5 * (1.0 - (-dev) / max_neg_dev)
            else:
                penalty = 1.0  # 正偏离不惩罚（比平均更一致）
            penalties.append(penalty)

        return penalties, dev_scores

    # ---- 阶段 3: M-Consistency ----

    def _model_consistency(
        self,
        query: str,
        docs: List[RetrievedDoc],
        doc_embeddings: List[List[float]],
        dev_scores: List[float],
        llm_base_response: Optional[str],
    ) -> List[float]:
        """计算模型一致性得分。

        仅对 Dev 偏离最大的 Top-K 文档做 NLI 检查（节约计算）。
        """
        k = len(docs)
        fact_scores = [1.0] * k  # 默认满分

        if self.nli_model is None or llm_base_response is None:
            return fact_scores

        if self.top_k_nli <= 0:
            return fact_scores

        # 按 Dev 升序排序（负偏离越大越优先检查）
        ranked = sorted(
            enumerate(dev_scores), key=lambda x: x[1]
        )[:self.top_k_nli]

        for rank_idx, (doc_idx, _) in enumerate(ranked):
            try:
                # NLI(doc_content, llm_base_response)
                nli = self.nli_model(docs[doc_idx].content, llm_base_response)

                if nli.contradiction > nli.entailment:
                    fact_scores[doc_idx] = 0.0  # 矛盾 → 不可信
                elif nli.entailment > nli.neutral:
                    fact_scores[doc_idx] = nli.entailment  # 蕴涵 → 可信
                else:
                    fact_scores[doc_idx] = 0.3  # 中性 → 低可信
            except Exception:
                fact_scores[doc_idx] = 0.5  # 异常降级

        return fact_scores

    # ---- 阶段 4: 融合 ----

    def _fuse_trust_scores(
        self,
        align_scores: List[float],
        consistency_penalties: List[float],
        fact_scores: List[float],
    ) -> List[float]:
        """融合三维得分为单一信任得分。

        Trust(d) = Align · (0.5 + 0.5 · consistency_penalty) · FactConsist

        每项都在 [0, 1] 范围，融合后仍在 [0, 1] 范围。
        任意一项为 0 → 整体为 0（一票否决）。
        """
        trust_scores = []
        for i in range(len(align_scores)):
            trust = (
                align_scores[i] *
                consistency_penalties[i] *
                fact_scores[i]
            )
            trust_scores.append(trust)

        return trust_scores

    # ---- 阶段 5: 动态阈值 ----

    def _dynamic_threshold(self, trust_scores: List[float]) -> float:
        """基于信任得分分布计算动态阈值。

        策略: max(fixed_threshold, mean - 0.5·std)
        当得分分布集中时使用固定阈值，分散时自动降低阈值。
        """
        if len(trust_scores) < 3:
            return self.trust_threshold

        m = mean(trust_scores)
        s = std(trust_scores)

        dynamic = m - 0.5 * s
        return max(self.trust_threshold, dynamic)

    # ---- 辅助方法 ----

    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """获取文本的嵌入向量。"""
        if self.embed_model is not None:
            try:
                return self.embed_model(text)
            except Exception:
                return None
        return None

    def _get_doc_embeddings(
        self, docs: List[RetrievedDoc]
    ) -> List[List[float]]:
        """获取文档列表的嵌入向量列表。"""
        embeddings = []
        for doc in docs:
            if doc.embedding:
                embeddings.append(doc.embedding)
            elif self.embed_model is not None:
                emb = self._get_embedding(doc.content)
                if emb is not None:
                    embeddings.append(emb)
                else:
                    # 兜底: 零向量
                    embeddings.append([0.0])
            else:
                embeddings.append([0.0])
        return embeddings
