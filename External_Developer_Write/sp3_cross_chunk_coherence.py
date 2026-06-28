"""
sp3_cross_chunk_coherence.py — 跨 chunk 语义连贯性验证（SP3）

对应攻击: 节点2 — 语义混淆注入（跨 chunk 碎片化攻击）
对应分析报告: 5.3 节
核心方法: 余弦连贯性 + LLM 条件概率 + 语义依赖图 三层融合

攻击者将恶意指令碎片化到多个 chunk 中，每个 chunk 单独无害，
检索拼接后重组为完整攻击载荷。SP3 的核心洞察是: 碎片化攻击
在 chunk 边界处会产生语义断裂或非自然的过渡。

数学基础:
    1. 余弦连贯性: Coh_cos(c_i, c_{i+1}) = cos(e_i, e_{i+1})
       — 相邻 chunk 嵌入向量的方向一致性
    2. LLM 条件概率: Coh_LM = (1/K) · Σ log p(w_j | w_{<j}, c_i)
       — 在前一个 chunk 条件下生成下一个 chunk 开头 token 的概率
    3. 语义图边密度: ρ(G) = 2|E| / (|V|(|V|-1))
       — 滑动窗口内 chunk 间的语义连通性

设计决策:
    - 延迟 LLM 检查: 仅当余弦得分 < coarse_threshold 时才触发 LLM 调用
    - 滑动窗口语义图: 分析局部区域的边密度而非全局
"""

import math
from typing import List, Optional, Tuple, Dict, Any, Callable

from .types import ChunkInfo, DetectionResult
from .utils import cosine_similarity, euclidean_distance, EPSILON, mean


class CrossChunkCoherenceVerifier:
    """跨 chunk 语义连贯性验证器。

    验证同一文档内相邻 chunk 之间的语义连贯性，检测碎片化攻击。
    支持三层检测: 余弦连贯性、LLM 条件概率、语义依赖图。

    使用方式:
        verifier = CrossChunkCoherenceVerifier(embed_model, llm_model)
        results = verifier.verify(chunks)
    """

    def __init__(
        self,
        embed_model: Optional[Callable] = None,
        llm_model: Optional[Callable] = None,
        threshold_cos: float = 0.5,
        threshold_llm: float = 0.3,
        coarse_threshold: float = 0.7,
        use_semantic_graph: bool = True,
        window_size: int = 3,
        graph_density_threshold: float = 0.3,
        llm_prefix_len: int = 5,
    ):
        """初始化验证器。

        Args:
            embed_model: 嵌入模型可调用对象。
                         签名: embed_model(text) -> List[float]
            llm_model: 语言模型（用于条件概率）。
                       签名: llm_model(context, target) -> float
            threshold_cos: 余弦连贯性阈值，低于此值标记异常
            threshold_llm: LLM 条件概率阈值，低于此值标记异常
            coarse_threshold: 触发 LLM 检查的粗阈值。
                              余弦 > 此值 → 跳过 LLM 检查（节约开销）
            use_semantic_graph: 是否启用语义图分析
            window_size: 语义图滑动窗口大小
            graph_density_threshold: 图密度异常阈值
            llm_prefix_len: LLM 条件概率检查时使用的前缀 token 数
        """
        self.embed_model = embed_model
        self.llm_model = llm_model
        self.threshold_cos = threshold_cos
        self.threshold_llm = threshold_llm
        self.coarse_threshold = coarse_threshold
        self.use_semantic_graph = use_semantic_graph
        self.window_size = window_size
        self.graph_density_threshold = graph_density_threshold
        self.llm_prefix_len = llm_prefix_len

    # ---- 公有接口 ----

    def verify(self, chunks: List[ChunkInfo]) -> List[DetectionResult]:
        """对文档的所有相邻 chunk 边界执行连贯性验证。

        Args:
            chunks: 有序的 chunk 列表（来自同一文档）

        Returns:
            DetectionResult 列表，每个 chunk 边界一个结果
        """
        if len(chunks) < 2:
            return []

        results: List[DetectionResult] = []

        # ---- 阶段 1: 逐对连贯性分析 ----
        for i in range(len(chunks) - 1):
            c_i = chunks[i]
            c_next = chunks[i + 1]

            # 1a. 余弦连贯性得分
            cos_coherence = self._cosine_coherence(c_i, c_next)

            # 1b. LLM 条件概率连贯性（延迟执行）
            llm_coherence_norm = 1.0
            llm_triggered = False

            if cos_coherence < self.coarse_threshold and self.llm_model is not None:
                llm_coherence_norm = self._llm_conditional_coherence(
                    c_i, c_next
                )
                llm_triggered = True

            # 1c. 单对决策
            is_coherent_pair = (
                cos_coherence >= self.threshold_cos or
                llm_coherence_norm >= self.threshold_llm
            )

            # 异常得分: 余弦和 LLM 中较高的那个的反向
            anomaly_score = 1.0 - max(cos_coherence, llm_coherence_norm)

            reason_parts = [
                f"chunk({i}-{i+1})",
                f"cos={cos_coherence:.3f}",
            ]
            if llm_triggered:
                reason_parts.append(f"llm={llm_coherence_norm:.3f}")

            results.append(DetectionResult(
                doc_id=c_i.doc_id,
                is_anomaly=not is_coherent_pair,
                anomaly_score=anomaly_score,
                reason=" | ".join(reason_parts) if not is_coherent_pair else "normal",
                details={
                    "boundary_i": i,
                    "boundary_j": i + 1,
                    "cos_coherence": round(cos_coherence, 4),
                    "llm_coherence": round(llm_coherence_norm, 4),
                    "llm_was_triggered": llm_triggered,
                    "is_coherent": is_coherent_pair,
                    "anomaly_score": round(anomaly_score, 4),
                },
            ))

        # ---- 阶段 2: 语义依赖图分析 ----
        if self.use_semantic_graph and len(chunks) >= 3:
            graph_anomalies = self._analyze_semantic_graph(chunks)
            for ga in graph_anomalies:
                idx = ga["boundary_index"]
                if idx < len(results):
                    if ga["is_anomaly"]:
                        results[idx].is_anomaly = True
                        results[idx].reason += f" | 图异常: {ga['reason']}"

        return results

    # ---- 内部方法 ----

    def _cosine_coherence(self, c_i: ChunkInfo, c_next: ChunkInfo) -> float:
        """计算两个相邻 chunk 的余弦连贯性得分。

        Coh_cos(c_i, c_{i+1}) = cosine(e_i, e_{i+1})

        Args:
            c_i:     前一个 chunk
            c_next:  后一个 chunk

        Returns:
            [-1, 1] 范围的连贯性得分
        """
        # 获取嵌入（如果已缓存则使用，否则即时编码）
        e_i = self._get_embedding(c_i)
        e_next = self._get_embedding(c_next)

        if e_i is None or e_next is None:
            return 0.0

        return cosine_similarity(e_i, e_next)

    def _llm_conditional_coherence(self, c_i: ChunkInfo,
                                   c_next: ChunkInfo) -> float:
        """计算以 c_i 为条件时 c_{i+1} 开头 token 的条件概率。

        Coh_LM(c_i, c_{i+1}) = (1/K) · Σ log p(w_j | w_{<j}, c_i)

        归一化到 [0, 1] 范围（通过 sigmoid 映射）。

        Args:
            c_i:    前一个 chunk
            c_next: 后一个 chunk

        Returns:
            [0, 1] 范围的归一化条件概率得分
        """
        if self.llm_model is None:
            return 1.0

        # 取 c_{i+1} 的前 K 个 token
        prefix_text = " ".join(c_next.content.split()[:self.llm_prefix_len])
        prefix_tokens = prefix_text.split()

        if len(prefix_tokens) < 2:
            return 0.5

        cond_probs = []
        for t in range(1, len(prefix_tokens)):
            context = " ".join(prefix_tokens[:t])
            full_context = c_i.content + " " + context
            try:
                prob = self.llm_model(full_context, prefix_tokens[t])
                if prob > 0:
                    cond_probs.append(math.log(prob))
                else:
                    cond_probs.append(math.log(EPSILON))
            except Exception:
                cond_probs.append(math.log(EPSILON))

        if not cond_probs:
            return 0.5

        # 平均对数概率 → 归一化到 [0, 1]
        mean_log_prob = mean(cond_probs)
        # 使用 sigmoid 映射: σ(x) = 1 / (1 + e^{-x})
        # 对数概率通常在 [-10, 0] 范围，平移使 0 对应 ~0.5
        normalized = 1.0 / (1.0 + math.exp(-mean_log_prob - 2.0))
        return normalized

    def _get_embedding(self, chunk: ChunkInfo) -> Optional[List[float]]:
        """获取 chunk 的嵌入向量（缓存优先）。"""
        if chunk.embedding is not None:
            return chunk.embedding
        if self.embed_model is not None:
            try:
                return self.embed_model(chunk.content)
            except Exception:
                return None
        return None

    def _analyze_semantic_graph(
        self, chunks: List[ChunkInfo]
    ) -> List[Dict[str, Any]]:
        """滑动窗口语义依赖图分析。

        对每个局部窗口构建语义图，计算边密度。
        碎片化攻击产生的跨 chunk 边密度显著低于正常文档。

        Args:
            chunks: 有序的 chunk 列表

        Returns:
            每个窗口的分析结果列表
        """
        anomalies = []

        for center in range(1, len(chunks) - 1):
            start = max(0, center - self.window_size // 2)
            end = min(len(chunks), center + self.window_size // 2 + 1)
            window = chunks[start:end]
            N = len(window)

            if N < 3:
                continue

            # 构建相似度矩阵
            embeddings = [self._get_embedding(c) for c in window]
            valid_indices = [
                i for i, e in enumerate(embeddings) if e is not None
            ]

            if len(valid_indices) < 3:
                continue

            # 计算有效嵌入间的相似度
            edge_count = 0
            total_pairs = 0
            for a_idx in range(len(valid_indices)):
                for b_idx in range(a_idx + 1, len(valid_indices)):
                    i = valid_indices[a_idx]
                    j = valid_indices[b_idx]
                    sim = cosine_similarity(
                        embeddings[i], embeddings[j]  # type: ignore
                    )
                    total_pairs += 1
                    if sim > 0.6:  # 语义相关阈值
                        edge_count += 1

            density = edge_count / total_pairs if total_pairs > 0 else 0.0

            is_anomaly = density < self.graph_density_threshold
            anomalies.append({
                "boundary_index": center,
                "is_anomaly": is_anomaly,
                "graph_density": round(density, 4),
                "window_start": start,
                "window_end": end,
                "reason": f"图密度={density:.3f}(<{self.graph_density_threshold})",
            })

        return anomalies


