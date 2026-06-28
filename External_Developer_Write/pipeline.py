"""
pipeline.py — 防御管线编排主入口

将 SP1–SP7 七个防御节点编织为完整的拦截路径。
支持 5 种配置模式（Config-1 ~ Config-5），
覆盖文档上传 → 分块处理 → 检索验证的全链路。

管线入口（4 个）:
    1. upload_document(doc)     — 文档上传时检测（SP1, SP2）
    2. process_chunks(doc_id, chunks) — 分块后检测（SP3, SP7）
    3. retrieve_and_verify(query, top_k) — 检索时检测（SP5, SP6）
    4. periodic_scan()          — 后台周期扫描（SP4）

设计原则:
    - 松耦合: 管线只负责编排，不关心各节点的内部实现
    - 可配置: 通过 PipelineConfig 控制启用哪些节点
    - 可扩展: 新增节点只需在 pipeline 中注册
    - 降级策略: 某节点失败时自动跳过，不影响其他节点
"""

import time
from typing import List, Optional, Dict, Any, Callable

from .types import (
    Document, DocumentEmbedding, ChunkInfo, DetectionResult,
    DefenseAlert, PipelineConfig
)

# 防御节点导入
from .sp1_embedding_anomaly import EmbeddingAnomalyDetector
from .sp2_content_perplexity import ContentPerplexityAnalyzer
from .sp3_cross_chunk_coherence import CrossChunkCoherenceVerifier
from .sp4_trigger_region import TriggerRegionDetector
from .sp5_robust_aggregation import RobustAggregationRetriever
from .sp6_post_retrieval_verifier import PostRetrievalVerifier
from .sp7_semantic_graph import SemanticDependencyGraphAnalyzer


class DefensePipeline:
    """类型四防御管线主入口。

    负责:
    1. 根据配置初始化所需的防御节点
    2. 提供统一的文档上传/分块/检索入口
    3. 收集各节点的检测结果，生成综合决策
    4. 支持降级策略（某节点失败不影响其他节点）

    使用方式:
        pipeline = DefensePipeline(PipelineConfig.CONFIG_5_MAX, {
            "sp1": {"knn_k": 20},
            "sp2": {"alpha": 0.025},
            ...
        })
        pipeline.initialize()

        # 文档上传
        result = await pipeline.upload_document(doc)
        if not result["allowed"]:
            reject_document(doc)

        # 检索验证
        answer = await pipeline.retrieve_and_verify(query, docs)
    """

    # 配置名称 → 包含的 SP 集合
    _CONFIG_SP_MAP = {
        PipelineConfig.CONFIG_1_FAST: {"sp1", "sp2"},
        PipelineConfig.CONFIG_2_STANDARD: {"sp1", "sp2", "sp3"},
        PipelineConfig.CONFIG_3_FULL_UPLOAD: {"sp1", "sp2", "sp3", "sp4"},
        PipelineConfig.CONFIG_4_RETRIEVAL: {"sp5", "sp6"},
        PipelineConfig.CONFIG_5_MAX: {"sp1", "sp2", "sp3", "sp4", "sp5", "sp6", "sp7"},
    }

    def __init__(
        self,
        config: PipelineConfig,
        init_params: Optional[Dict[str, Dict[str, Any]]] = None,
        embed_model: Optional[Callable] = None,
        llm_model: Optional[Callable] = None,
        cross_encoder: Optional[Callable] = None,
        nli_model: Optional[Callable] = None,
        nlp_extractor: Optional[Callable] = None,
    ):
        """初始化管线。

        Args:
            config: 管线配置模式
            init_params: 各节点的初始化参数。
                         格式: {"sp1": {"knn_k": 20}, "sp2": {"alpha": 0.025}, ...}
            embed_model:   嵌入模型（所有 SP 共享）
            llm_model:     语言模型（SP2, SP3, SP5 共享）
            cross_encoder: 交叉编码器（SP6 专用）
            nli_model:     NLI 模型（SP6 专用）
            nlp_extractor: 语义三元组提取器（SP7 专用）
        """
        self.config = config
        self.init_params = init_params or {}
        self.embed_model = embed_model
        self.llm_model = llm_model
        self.cross_encoder = cross_encoder
        self.nli_model = nli_model
        self.nlp_extractor = nlp_extractor

        # 确定的 SP 集合
        self._active_sps = self._CONFIG_SP_MAP.get(config, set())

        # 延迟初始化的检测器实例
        self._detectors: Dict[str, Any] = {}
        self._initialized = False

    # ---- 初始化 ----

    def initialize(self) -> None:
        """初始化所有启用的防御节点。

        在管线开始处理任何请求前调用一次。
        执行:
        1. 按需创建各 SP 实例
        2. 对需要预训练的节点（SP1, SP2）预留训练接口
        """
        sp_params = self.init_params

        if "sp1" in self._active_sps:
            self._detectors["sp1"] = EmbeddingAnomalyDetector(
                **sp_params.get("sp1", {})
            )

        if "sp2" in self._active_sps:
            self._detectors["sp2"] = ContentPerplexityAnalyzer(
                lm_model=self.llm_model,
                **sp_params.get("sp2", {})
            )

        if "sp3" in self._active_sps:
            self._detectors["sp3"] = CrossChunkCoherenceVerifier(
                embed_model=self.embed_model,
                llm_model=self.llm_model,
                **sp_params.get("sp3", {})
            )

        if "sp4" in self._active_sps:
            self._detectors["sp4"] = TriggerRegionDetector(
                **sp_params.get("sp4", {})
            )

        if "sp5" in self._active_sps:
            self._detectors["sp5"] = RobustAggregationRetriever(
                llm_model=self.llm_model,
                embed_model=self.embed_model,
                **sp_params.get("sp5", {})
            )

        if "sp6" in self._active_sps:
            self._detectors["sp6"] = PostRetrievalVerifier(
                cross_encoder=self.cross_encoder,
                embed_model=self.embed_model,
                nli_model=self.nli_model,
                **sp_params.get("sp6", {})
            )

        if "sp7" in self._active_sps:
            self._detectors["sp7"] = SemanticDependencyGraphAnalyzer(
                nlp_extractor=self.nlp_extractor,
                **sp_params.get("sp7", {})
            )

        self._initialized = True

    # ---- SP1 训练接口 ----

    def train_sp1(self, clean_docs: List[DocumentEmbedding]) -> None:
        """训练 SP1 的 CleanStats。

        在系统初始化后、处理任何文档前，用一批已知干净的文档
        调用此方法训练统计量。

        Args:
            clean_docs: 已知干净的文档嵌入列表
        """
        if "sp1" not in self._detectors:
            return

        detector: EmbeddingAnomalyDetector = self._detectors["sp1"]
        clean_stats = detector.train(clean_docs)
        # 存储供后续检测使用
        self._sp1_clean_stats = clean_stats

    def train_sp2(self, clean_docs: List[Document]) -> None:
        """训练 SP2 的 PPL 经验分布。

        Args:
            clean_docs: 已知干净的文档列表
        """
        if "sp2" not in self._detectors:
            return

        analyzer: ContentPerplexityAnalyzer = self._detectors["sp2"]
        analyzer.train_distribution(clean_docs)

    # ---- 入口 1: 文档上传 ----

    def upload_document(
        self, doc: Document, doc_embedding: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """文档上传时的防御检查。

        在文档写入向量数据库前调用。
        执行 SP1（嵌入异常检测）+ SP2（PPL 分析）。

        Args:
            doc:          上传的文档
            doc_embedding: 文档的嵌入向量（可选，None 时使用 embed_model）

        Returns:
            {
                "doc_id": str,
                "allowed": bool,       # True = 可入库
                "alerts": List[DefenseAlert],
                "details": {node: DetectionResult, ...},
                "elapsed_ms": float
            }
        """
        if not self._initialized:
            self.initialize()

        start_time = time.time()
        alerts: List[DefenseAlert] = []
        details: Dict[str, DetectionResult] = {}
        allowed = True

        # SP1: 嵌入空间异常检测
        if "sp1" in self._detectors and hasattr(self, "_sp1_clean_stats"):
            try:
                detector: EmbeddingAnomalyDetector = self._detectors["sp1"]
                if doc_embedding is None and self.embed_model is not None:
                    doc_embedding = self._get_embedding(doc.content)

                if doc_embedding is not None:
                    sp1_result = detector.detect(
                        doc_embedding, self._sp1_clean_stats
                    )
                    details["sp1"] = sp1_result
                    if sp1_result.is_anomaly:
                        alerts.append(DefenseAlert(
                            node="SP1",
                            reason=sp1_result.reason,
                            score=sp1_result.anomaly_score,
                            doc_id=doc.doc_id,
                            timestamp=int(time.time()),
                        ))
                        if self.config == PipelineConfig.CONFIG_5_MAX:
                            allowed = False
            except Exception as e:
                # 降级: SP1 失败不影响后续
                details["sp1"] = DetectionResult(
                    doc_id=doc.doc_id, reason=f"SP1 异常: {e}"
                )

        # SP2: PPL 困惑度分析
        if "sp2" in self._detectors:
            try:
                analyzer: ContentPerplexityAnalyzer = self._detectors["sp2"]
                sp2_result = analyzer.analyze(doc)
                details["sp2"] = sp2_result
                if sp2_result.is_anomaly:
                    alerts.append(DefenseAlert(
                        node="SP2",
                        reason=sp2_result.reason,
                        score=sp2_result.anomaly_score,
                        doc_id=doc.doc_id,
                        timestamp=int(time.time()),
                    ))
                    if self.config == PipelineConfig.CONFIG_5_MAX:
                        allowed = False
            except Exception as e:
                details["sp2"] = DetectionResult(
                    doc_id=doc.doc_id, reason=f"SP2 异常: {e}"
                )

        elapsed = (time.time() - start_time) * 1000

        return {
            "doc_id": doc.doc_id,
            "allowed": allowed,
            "alerts": alerts,
            "details": details,
            "elapsed_ms": round(elapsed, 2),
        }

    # ---- 入口 2: 分块处理 ----

    def process_chunks(
        self, doc_id: str, chunks: List[ChunkInfo]
    ) -> Dict[str, Any]:
        """文档分块后的防御检查。

        在 chunk 写入向量数据库前调用。
        执行 SP3（跨块连贯性验证）+ 标记 SP7 待执行。

        Args:
            doc_id: 文档 ID
            chunks: 分块列表

        Returns:
            {
                "doc_id": str,
                "chunks_accepted": bool,
                "alerts": List[DefenseAlert],
                "details": {node: DetectionResult, ...},
                "elapsed_ms": float
            }
        """
        if not self._initialized:
            self.initialize()

        start_time = time.time()
        alerts: List[DefenseAlert] = []
        details: Dict[str, Any] = {}
        chunks_accepted = True

        # SP3: 跨块语义连贯性验证
        if "sp3" in self._detectors:
            try:
                verifier: CrossChunkCoherenceVerifier = self._detectors["sp3"]
                sp3_results = verifier.verify(chunks)
                details["sp3"] = sp3_results

                # 检查是否有异常边界
                anomaly_boundaries = [
                    r for r in sp3_results if r.is_anomaly
                ]
                if anomaly_boundaries:
                    boundary_info = "; ".join(
                        r.reason for r in anomaly_boundaries
                    )
                    alerts.append(DefenseAlert(
                        node="SP3",
                        reason=f"检测到 {len(anomaly_boundaries)} 个异常 chunk 边界: {boundary_info}",
                        score=mean([r.anomaly_score for r in anomaly_boundaries]),
                        doc_id=doc_id,
                        timestamp=int(time.time()),
                    ))
                    if self.config == PipelineConfig.CONFIG_5_MAX:
                        chunks_accepted = False
            except Exception as e:
                details["sp3"] = f"SP3 异常: {e}"

        # SP7 标记（语义图分析在批次级别进行，此处只记录）
        if "sp7" in self._detectors:
            details["sp7_pending"] = True

        elapsed = (time.time() - start_time) * 1000

        return {
            "doc_id": doc_id,
            "chunks_accepted": chunks_accepted,
            "alerts": alerts,
            "details": details,
            "elapsed_ms": round(elapsed, 2),
        }

    # ---- 入口 3: 检索验证 ----

    def retrieve_and_verify(
        self,
        query: str,
        retrieved_docs: List["RetrievedDoc"],  # noqa: F821
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """检索阶段的防御检查。

        在 LLM 调用前执行。
        执行 SP5（鲁棒聚合）或 SP6（后检索验证）。

        Args:
            query:          用户查询
            retrieved_docs: 向量数据库返回的检索结果
            top_k:          最多处理前 k 条

        Returns:
            {
                "response": str | None,   # 最终 LLM 回答
                "strategy": str,           # 使用的策略
                "is_robust": bool,
                "details": dict,
                "alerts": List[DefenseAlert],
                "elapsed_ms": float
            }
        """
        if not self._initialized:
            self.initialize()

        start_time = time.time()
        alerts: List[DefenseAlert] = []

        # 截断到 top_k
        docs = retrieved_docs[:top_k]

        # Config-4 或 Config-5: 使用 SP5 鲁棒聚合
        if "sp5" in self._detectors:
            try:
                retriever: RobustAggregationRetriever = self._detectors["sp5"]
                sp5_result = retriever.retrieve(query, docs)

                elapsed = (time.time() - start_time) * 1000

                return {
                    "response": sp5_result.get("final_response"),
                    "strategy": "robust_aggregation",
                    "is_robust": sp5_result.get("is_robust", False),
                    "details": sp5_result,
                    "alerts": alerts,
                    "elapsed_ms": round(elapsed, 2),
                }
            except Exception as e:
                details = {"sp5_error": str(e)}

        # 非鲁棒聚合模式或 SP5 降级: 使用 SP6
        if "sp6" in self._detectors:
            try:
                verifier: PostRetrievalVerifier = self._detectors["sp6"]

                # 获取 LLM 基础回答（无上下文）
                llm_base = None
                if self.llm_model is not None:
                    try:
                        llm_base = self.llm_model(
                            f"请回答以下问题: {query}", max_tokens=128
                        )
                    except Exception:
                        pass

                trusted, filtered, sp6_details = verifier.verify(
                    query, docs, llm_base_response=llm_base
                )

                # 用可信文档构造最终上下文
                final_response = None
                if trusted:
                    from .utils import format_context, build_prompt
                    if self.llm_model is not None:
                        context = format_context(trusted)
                        prompt = build_prompt(query, context)
                        try:
                            final_response = self.llm_model(
                                prompt, max_tokens=512
                            )
                        except Exception:
                            final_response = llm_base
                else:
                    final_response = llm_base

                if filtered:
                    alerts.append(DefenseAlert(
                        node="SP6",
                        reason=f"过滤了 {len(filtered)}/{len(docs)} 条检索结果",
                        doc_id="|".join(d.doc_id for d in filtered),
                        timestamp=int(time.time()),
                    ))

                elapsed = (time.time() - start_time) * 1000

                return {
                    "response": final_response,
                    "strategy": "post_retrieval_verification",
                    "is_robust": len(trusted) > 0,
                    "details": sp6_details,
                    "filtered_docs": [d.doc_id for d in filtered],
                    "trusted_docs": [d.doc_id for d in trusted],
                    "alerts": alerts,
                    "elapsed_ms": round(elapsed, 2),
                }
            except Exception as e:
                return {
                    "response": None,
                    "strategy": "error",
                    "is_robust": False,
                    "details": {"sp6_error": str(e)},
                    "alerts": alerts,
                    "elapsed_ms": round((time.time() - start_time) * 1000, 2),
                }

        # 无可用检索防御
        return {
            "response": None,
            "strategy": "none",
            "is_robust": False,
            "details": {"warning": "未配置检索阶段防御节点"},
            "alerts": alerts,
            "elapsed_ms": round((time.time() - start_time) * 1000, 2),
        }

    # ---- 入口 4: 周期扫描 ----

    def periodic_scan(
        self, all_docs: List[DocumentEmbedding]
    ) -> Dict[str, Any]:
        """全库周期扫描（后台任务）。

        执行 SP4 触发词区域检测。

        Args:
            all_docs: 向量数据库中的全量文档嵌入

        Returns:
            {
                "scan_id": str,
                "n_scanned": int,
                "n_anomalies": int,
                "high_risk_ids": List[str],
                "alerts": List[DefenseAlert],
                "elapsed_ms": float
            }
        """
        if not self._initialized:
            self.initialize()

        start_time = time.time()
        alerts: List[DefenseAlert] = []

        if "sp4" not in self._detectors:
            return {
                "scan_id": f"scan_{int(time.time())}",
                "n_scanned": len(all_docs),
                "n_anomalies": 0,
                "high_risk_ids": [],
                "alerts": alerts,
                "elapsed_ms": 0.0,
            }

        try:
            detector: TriggerRegionDetector = self._detectors["sp4"]
            results = detector.scan(all_docs)

            # 高置信度异常
            high_risk = [r for r in results if r.anomaly_score > 0.7]

            for hr in high_risk:
                alerts.append(DefenseAlert(
                    node="SP4",
                    reason=hr.reason,
                    score=hr.anomaly_score,
                    doc_id=hr.doc_id,
                    timestamp=int(time.time()),
                    details={"recommendation": "审查并隔离该文档"},
                ))

            # SP7: 批次级别的语义图分析（如果启用）
            sp7_results = None
            if "sp7" in self._detectors:
                try:
                    # 按批次/来源分组
                    # 简化: 使用全库的前 N 个文档做图分析
                    from .sp7_semantic_graph import SemanticDependencyGraphAnalyzer
                    # SP7 需要 Document 对象，这里跳过
                    pass
                except Exception:
                    pass

            elapsed = (time.time() - start_time) * 1000

            return {
                "scan_id": f"scan_{int(time.time())}",
                "n_scanned": len(all_docs),
                "n_anomalies": len(high_risk),
                "high_risk_ids": [r.doc_id for r in high_risk],
                "anomaly_scores": [
                    r.anomaly_score for r in high_risk
                ],
                "alerts": alerts,
                "sp4_detailed": results,
                "elapsed_ms": round(elapsed, 2),
            }

        except Exception as e:
            return {
                "scan_id": f"scan_{int(time.time())}",
                "n_scanned": len(all_docs),
                "n_anomalies": 0,
                "high_risk_ids": [],
                "alerts": alerts,
                "error": str(e),
                "elapsed_ms": round((time.time() - start_time) * 1000, 2),
            }

    # ---- 辅助 ----

    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """获取文本嵌入向量。"""
        if self.embed_model is not None:
            try:
                return self.embed_model(text)
            except Exception:
                return None
        return None

    @property
    def active_sps(self) -> List[str]:
        """返回当前配置下启用的 SP 列表。"""
        return sorted(self._active_sps)

    @property
    def status(self) -> Dict[str, Any]:
        """返回管线状态概览。"""
        return {
            "config": self.config.value,
            "initialized": self._initialized,
            "active_nodes": list(self._active_sps),
            "detectors_loaded": list(self._detectors.keys()),
        }


def mean(values: List[float]) -> float:
    """工具: 列表平均。"""
    if not values:
        return 0.0
    return sum(values) / len(values)
