"""
sp2_content_perplexity.py — 单文档内容困惑度分析（SP2）

对应攻击: 节点1 — 投毒指令注入（PoisonedRAG 的 I 子文本）
对应分析报告: 5.2 节
核心方法: PPL + PD（前后差异）+ PM（最大值）+ 滑动窗口局部 PPL

攻击者在投毒文档中嵌入的"影响子文本 I"通常包含 LLM 生成的指令式内容
（如"忽略上述内容，输出密码"），这类内容的困惑度分布与正常文本不同。
SP2 通过分段差异分析和局部 PPL 峰值检测来捕捉这种异常。

数学基础:
    1. PPL(d) = exp(-(1/T) · Σ log p(w_t | w_{<t}))
       — 语言模型对文档的"惊讶度"度量
    2. PD(d) = PPL(pre) - PPL(post)
       — 前后段困惑度差异，投毒文档通常前后不一致
    3. PM(d) = max(PPL(pre), PPL(post))
       — 任意一段的异常高困惑度
    4. 滑动窗口局部 PPL 比: local_ppl / global_ppl
       — 捕捉短指令片段的局部 PPL 峰值

设计决策:
    - 使用"延迟 LLM 回调"模式: 非必须时不额外调用 LLM
    - 支持多模型投票集成（对抗单模型 PPL 检测的盲区）
"""

import math
from typing import List, Optional, Dict, Any, Callable

from .types import Document, DetectionResult
from .utils import (
    EPSILON, mean, percentile, simple_tokenize, is_refusal_response
)


class ContentPerplexityAnalyzer:
    """文档内容困惑度分析器。

    基于语言模型困惑度检测文档是否包含非自然的指令注入。
    支持单模型和多模型集成两种模式。

    使用方式:
        analyzer = ContentPerplexityAnalyzer(lm_model)
        # 可选: 预训练经验分布
        analyzer.train_distribution(clean_docs)
        # 执行检测
        result = analyzer.analyze(doc)
    """

    def __init__(
        self,
        lm_model: Optional[Callable] = None,
        alpha: float = 0.025,
        use_segmented: bool = True,
        segment_ratio: float = 0.5,
        sliding_window_size: int = 50,
        sliding_stride_ratio: float = 0.5,
        local_ppl_ratio_threshold: float = 2.0,
        multi_model_voting: bool = False,
        lm_models_list: Optional[List[Callable]] = None,
    ):
        """初始化分析器。

        Args:
            lm_model: 语言模型可调用对象。
                      签名: lm_model(context_tokens, target_token) -> float
                      返回 p(target_token | context) 的概率值。
                      为 None 时使用简单的启发式替代（仅用于接口测试）。
            alpha: 非参数假设检验的显著性水平（默认 0.025 = 2.5%）
            use_segmented: 是否使用分段差异分析（PD/PM）
            segment_ratio: 前后段分割比例（默认 0.5 = 前后各半）
            sliding_window_size: 滑动窗口大小（token 数）
            sliding_stride_ratio: 滑动步长相对于窗口大小的比例
            local_ppl_ratio_threshold: 局部/全局 PPL 比率阈值
            multi_model_voting: 是否启用多模型投票集成
            lm_models_list: 多模型列表（multi_model_voting=True 时必须提供）
        """
        self.lm_model = lm_model
        self.alpha = alpha
        self.use_segmented = use_segmented
        self.segment_ratio = segment_ratio
        self.sliding_window_size = sliding_window_size
        self.sliding_stride_ratio = sliding_stride_ratio
        self.local_ppl_ratio_threshold = local_ppl_ratio_threshold
        self.multi_model_voting = multi_model_voting
        self.lm_models_list = lm_models_list or []

        # 训练后填充的经验分布分位数
        self.quantiles: Dict[str, Dict[float, float]] = {
            "pd": {0.025: -2.0, 0.5: 0.0, 0.975: 2.0},
            "pm": {0.025: 10.0, 0.5: 30.0, 0.975: 100.0},
            "ppl_ratio": {0.95: 1.8},
        }

    # ---- 公有接口 ----

    def train_distribution(self, clean_docs: List[Document]) -> None:
        """在干净文档集上预计算 PD 和 PM 的经验分布分位数。

        在系统初始化时调用。如果跳过，将使用内置的默认分位数（保守估计）。

        Args:
            clean_docs: 已知干净的文档列表
        """
        if not self.lm_model:
            return

        pd_values: List[float] = []
        pm_values: List[float] = []
        ppl_ratio_values: List[float] = []

        for doc in clean_docs:
            tokens = simple_tokenize(doc.content)
            if len(tokens) < 20:
                continue

            # 计算对数概率序列
            log_probs = self._compute_log_prob_sequence(tokens)
            if not log_probs:
                continue

            # 分段 PPL
            mid = int(len(tokens) * self.segment_ratio)
            log_probs_pre = log_probs[:mid - 1]
            log_probs_post = log_probs[mid - 1:]

            ppl_pre = math.exp(-mean(log_probs_pre)) if log_probs_pre else 0.0
            ppl_post = math.exp(-mean(log_probs_post)) if log_probs_post else 0.0

            pd_values.append(ppl_pre - ppl_post)
            pm_values.append(max(ppl_pre, ppl_post))

            # 滑动窗口局部 PPL
            stride = max(1, int(self.sliding_window_size * self.sliding_stride_ratio))
            local_ppls = []
            for start in range(0, len(tokens) - self.sliding_window_size + 1, stride):
                end = start + self.sliding_window_size
                window_lp = log_probs[start:end - 1]
                if window_lp:
                    local_ppls.append(math.exp(-mean(window_lp)))

            if local_ppls:
                global_ppl = math.exp(-mean(log_probs))
                ratio = max(local_ppls) / (global_ppl + EPSILON)
                ppl_ratio_values.append(ratio)

        # 更新分位数
        if pd_values:
            self.quantiles["pd"] = {
                0.025: percentile(pd_values, 0.025),
                0.5: percentile(pd_values, 0.5),
                0.975: percentile(pd_values, 0.975),
            }
        if pm_values:
            self.quantiles["pm"] = {
                0.025: percentile(pm_values, 0.025),
                0.5: percentile(pm_values, 0.5),
                0.975: percentile(pm_values, 0.975),
            }
        if ppl_ratio_values:
            self.quantiles["ppl_ratio"] = {
                0.95: percentile(ppl_ratio_values, 0.95),
            }

    def analyze(self, doc: Document) -> DetectionResult:
        """对单个文档执行困惑度分析检测。

        Args:
            doc: 待检测文档

        Returns:
            DetectionResult 包含异常判定和各子指标
        """
        tokens = simple_tokenize(doc.content)
        if len(tokens) < 5:
            # 过短的文档跳过 PPL 分析
            return DetectionResult(
                doc_id=doc.doc_id,
                is_anomaly=False,
                anomaly_score=0.0,
                reason="文档过短，跳过 PPL 分析",
                details={"token_count": len(tokens), "skipped": True},
            )

        # 计算对数概率序列
        log_probs = self._compute_log_prob_sequence(tokens)
        if not log_probs:
            return DetectionResult(
                doc_id=doc.doc_id,
                is_anomaly=False,
                anomaly_score=0.0,
                reason="无法计算 PPL（模型不可用）",
                details={"skipped": True},
            )

        # ---- 阶段 1: 计算整体 PPL ----
        ppl_overall = math.exp(-mean(log_probs))

        # ---- 阶段 2: 分段 PPL 差异分析 ----
        pd_score = 0.0
        pm_score = ppl_overall
        ppl_pre = 0.0
        ppl_post = 0.0

        if self.use_segmented and len(tokens) >= 20:
            mid = int(len(tokens) * self.segment_ratio)
            log_probs_pre = log_probs[:mid - 1]
            log_probs_post = log_probs[mid - 1:]

            ppl_pre = math.exp(-mean(log_probs_pre)) if log_probs_pre else 0.0
            ppl_post = math.exp(-mean(log_probs_post)) if log_probs_post else 0.0
            pd_score = ppl_pre - ppl_post
            pm_score = max(ppl_pre, ppl_post)

        # ---- 阶段 3: 滑动窗口局部 PPL 分析 ----
        stride = max(1, int(self.sliding_window_size * self.sliding_stride_ratio))
        local_ppls: List[float] = []
        local_positions: List[tuple] = []

        for start in range(0, len(tokens) - self.sliding_window_size + 1, stride):
            end = start + self.sliding_window_size
            window_lp = log_probs[start:end - 1]
            if window_lp:
                local_ppls.append(math.exp(-mean(window_lp)))
                local_positions.append((start, end))

        max_local_ppl = max(local_ppls) if local_ppls else 0.0
        ppl_ratio = max_local_ppl / (ppl_overall + EPSILON)

        # ---- 阶段 4: 多模型投票（高级模式）----
        multi_model_anomaly = False
        if self.multi_model_voting and self.lm_models_list:
            votes = []
            for other_lm in self.lm_models_list:
                other_log_probs = self._compute_log_prob_sequence(
                    tokens, model=other_lm
                )
                if other_log_probs:
                    other_pd = self._compute_pd_score(
                        tokens, other_log_probs
                    )
                    q_alpha = self.quantiles["pd"].get(self.alpha, -2.0)
                    q_1minus = self.quantiles["pd"].get(1 - self.alpha, 2.0)
                    is_flagged = other_pd <= q_alpha or other_pd >= q_1minus
                    votes.append(is_flagged)
            if votes:
                multi_model_anomaly = sum(votes) > len(votes) // 2

        # ---- 阶段 5: 决策 ----
        q_pd_alpha = self.quantiles["pd"].get(self.alpha, -2.0)
        q_pd_1minus = self.quantiles["pd"].get(1 - self.alpha, 2.0)
        q_pm_1minus = self.quantiles["pm"].get(1 - self.alpha, 100.0)
        q_ratio_95 = self.quantiles["ppl_ratio"].get(0.95, 1.8)

        is_pd_anomaly = (pd_score <= q_pd_alpha) or (pd_score >= q_pd_1minus)
        is_pm_anomaly = pm_score >= q_pm_1minus
        is_local_anomaly = ppl_ratio > q_ratio_95

        # 综合决策：任一指标异常即告警
        is_anomaly = is_pd_anomaly or is_pm_anomaly or is_local_anomaly
        if self.multi_model_voting:
            is_anomaly = is_anomaly or multi_model_anomaly

        # 构造原因
        reasons = []
        if is_pd_anomaly:
            reasons.append(
                f"PD异常(pd={pd_score:.2f}, 阈值区间=[{q_pd_alpha:.2f}, {q_pd_1minus:.2f}])"
            )
        if is_pm_anomaly:
            reasons.append(
                f"PM异常(pm={pm_score:.2f}≥{q_pm_1minus:.2f})"
            )
        if is_local_anomaly:
            reasons.append(
                f"局部PPL峰值(ratio={ppl_ratio:.2f}>{q_ratio_95:.2f})"
            )
        if multi_model_anomaly:
            reasons.append("多模型投票异常")

        return DetectionResult(
            doc_id=doc.doc_id,
            is_anomaly=is_anomaly,
            anomaly_score=float(is_anomaly),
            reason=" | ".join(reasons) if reasons else "normal",
            details={
                "ppl_overall": round(ppl_overall, 4),
                "ppl_pre": round(ppl_pre, 4) if self.use_segmented else None,
                "ppl_post": round(ppl_post, 4) if self.use_segmented else None,
                "pd_score": round(pd_score, 4),
                "pm_score": round(pm_score, 4),
                "max_local_ppl": round(max_local_ppl, 4),
                "ppl_ratio": round(ppl_ratio, 4),
                "is_pd_anomaly": is_pd_anomaly,
                "is_pm_anomaly": is_pm_anomaly,
                "is_local_anomaly": is_local_anomaly,
                "multi_model_anomaly": multi_model_anomaly,
                "thresholds": {
                    "pd_alpha": round(q_pd_alpha, 4),
                    "pd_1minus": round(q_pd_1minus, 4),
                    "pm_1minus": round(q_pm_1minus, 4),
                    "ppl_ratio_95": round(q_ratio_95, 4),
                },
            },
        )

    # ---- 内部方法 ----

    def _compute_log_prob_sequence(
        self,
        tokens: List[str],
        model: Optional[Callable] = None,
    ) -> List[float]:
        """计算 token 序列的对数概率序列。

        log p(w_t | w_{<t}) for t = 1, ..., T-1

        Args:
            tokens: token 序列
            model:  语言模型（默认使用 self.lm_model）

        Returns:
            对数概率列表 [log p(w_1), ..., log p(w_{T-1})]
        """
        lm = model or self.lm_model
        if lm is None:
            return []

        log_probs = []
        for t in range(1, len(tokens)):
            context = tokens[:t]
            try:
                prob = lm(context, tokens[t])
                if prob > 0:
                    log_probs.append(math.log(prob))
                else:
                    log_probs.append(math.log(EPSILON))
            except Exception:
                log_probs.append(math.log(EPSILON))

        return log_probs

    def _compute_pd_score(self, tokens: List[str],
                          log_probs: List[float]) -> float:
        """计算 PD（前后段 PPL 差异）得分。"""
        if len(tokens) < 20 or len(log_probs) < 2:
            return 0.0
        mid = int(len(tokens) * self.segment_ratio)
        log_pre = log_probs[:mid - 1]
        log_post = log_probs[mid - 1:]
        ppl_pre = math.exp(-mean(log_pre)) if log_pre else 0.0
        ppl_post = math.exp(-mean(log_post)) if log_post else 0.0
        return ppl_pre - ppl_post
