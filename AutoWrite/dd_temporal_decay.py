"""
D-D：时序衰减检索（Temporal Decay Retrieval）。

防御目标
--------
削弱跨会话持久化污染的长期影响。
攻击者注入的记忆随时间自然衰减，使其对未来检索的影响持续降低，
最终趋近于零——即使 D4/D5 未能识别该条目为恶意。

衰减公式
--------
    score_decay(m_i, q, t) = original_weight_i · exp(-λ · (t - t_i))

其中：
  t      = 当前时间（Unix 秒）
  t_i    = 条目写入时间（从 ProvenanceTag.write_time 读取）
  λ      = 衰减速率，半衰期 T_{1/2} = ln(2) / λ
  λ=1e-5 → T_{1/2} ≈ 19.3 小时（19 小时前写入的记忆权重减半）

攻击者时序优势分析（以 λ=1e-5 为例）：
  注入 1 天前  → 衰减因子 ≈ 0.42（权重降至 42%）
  注入 1 周前  → 衰减因子 ≈ 0.002（权重降至 0.2%，实际失效）

设计局限
--------
若攻击者能持续高频刷新注入（注入频率 > 1/T_{1/2}），衰减效果失效。
必须与 D-B（写入速率限制）联合部署，才能同时阻断高频刷新路径。

整合位置
--------
on_retrieval() 中，D5 集合级检测之后、D-E 对齐核查之前。
纯重排序操作，不产出 BLOCK 判决，不删除条目，仅调整 weight。
"""
from __future__ import annotations

import math
import time as _time_mod
from typing import Optional

from .config import DDConfig
from .types import DefenseVerdict, RetrievedEntry


class DDTemporalDecayReranker:
    """
    D-D 时序衰减重排序节点。

    对检索结果按指数衰减公式重新加权，旧条目权重降低，
    重排序后返回按新权重降序排列的条目列表。

    不产出 BLOCK/FLAG 判决——定位为后处理步骤，不拦截检索结果。
    仅通过权重调整影响上层 LLM 使用记忆时的优先级。

    用法
    ----
        reranker = DDTemporalDecayReranker(DDConfig())
        entries, verdict = reranker.rerank(entries)
    """

    def __init__(self, config: DDConfig):
        self.cfg = config

    def rerank(
        self,
        entries: list[RetrievedEntry],
        current_time: Optional[float] = None,
    ) -> tuple[list[RetrievedEntry], DefenseVerdict]:
        """
        对检索结果执行时序衰减重排序。

        参数
        ----
        entries      : 检索结果列表（weight 字段将被原地修改）
        current_time : 当前时间戳（Unix 秒）；None 时取系统时间，
                       测试时注入固定值以保证可重复性

        返回
        ----
        entries : 按新 weight 降序排列的条目列表（原地修改）
        verdict : 本次重排序的审计判决（始终为 PASS）
        """
        if not self.cfg.enabled or not entries:
            return entries, DefenseVerdict(
                "D-D", passed=True, score=1.0,
                reason="D-D 已禁用或无条目", action="PASS",
            )

        t_now = current_time or _time_mod.time()
        decay_log = []

        for entry in entries:
            # apply_to_flagged_only 模式：只对 D5/D-E 标记的可疑条目衰减
            if self.cfg.apply_to_flagged_only and not entry.flagged:
                decay_log.append({
                    "entry_id": entry.entry_id[:8],
                    "skipped": True,
                    "weight": round(entry.weight, 4),
                })
                continue

            # 从 ProvenanceTag 读取写入时间；无溯源时默认 t=0（最大衰减）
            t_write = (
                entry.provenance.write_time
                if entry.provenance is not None
                else 0.0
            )
            delta_t = max(0.0, t_now - t_write)

            # 指数衰减，下界 min_weight 防止极老记忆权重完全归零
            decay_factor = max(
                math.exp(-self.cfg.lambda_decay * delta_t),
                self.cfg.min_weight,
            )
            original_weight = entry.weight
            entry.weight = original_weight * decay_factor

            decay_log.append({
                "entry_id": entry.entry_id[:8],
                "delta_t_h": round(delta_t / 3600, 1),    # 经过小时数
                "decay":     round(decay_factor, 4),
                "w_before":  round(original_weight, 4),
                "w_after":   round(entry.weight, 4),
            })

        # 按新权重降序重排（已降权条目自然后移）
        entries.sort(key=lambda e: e.weight, reverse=True)

        # 计算平均衰减因子（用于审计日志的摘要统计）
        active = [d for d in decay_log if "decay" in d]
        avg_decay = (
            sum(d["decay"] for d in active) / len(active)
            if active else 1.0
        )

        return entries, DefenseVerdict(
            node="D-D", passed=True,
            score=round(avg_decay, 4),
            reason=(
                f"时序衰减重排序完成：λ={self.cfg.lambda_decay:.1e}，"
                f"平均衰减因子={avg_decay:.3f}，处理 {len(active)} 条"
            ),
            action="PASS",
            metadata={"decay_log": decay_log[:10]},  # 截断避免日志过大
        )
