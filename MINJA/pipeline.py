"""
MINJA 纵深防御主管线。

将 D1-D6 六个防御节点编排为两条完整的拦截路径：

  写入路径（on_write_request）：
    D1（快速意图筛查）
    → D2（因果归因，D1 FLAG 或 always_run 时触发）
    → D3（前瞻仿真，D2 边界区域时触发）
    → D4（溯源标签绑定，写入执行）

  检索路径（on_retrieval）：
    D4 验签（过滤被篡改条目）
    → D5（Hubness + 语义一致性图，集合级检测）

  执行路径（on_tool_call）：
    D6（任务对齐最终核查）

设计原则：
  1. 任意节点返回 BLOCK → 操作终止，返回 None + 判决列表
  2. FLAG → 升级至后续更精确的节点，不终止
  3. ASK  → 暂停并通知调用方等待人工确认
  4. 各节点可独立通过 config 禁用，便于 A/B 测试单节点贡献
"""
from __future__ import annotations

import sys
import time
import json
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .config import PipelineConfig
from .d1_query_intent import D1QueryIntentDetector
from .d2_causal_write import D2CausalWriteAuditor
from .d3_prospective_sim import D3ProspectiveSimulator
from .d4_provenance import D4ProvenanceBinder
from .d5_retrieval_audit import D5RetrievalSetAuditor, HubnessTracker
from .d6_execution_align import D6ExecutionAlignmentGuard
from .types import (
    CandidateEntry,
    DefenseVerdict,
    IntegrityLabel,
    RetrievedEntry,
    SourceLabel,
    ToolCallRequest,
    WriteContext,
)

EmbedFn = Callable[[list[str]], np.ndarray]
LLMClient = object


# ── 管线执行结果 ──────────────────────────────────────────────────────────────

@dataclass
class WriteResult:
    """写入路径的执行结果。"""
    accepted: bool                        # True = 写入被接受（D4 已绑定溯源）
    entry: Optional[CandidateEntry]       # 接受时为绑定了 ProvenanceTag 的条目
    verdicts: list[DefenseVerdict]        # 各节点的判决序列（用于审计）
    blocked_by: Optional[str] = None     # 被哪个节点拦截（BLOCK 时填入）


@dataclass
class RetrievalResult:
    """检索路径的执行结果。"""
    entries: list[RetrievedEntry]         # 经 D4 验签 + D5 降权后的条目（已排序）
    verdicts: list[DefenseVerdict]
    tampered_count: int = 0              # D4 验签失败的条目数


@dataclass
class ToolCallResult:
    """执行路径的执行结果。"""
    verdict: DefenseVerdict
    proceed: bool                         # True = 可以执行工具调用
    approval_required: bool = False       # True = 需要等待人工确认（ASK）


# ── 审计日志 ──────────────────────────────────────────────────────────────────

def _audit_log(path: Optional[str], event: dict) -> None:
    """将审计事件写入 JSONL 文件（同时打印到 stderr）。"""
    line = json.dumps(event, ensure_ascii=False, default=str)
    print(f"[MINJA-AUDIT] {line}", file=sys.stderr)
    if path:
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass


# ── 主管线 ────────────────────────────────────────────────────────────────────

class MINJADefensePipeline:
    """
    MINJA 纵深防御主管线。

    初始化示例：
        from openai import OpenAI
        import numpy as np

        client = OpenAI(api_key="...")

        def embed(texts):
            resp = client.embeddings.create(
                model="text-embedding-3-small", input=texts
            )
            return np.array([d.embedding for d in resp.data])

        pipeline = MINJADefensePipeline.from_config(
            PipelineConfig(openai_api_key="..."),
            embed_fn=embed,
            llm_client=client,
        )
    """

    def __init__(
        self,
        config: PipelineConfig,
        d1: D1QueryIntentDetector,
        d2: D2CausalWriteAuditor,
        d3: D3ProspectiveSimulator,
        d4: D4ProvenanceBinder,
        d5: D5RetrievalSetAuditor,
        d6: D6ExecutionAlignmentGuard,
        hubness_tracker: HubnessTracker,
    ):
        self.cfg = config
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3
        self.d4 = d4
        self.d5 = d5
        self.d6 = d6
        self.hubness = hubness_tracker

    @classmethod
    def from_config(
        cls,
        config: PipelineConfig,
        embed_fn: Optional[EmbedFn] = None,
        llm_client: Optional[LLMClient] = None,
    ) -> "MINJADefensePipeline":
        """
        工厂方法：从 PipelineConfig 一键构建完整管线。
        embed_fn 和 llm_client 按需注入，未使用的节点不需要对应依赖。
        """
        d1 = D1QueryIntentDetector(config.d1, embed_fn=embed_fn)
        d2 = D2CausalWriteAuditor(config.d2, llm_client=llm_client)
        d3 = D3ProspectiveSimulator(config.d3, llm_client=llm_client)
        d4 = D4ProvenanceBinder(config.d4)
        d5 = D5RetrievalSetAuditor(config.d5, embed_fn=embed_fn)
        d6 = D6ExecutionAlignmentGuard(config.d6, embed_fn=embed_fn, llm_client=llm_client)
        hubness = HubnessTracker()
        return cls(config, d1, d2, d3, d4, d5, d6, hubness)

    # ── 写入路径 ──────────────────────────────────────────────────────────────

    def on_write_request(
        self,
        entry: CandidateEntry,
        write_ctx: WriteContext,
        source_labels: list[SourceLabel],
    ) -> WriteResult:
        """
        写入路径：D1 → D2 → D3 → D4。

        参数：
          entry        : 待写入的候选条目（content 已填入，embedding 可选）
          write_ctx    : 写入上下文（用户目标、indication prompt 等）
          source_labels: 所有信息源的 IFC 标签列表（供 D4 格 Join 使用）

        返回 WriteResult：
          accepted=True  → D4 已绑定溯源标签，entry.provenance 已填入
          accepted=False → BLOCK，entry=None
        """
        verdicts: list[DefenseVerdict] = []
        t0 = time.time()

        # ── D1：快速意图筛查 ──────────────────────────────────────────────────
        # 写入路径优先检查 indication_prompt（攻击者控制的载荷内容），
        # 无 indication 时退回 triggering_query（兼容正常写入场景）。
        d1_input = write_ctx.indication_prompt or write_ctx.triggering_query
        v1 = self.d1.check(d1_input)
        verdicts.append(v1)
        needs_d2 = (v1.action == "FLAG") or self.cfg.d2.always_run

        if v1.action == "BLOCK":
            return self._blocked(entry, verdicts, "D1", write_ctx, t0)

        # ── D2：因果归因 ──────────────────────────────────────────────────────
        needs_d3 = False
        if needs_d2:
            v2 = self.d2.check(write_ctx)
            verdicts.append(v2)

            if v2.action == "BLOCK":
                return self._blocked(entry, verdicts, "D2", write_ctx, t0)

            # D2 边界区域 → FLAG，升级 D3
            needs_d3 = (v2.action == "FLAG") and self.cfg.d3.enabled

        # ── D3：前瞻仿真（仅边界区域触发，成本最高）─────────────────────────
        if needs_d3 or (not self.cfg.d3.trigger_on_boundary_only and self.cfg.d3.enabled):
            v3 = self.d3.check(write_ctx.candidate_content)
            verdicts.append(v3)

            if v3.action == "BLOCK":
                return self._blocked(entry, verdicts, "D3", write_ctx, t0)

        # ── D4：溯源标签绑定（写入执行）─────────────────────────────────────
        self.d4.bind(entry, source_labels, write_ctx.triggering_query)
        verdicts.append(DefenseVerdict(
            node="D4", passed=True, score=1.0,
            reason=(
                f"溯源绑定完成：label={entry.provenance.label.name}"  # type: ignore
                if entry.provenance else "D4 已禁用"
            ),
            action="PASS",
        ))

        elapsed = round(time.time() - t0, 4)
        _audit_log(self.cfg.audit_log_path, {
            "event": "write_accepted",
            "entry_id": entry.entry_id,
            "elapsed_s": elapsed,
            "nodes_run": [v.node for v in verdicts],
        })
        return WriteResult(accepted=True, entry=entry, verdicts=verdicts)

    # ── 检索路径 ──────────────────────────────────────────────────────────────

    def on_retrieval(
        self,
        retrieved: list[RetrievedEntry],
        user_task_embedding: list[float],
    ) -> RetrievalResult:
        """
        检索路径：D4 验签 → D5 集合级检测。

        参数：
          retrieved          : 从记忆库召回的条目列表
          user_task_embedding: 用户原始任务的嵌入向量（D5 对齐计算用）

        返回 RetrievalResult：
          entries 中可疑条目已降权（weight < 1.0）并标记（flagged=True），
          但不会被删除，由上层决策是否使用。
        """
        verdicts: list[DefenseVerdict] = []
        tampered = 0

        # ── D4 验签：过滤被篡改条目 ──────────────────────────────────────────
        verified: list[RetrievedEntry] = []
        for entry in retrieved:
            # 将 RetrievedEntry 临时包装为 CandidateEntry 进行签名验证
            candidate = CandidateEntry(
                entry_id=entry.entry_id,
                content=entry.content,
                provenance=entry.provenance,
            )
            ok, reason = self.d4.verify(candidate)
            if ok:
                verified.append(entry)
            else:
                tampered += 1
                verdicts.append(DefenseVerdict(
                    node="D4", passed=False, score=0.0,
                    reason=f"签名验证失败（{reason}）",
                    action="BLOCK",
                    metadata={"entry_id": entry.entry_id},
                ))

        # 更新 Hubness 追踪器（记录本次检索的命中情况）
        self.hubness.record_query([e.entry_id for e in verified])

        # ── D5：Hubness + 语义一致性图 ───────────────────────────────────────
        filtered, d5_verdicts = self.d5.check(
            verified, user_task_embedding, self.hubness
        )
        verdicts.extend(d5_verdicts)

        # 按权重降序排列（未被降权的条目优先）
        filtered.sort(key=lambda e: e.weight, reverse=True)

        _audit_log(self.cfg.audit_log_path, {
            "event": "retrieval_audited",
            "total": len(retrieved),
            "tampered": tampered,
            "flagged": sum(1 for e in filtered if e.flagged),
        })
        return RetrievalResult(
            entries=filtered,
            verdicts=verdicts,
            tampered_count=tampered,
        )

    # ── 执行路径 ──────────────────────────────────────────────────────────────

    def on_tool_call(self, req: ToolCallRequest) -> ToolCallResult:
        """
        执行路径：D6 任务对齐最终核查。

        返回 ToolCallResult：
          proceed=True,  approval_required=False → 直接执行
          proceed=False, approval_required=True  → 等待人工确认（ASK）
          proceed=False, approval_required=False → 此路径不出现（D6 不发 BLOCK）
        """
        verdict = self.d6.check(req)

        _audit_log(self.cfg.audit_log_path, {
            "event": "tool_call_checked",
            "tool": req.tool_name,
            "action": verdict.action,
            "alignment": verdict.score,
        })

        if verdict.action == "ASK":
            return ToolCallResult(
                verdict=verdict, proceed=False, approval_required=True
            )
        return ToolCallResult(verdict=verdict, proceed=True)

    # ── 内部工具方法 ──────────────────────────────────────────────────────────

    def _blocked(
        self,
        entry: CandidateEntry,
        verdicts: list[DefenseVerdict],
        blocked_by: str,
        ctx: WriteContext,
        t0: float,
    ) -> WriteResult:
        """统一处理 BLOCK 结果：记录审计日志并返回 WriteResult(accepted=False)。"""
        elapsed = round(time.time() - t0, 4)
        _audit_log(self.cfg.audit_log_path, {
            "event": "write_blocked",
            "entry_id": entry.entry_id,
            "blocked_by": blocked_by,
            "elapsed_s": elapsed,
            "query_preview": ctx.triggering_query[:80],
        })
        return WriteResult(
            accepted=False,
            entry=None,
            verdicts=verdicts,
            blocked_by=blocked_by,
        )
