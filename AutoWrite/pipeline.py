"""
AutoWrite 纵深防御主管线（AutoWriteDefensePipeline）。

架构概览
--------
本管线是 MINJA MINJADefensePipeline 的扩展包装层，
专门针对"类型一：框架自动写入"路径（LangChain ConversationBufferMemory
等框架的 save_context() 无过滤直写）提供完整的六节点纵深防御。

完整写入路径
-----------
    D-A（token净化）
    → D-B（选择性写入）
    → [委托 MINJA: D1→D2→D3→D4]
    → D-C（链哈希绑定）
    → D-F.update()（更新分布统计，非阻塞）

完整检索路径
-----------
    D-C（链哈希验签）
    → [委托 MINJA: D4验签→D5]
    → D-D（时序衰减重排序）
    → D-E（检索对齐核查）
    → D-F.scan()（分布健康扫描）

设计原则
--------
1. 与 MINJA 管线分层解耦：AutoWrite 仅处理新增防御逻辑，
   D1-D6 的原有逻辑完整保留，不修改 MINJA 代码
2. 任意节点 BLOCK → 操作终止，返回失败结果
3. 所有节点可独立通过 config 禁用，便于 A/B 消融实验
4. 审计日志统一写入 JSONL 文件（与 MINJA 审计日志路径可分开配置）
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ..MINJA.pipeline import MINJADefensePipeline, RetrievalResult, WriteResult
from ..MINJA.types import SourceLabel, ToolCallRequest, WriteContext
from .config import AutoWriteConfig
from .da_token_sanitizer import DAPreWriteSanitizer
from .db_selective_write import DBSelectiveWritePolicy, MemoryIndex
from .dc_integrity_chain import DCStorageIntegrityChain
from .dd_temporal_decay import DDTemporalDecayReranker
from .de_retrieval_align import DERetrievalAlignmentVerifier
from .df_distribution_monitor import DFMemoryDistributionMonitor
from .types import (
    CandidateEntry,
    ChainedCandidateEntry,
    ChainedRetrievedEntry,
    DefenseVerdict,
    RetrievedEntry,
)

EmbedFn = Callable[[list[str]], np.ndarray]
LLMClient = object


# ── 审计日志工具 ──────────────────────────────────────────────────────────────

def _audit(path: Optional[str], event: dict) -> None:
    """将审计事件写入 JSONL 文件，同时输出到 stderr。"""
    line = json.dumps(event, ensure_ascii=False, default=str)
    print(f"[AutoWrite-AUDIT] {line}", file=sys.stderr)
    if path:
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass


# ── 管线主类 ──────────────────────────────────────────────────────────────────

class AutoWriteDefensePipeline:
    """
    类型一（框架自动写入）完整纵深防御管线。

    推荐通过 from_config() 工厂方法构建，无需手动实例化各节点。

    示例
    ----
        from openai import OpenAI
        import numpy as np

        client = OpenAI(base_url="...", api_key="...")

        def embed(texts):
            resp = client.embeddings.create(
                model="text-embedding-3-small", input=texts
            )
            return np.array([d.embedding for d in resp.data])

        cfg = AutoWriteConfig()
        pipeline = AutoWriteDefensePipeline.from_config(
            cfg,
            minja_pipeline=minja,   # 已构建的 MINJADefensePipeline
            embed_fn=embed,
            llm_client=client,
            memory_index=my_index,
        )
    """

    def __init__(
        self,
        config: AutoWriteConfig,
        minja: MINJADefensePipeline,
        da: DAPreWriteSanitizer,
        db: DBSelectiveWritePolicy,
        dc: DCStorageIntegrityChain,
        dd: DDTemporalDecayReranker,
        de: DERetrievalAlignmentVerifier,
        df: DFMemoryDistributionMonitor,
    ):
        self.cfg = config
        self._minja = minja
        self.da = da
        self.db = db
        self.dc = dc
        self.dd = dd
        self.de = de
        self.df = df

    @classmethod
    def from_config(
        cls,
        config: AutoWriteConfig,
        minja_pipeline: MINJADefensePipeline,
        embed_fn: EmbedFn,
        memory_index: MemoryIndex,
        llm_client: Optional[LLMClient] = None,
        d1_check_fn: Optional[Callable[[str], float]] = None,
    ) -> "AutoWriteDefensePipeline":
        """
        工厂方法：从 AutoWriteConfig 一键构建完整管线。

        参数
        ----
        minja_pipeline : 已构建的 MINJADefensePipeline（D1-D6）
        embed_fn       : 嵌入函数，供 D-A/D-B/D-E/D-F 使用
        memory_index   : 只读记忆库索引，供 D-B 计算新颖度
        llm_client     : LLM 客户端，供 D-E dual_channel 使用（可选）
        d1_check_fn    : D1 风险评估函数，供 D-B 计算 risk 分量（可选）
        """
        return cls(
            config=config,
            minja=minja_pipeline,
            da=DAPreWriteSanitizer(config.da, embed_fn),
            db=DBSelectiveWritePolicy(config.db, embed_fn, memory_index, d1_check_fn),
            dc=DCStorageIntegrityChain(config.dc),
            dd=DDTemporalDecayReranker(config.dd),
            de=DERetrievalAlignmentVerifier(config.de, embed_fn, llm_client),
            df=DFMemoryDistributionMonitor(config.df, embed_fn),
        )

    # ── 写入路径 ──────────────────────────────────────────────────────────────

    def on_write_request(
        self,
        entry: CandidateEntry,
        write_ctx: WriteContext,
        source_labels: list[SourceLabel],
    ) -> WriteResult:
        """
        扩展写入路径：D-A → D-B → [MINJA D1-D4] → D-C → D-F.update()

        任意节点 BLOCK 时立即返回 WriteResult(accepted=False)。
        D-C 和 D-F 为写入后处理，不阻塞写入判决。

        返回
        ----
        WriteResult.accepted=True  : 写入被接受，entry.chain_hash 已填入
        WriteResult.accepted=False : 写入被拦截，blocked_by 指明拦截节点
        """
        all_verdicts: list[DefenseVerdict] = []
        t0 = time.time()

        # ── D-A：token 净化（原地修改 entry.content）─────────────────────────
        v_da = self.da.sanitize(entry)
        all_verdicts.append(v_da)
        if v_da.action == "BLOCK":
            _audit(self.cfg.audit_log_path, {
                "event": "write_blocked", "node": "D-A",
                "entry_id": entry.entry_id, "elapsed_s": round(time.time() - t0, 4),
            })
            return WriteResult(
                accepted=False, entry=None,
                verdicts=all_verdicts, blocked_by="D-A",
            )

        # ── D-B：选择性写入（信息价值过滤 + 洪水限速）───────────────────────
        v_db = self.db.check(entry, write_ctx)
        all_verdicts.append(v_db)
        if v_db.action == "BLOCK":
            _audit(self.cfg.audit_log_path, {
                "event": "write_blocked", "node": "D-B",
                "entry_id": entry.entry_id, "elapsed_s": round(time.time() - t0, 4),
            })
            return WriteResult(
                accepted=False, entry=None,
                verdicts=all_verdicts, blocked_by="D-B",
            )

        # ── MINJA D1-D4：委托原有管线执行意图检测、因果归因、仿真、溯源签名 ──
        base_result = self._minja.on_write_request(entry, write_ctx, source_labels)
        all_verdicts.extend(base_result.verdicts)

        if not base_result.accepted:
            # MINJA 内部节点（D1/D2/D3）拦截
            base_result.verdicts = all_verdicts
            return base_result

        # ── D-C：绑定链哈希（写入成功后，不影响写入判决）────────────────────
        chain_hash = self.dc.bind_chain(entry)
        # 将 chain_hash 写入 entry 扩展字段（持久化层需一同存储）
        if hasattr(entry, "chain_hash"):
            entry.chain_hash = chain_hash       # type: ignore[attr-defined]
            entry.prev_chain_hash = self.dc._head  # type: ignore[attr-defined]

        # ── D-F：更新分布统计（异步更新，不阻塞返回值）──────────────────────
        self.df.update(entry)

        elapsed = round(time.time() - t0, 4)
        _audit(self.cfg.audit_log_path, {
            "event": "write_accepted",
            "entry_id": entry.entry_id,
            "nodes_run": [v.node for v in all_verdicts],
            "elapsed_s": elapsed,
        })
        return WriteResult(accepted=True, entry=entry, verdicts=all_verdicts)

    # ── 检索路径 ──────────────────────────────────────────────────────────────

    def on_retrieval(
        self,
        retrieved: list[RetrievedEntry],
        user_task: str,
        user_task_embedding: Optional[list[float]] = None,
    ) -> RetrievalResult:
        """
        扩展检索路径：D-C验签 → [MINJA D4验签+D5] → D-D → D-E → D-F.scan()

        返回
        ----
        RetrievalResult.entries : 经过全部防御节点处理后的条目列表
                                   可疑条目已降权（weight < 1.0），不删除
        """
        all_verdicts: list[DefenseVerdict] = []
        tampered = 0
        t0 = time.time()

        # ── D-C：链哈希验签（过滤跨会话篡改条目）───────────────────────────
        verified: list[RetrievedEntry] = []
        prev_hash = self.dc.cfg.chain_anchor

        for entry in retrieved:
            stored_hash = getattr(entry, "chain_hash", "")
            prev = getattr(entry, "prev_chain_hash", prev_hash)

            # 临时封装为 CandidateEntry 进行验签（D-C 接口要求）
            candidate = CandidateEntry(
                entry_id=entry.entry_id,
                content=entry.content,
                provenance=entry.provenance,
            )
            v_dc = self.dc.verify_entry(candidate, stored_hash, prev)
            all_verdicts.append(v_dc)

            if v_dc.passed:
                prev_hash = stored_hash or prev_hash
                verified.append(entry)
            else:
                tampered += 1
                # 被篡改条目直接过滤，不进入后续检索

        # ── MINJA D4验签 + D5：委托原有检索管线 ─────────────────────────────
        task_emb = (
            np.asarray(user_task_embedding)
            if user_task_embedding is not None
            else None
        )
        if task_emb is None:
            # 需要嵌入但未提供时，从 D-E 的 embed_fn 计算
            task_emb = self.de._embed_fn([user_task])[0]

        base_result = self._minja.on_retrieval(verified, task_emb.tolist())
        all_verdicts.extend(base_result.verdicts)
        entries = base_result.entries

        # ── D-D：时序衰减重排序 ───────────────────────────────────────────────
        entries, v_dd = self.dd.rerank(entries)
        all_verdicts.append(v_dd)

        # ── D-E：检索对齐核查（双通道：嵌入 + LLM judge）────────────────────
        entries, de_verdicts = self.de.verify(entries, user_task, task_emb)
        all_verdicts.extend(de_verdicts)

        # ── D-F：分布健康扫描 ─────────────────────────────────────────────────
        v_df = self.df.scan()
        all_verdicts.append(v_df)

        if v_df.action == "BLOCK":
            # 分布严重异常（action_on_anomaly=BLOCK）→ 清空检索结果
            _audit(self.cfg.audit_log_path, {
                "event": "retrieval_blocked_by_df",
                "kl": v_df.metadata.get("kl"),
                "elapsed_s": round(time.time() - t0, 4),
            })
            return RetrievalResult(
                entries=[], verdicts=all_verdicts,
                tampered_count=tampered + len(entries),
            )

        _audit(self.cfg.audit_log_path, {
            "event": "retrieval_complete",
            "total_input": len(retrieved),
            "tampered": tampered,
            "flagged": sum(1 for e in entries if e.flagged),
            "elapsed_s": round(time.time() - t0, 4),
        })
        return RetrievalResult(
            entries=entries,
            verdicts=all_verdicts,
            tampered_count=tampered + base_result.tampered_count,
        )
