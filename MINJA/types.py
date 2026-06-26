"""
MINJA 防御体系共享数据结构。

所有防御节点（D1-D6）和主管线共用的类型定义集中在此文件。
原则：数据结构只描述状态，不包含业务逻辑。
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Literal, Optional


# ── IFC 完整性标签（格序结构）────────────────────────────────────────────────

class IntegrityLabel(IntEnum):
    """
    信息流控制（IFC）完整性格（Lattice）。

    格序：TRUSTED > CANDIDATE > UNTRUSTED
    传播规则（Lattice Join）：label(m) = min(所有信息源的标签)
    含义：只要任意一个来源是 UNTRUSTED，写入内容就继承最低完整性。
    这直接对应 MINJA 的核心缺陷：攻击者查询污染了 Agent 写入的内容。
    """
    UNTRUSTED = 1   # 来自外部输入、工具返回、攻击者查询
    CANDIDATE = 2   # 来自用户输入，已过基础过滤，未深度核查
    TRUSTED   = 3   # 来自系统内部生成，完全可信


def lattice_join(labels: list[IntegrityLabel]) -> IntegrityLabel:
    """格下确界：取所有来源标签中的最小值（最低完整性传染）。"""
    if not labels:
        return IntegrityLabel.UNTRUSTED
    return IntegrityLabel(min(int(l) for l in labels))


# ── 防御判决 ──────────────────────────────────────────────────────────────────

@dataclass
class DefenseVerdict:
    """
    单个防御节点的判决结果。

    节点只产出判决，不执行动作。
    主管线负责根据判决序列决定最终行为。

    action 语义：
      PASS  = 通过，继续下一节点
      FLAG  = 软告警，升级到更精确的后续节点核查
      BLOCK = 硬拦截，直接中止操作
      ASK   = 暂停等待人工确认（用于高代价操作的保守处理）
    """
    node: str
    passed: bool
    score: float
    reason: str
    action: Literal["PASS", "FLAG", "BLOCK", "ASK"]
    metadata: dict[str, Any] = field(default_factory=dict)


# ── 溯源标签 ──────────────────────────────────────────────────────────────────

@dataclass
class SourceLabel:
    """单个信息源的类型与完整性标签对，供 D4 做格 Join 使用。"""
    source_type: str        # e.g. "user_input", "tool_output", "system_internal"
    label: IntegrityLabel


@dataclass
class ProvenanceTag:
    """
    绑定到每条记忆条目的不可伪造溯源标签（D4 产出）。

    字段设计遵循 CaMeL 双 LLM 架构（arXiv:2503.18813）的信息流追踪思路：
    触发写入的查询哈希是因果溯源的核心字段，使攻击者无法伪造
    "此写入是由用户任务自然驱动的"这一假象。
    """
    label: IntegrityLabel
    triggering_query_hash: str  # SHA-256(triggering_query)，因果记录
    source_types: list[str]     # 写入时涉及的所有信息源类型
    write_time: float           # Unix 时间戳
    signature: str              # base64 编码的密码学签名
    sign_algo: str              # "ed25519" 或 "hmac"（对应 D4Config.signing_backend）


# ── 候选记忆条目 ──────────────────────────────────────────────────────────────

@dataclass
class CandidateEntry:
    """
    待写入记忆库的候选条目，贯穿写入路径（D1→D2→D3→D4）。
    D4 绑定 ProvenanceTag 后成为带溯源的完整记录。
    """
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    embedding: Optional[list[float]] = None   # 延迟计算，首次需要时填入
    provenance: Optional[ProvenanceTag] = None
    write_time: float = field(default_factory=time.time)

    def content_hash(self) -> str:
        """SHA-256(content)，用于签名载荷和完整性校验。"""
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()

    def sign_payload(self) -> bytes:
        """
        构造签名载荷：content_hash + label + triggering_query_hash。
        与 MemGuard 现有 MemoryEntry._signable_payload() 设计对齐，
        并额外将触发查询哈希纳入签名范围（MINJA 防御关键扩展）。
        """
        if self.provenance is None:
            raise ValueError("sign_payload() 需先绑定 ProvenanceTag")
        payload = {
            "content_hash": self.content_hash(),
            "integrity_label": int(self.provenance.label),
            "triggering_query_hash": self.provenance.triggering_query_hash,
            "write_time": self.provenance.write_time,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


# ── 写入上下文（D2 因果归因所需）─────────────────────────────────────────────

@dataclass
class WriteContext:
    """
    Agent 写入一条记忆时所有信息源的打包。
    D2 需要这些来计算每个来源对写入决策的因果贡献。
    """
    user_goal: str            # 用户的原始任务描述（写入行为应服务于此）
    current_context: str      # 当前对话/工具返回的拼接文本
    indication_prompt: str    # Q_inject 中的 indication 部分（Progressive Shortening 的载体）
    candidate_content: str    # Agent 准备写入的内容 m
    triggering_query: str     # 触发本次写入的原始查询（D4 溯源用）


# ── 工具调用请求（D6 核查对象）───────────────────────────────────────────────

@dataclass
class ToolCallRequest:
    """D6 需要核查的工具调用请求。"""
    tool_name: str
    params: dict[str, Any]
    user_original_task: str   # 用户原始任务，D6 的对比基准
    conversation_ctx: str     # 当前对话上下文（辅助 llm_judge 模式）

    def verbalize(self) -> str:
        """序列化为自然语言意图描述，用于嵌入相似度计算。"""
        params_str = json.dumps(self.params, ensure_ascii=False, indent=None)
        return f"调用工具 {self.tool_name}，参数：{params_str}"


# ── 检索条目（D5 处理对象）───────────────────────────────────────────────────

@dataclass
class RetrievedEntry:
    """从记忆库检索出的单条条目，D5 对其进行集合级安全验证。"""
    entry_id: str
    content: str
    embedding: list[float]
    provenance: Optional[ProvenanceTag] = None
    weight: float = 1.0    # D5 可降权（初始 1.0），降权后不删除条目
    flagged: bool = False  # D5 标记为可疑
    flag_reason: str = ""
