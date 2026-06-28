"""
AutoWrite 防御体系扩展类型定义。

本模块在 MINJA 共享类型基础上，为 AutoWrite（类型一：框架自动写入）
防御体系新增以下数据结构：
  - ChainedCandidateEntry : 带链式哈希字段的候选条目（D-C 使用）
  - ChainedRetrievedEntry : 带链式哈希字段的检索条目（D-C 验签使用）

其余类型（DefenseVerdict、WriteContext、ProvenanceTag 等）直接从
MINJA.types 导入，保持类型系统统一，避免重复定义。
"""
from __future__ import annotations

from dataclasses import dataclass, field

# 从 MINJA 导入所有共享类型；AutoWrite 各节点直接使用这些类型，
# 保持两个子包的类型系统一致，防止隐式转换错误。
from ..MINJA.types import (
    CandidateEntry,
    DefenseVerdict,
    IntegrityLabel,
    ProvenanceTag,
    RetrievedEntry,
    SourceLabel,
    ToolCallRequest,
    WriteContext,
    lattice_join,
)

__all__ = [
    # 透传 MINJA 类型，方便上层代码统一从本包导入
    "CandidateEntry",
    "DefenseVerdict",
    "IntegrityLabel",
    "ProvenanceTag",
    "RetrievedEntry",
    "SourceLabel",
    "ToolCallRequest",
    "WriteContext",
    "lattice_join",
    # AutoWrite 新增类型
    "ChainedCandidateEntry",
    "ChainedRetrievedEntry",
]


@dataclass
class ChainedCandidateEntry(CandidateEntry):
    """
    带链式哈希字段的候选条目。

    D-C（存储完整性链）在写入成功后将本条目的链式哈希写入
    chain_hash 字段，并要求持久化层一同存储。
    prev_chain_hash 记录写入时的前序链头，供日后验签回溯。
    """
    # D-C 绑定后填入；空字符串表示 D-C 未启用或尚未绑定
    chain_hash: str = ""
    # 写入时的前序哈希（验签链式结构必需）
    prev_chain_hash: str = ""


@dataclass
class ChainedRetrievedEntry(RetrievedEntry):
    """
    带链式哈希字段的检索条目。

    从持久化存储读取时需同步读入 chain_hash 和 prev_chain_hash，
    D-C 的 verify_entry() 方法将据此重算哈希并比对。
    """
    chain_hash: str = ""
    prev_chain_hash: str = ""
