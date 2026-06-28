"""
D-C：存储完整性链（Storage Integrity Chain）。

防御目标
--------
检测并阻断跨会话记忆篡改（offline tamper）。
攻击者可能在 Agent 下线期间直接操作持久化存储（SQLite、JSON 文件等），
悄无声息地修改或注入历史记忆条目。

防御原理
--------
在 D4 单条签名基础上，引入链式 HMAC 结构：

    chain_hash_0 = GENESIS（常量锚点）
    chain_hash_i = HMAC_K(
        entry_id_i  || content_hash_i  || label_i  ||
        query_hash_i || write_time_i   || chain_hash_{i-1}
    )

安全性（归谬法）：
  假设攻击者篡改第 j 条记忆 → content_hash_j 变化
  → chain_hash_j 变化（HMAC 的伪随机性，攻击者不知密钥 K）
  → 第 j+1 条及以后所有链哈希验签失败
  → 篡改范围可精确定位到第一条失败条目

与 D4 的分工
-----------
  D4  → 单条签名：保证单条内容 + 来源标签的完整性
  D-C → 链式哈希：保证跨会话历史序列的顺序不可篡改

整合位置
--------
  写入时：D4.bind() 之后调用 bind_chain()，将 chain_hash 存入条目扩展字段
  检索时：D4.verify() 之后调用 verify_entry()，发现篡改则 BLOCK
"""
from __future__ import annotations

import base64
import hashlib
import hmac as _hmac_lib
import json
import os
import secrets
from typing import Optional

from .config import DCConfig
from .types import CandidateEntry, ChainedCandidateEntry, DefenseVerdict


# ── 链式载荷构造 ──────────────────────────────────────────────────────────────

def _build_chain_payload(
    entry_id: str,
    content_hash: str,
    label: int,
    query_hash: str,
    write_time: float,
    prev_chain_hash: str,
) -> bytes:
    """
    构造链式 HMAC 的签名载荷（确定性 JSON 序列化）。

    prev_chain_hash 将前序条目链接进来，任何历史篡改都会破坏后续哈希。
    使用 sort_keys + 无空格分隔符保证跨平台序列化一致性。
    """
    payload = {
        "entry_id":   entry_id,
        "chash":      content_hash,
        "label":      label,
        "qhash":      query_hash,
        "wtime":      round(write_time, 3),
        "prev":       prev_chain_hash,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def compute_chain_hash(
    entry: CandidateEntry,
    prev_chain_hash: str,
    hmac_key: bytes,
) -> str:
    """
    计算当前条目的链式 HMAC 哈希，返回 URL-safe base64 编码字符串。

    entry.provenance 必须已由 D4.bind() 填充，否则抛出 ValueError。
    """
    if entry.provenance is None:
        raise ValueError(
            "compute_chain_hash() 要求 entry.provenance 已绑定（需先调用 D4.bind()）"
        )
    prov = entry.provenance
    payload = _build_chain_payload(
        entry.entry_id,
        entry.content_hash(),
        int(prov.label),
        prov.triggering_query_hash,
        prov.write_time,
        prev_chain_hash,
    )
    mac = _hmac_lib.new(hmac_key, payload, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(mac).decode()


# ── D-C 存储完整性链节点主类 ──────────────────────────────────────────────────

class DCStorageIntegrityChain:
    """
    D-C 存储完整性链节点。

    密钥管理
    --------
    优先从环境变量（DCConfig.hmac_key_env）读取 hex 编码的 32 字节密钥。
    未设置时自动生成随机密钥并打印警告——此时重启后无法验证历史签名，
    生产环境务必持久化密钥。

    用法
    ----
        chain = DCStorageIntegrityChain(DCConfig())
        # 写入时（D4.bind() 之后）
        chain_hash = chain.bind_chain(entry)
        entry.chain_hash = chain_hash        # 持久化到存储层

        # 检索时
        verdict = chain.verify_entry(entry, stored_hash, prev_hash)
        if verdict.action == "BLOCK":
            # 过滤被篡改条目
    """

    def __init__(self, config: DCConfig):
        self.cfg = config
        self._key: bytes = self._load_key()
        # 运行时维护链头：最近一次成功写入的 chain_hash
        self._head: str = config.chain_anchor

    def _load_key(self) -> bytes:
        """从环境变量加载 HMAC 密钥，未设置时生成随机密钥并警告。"""
        raw = os.environ.get(self.cfg.hmac_key_env, "")
        if raw:
            try:
                return bytes.fromhex(raw)
            except ValueError:
                pass  # 格式错误，降级到随机生成
        import sys
        key = secrets.token_bytes(32)
        print(
            f"[D-C] 警告：未找到环境变量 {self.cfg.hmac_key_env}，"
            f"已生成随机 HMAC 密钥。重启后历史签名将无法验证。\n"
            f"      请将以下值写入环境变量以持久化密钥：\n"
            f"      {self.cfg.hmac_key_env}={key.hex()}",
            file=sys.stderr,
        )
        return key

    # ── 写入时：绑定链哈希 ────────────────────────────────────────────────────

    def bind_chain(self, entry: CandidateEntry) -> str:
        """
        为新写入条目计算链式哈希并推进链头。

        调用方需将返回的 chain_hash 和当前 self._head（作为 prev_chain_hash）
        一同持久化到存储层的扩展字段，供日后验签回溯。

        Returns
        -------
        chain_hash : 当前条目的链式哈希字符串
        """
        if not self.cfg.enabled:
            return self.cfg.chain_anchor

        chain_hash = compute_chain_hash(entry, self._head, self._key)
        # 推进链头，下一条写入将以此为 prev_chain_hash
        self._head = chain_hash
        return chain_hash

    # ── 检索时：验证链哈希 ────────────────────────────────────────────────────

    def verify_entry(
        self,
        entry: CandidateEntry,
        stored_chain_hash: str,   # 从持久化存储读取的期望值
        prev_chain_hash: str,     # 前序条目的 chain_hash（链式验证必需）
    ) -> DefenseVerdict:
        """
        验证单条条目的链式完整性。

        重新计算 HMAC 并与 stored_chain_hash 比对。
        任何内容修改、溯源标签篡改或顺序调换都会导致验签失败。

        判决语义
        --------
          PASS  : 哈希匹配，条目未被篡改
          BLOCK : 哈希不匹配，条目内容或历史顺序已被篡改
        """
        if not self.cfg.enabled:
            return DefenseVerdict(
                "D-C", passed=True, score=1.0,
                reason="D-C 已禁用", action="PASS",
            )

        if not stored_chain_hash:
            # 旧条目（D-C 部署前写入）无链哈希，跳过验证
            return DefenseVerdict(
                "D-C", passed=True, score=1.0,
                reason="条目无链哈希（D-C 部署前写入），跳过链验证",
                action="PASS",
                metadata={"entry_id": entry.entry_id[:8]},
            )

        computed = compute_chain_hash(entry, prev_chain_hash, self._key)
        if _hmac_lib.compare_digest(computed, stored_chain_hash):
            return DefenseVerdict(
                node="D-C", passed=True, score=1.0,
                reason=f"链式完整性验证通过 (entry={entry.entry_id[:8]}…)",
                action="PASS",
                metadata={
                    "entry_id": entry.entry_id[:8],
                    "chain_hash_prefix": computed[:16],
                },
            )

        return DefenseVerdict(
            node="D-C", passed=False, score=0.0,
            reason=(
                f"链式哈希不匹配，条目已被篡改或历史顺序被破坏 "
                f"(entry={entry.entry_id[:8]}…, "
                f"expected={stored_chain_hash[:16]}…, "
                f"computed={computed[:16]}…)"
            ),
            action="BLOCK",
            metadata={
                "entry_id": entry.entry_id,
                "expected_prefix": stored_chain_hash[:32],
                "computed_prefix": computed[:32],
            },
        )

    def verify_chain_segment(
        self,
        entries: list[tuple[CandidateEntry, str, str]],
        # 每个元素：(entry, stored_chain_hash, prev_chain_hash)，按写入顺序排列
    ) -> list[DefenseVerdict]:
        """
        批量验证一段历史链（增量验签，通常用于启动时完整性扫描）。

        发现失败条目后继续扫描（不 break），汇报所有被篡改位置，
        便于取证和审计。
        """
        return [
            self.verify_entry(entry, stored_hash, prev_hash)
            for entry, stored_hash, prev_hash in entries
        ]
