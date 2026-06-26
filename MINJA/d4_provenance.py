"""
D4：存储层——溯源标签绑定与验证。

防御目标：为每条写入记忆库的条目绑定不可伪造的 IFC 溯源标签，
支持事后审计与取证。不实时拦截，而是建立"因果记录链"。

理论依据：
  IFC 格模型（Denning 1976）+ CaMeL 双 LLM 架构（arXiv:2503.18813）
  label(m) = lattice_join(所有信息源的标签) = min(labels)

关键设计：
  将 triggering_query_hash 纳入签名载荷（MINJA 防御的核心扩展）。
  若攻击者伪造 source_type=USER_INPUT 但实际来自工具返回/外部查询，
  签名验证会暴露矛盾，触发告警。

策略切换：修改 D4Config.signing_backend
  "ed25519" → 非对称签名，可公开验证，需密钥管理
  "hmac"    → 对称签名，部署简单，无法第三方验证
"""
from __future__ import annotations

import base64
import hashlib
import hmac as _hmac
import json
import os
import time
from typing import Optional

from .config import D4Config
from .types import (
    CandidateEntry,
    IntegrityLabel,
    ProvenanceTag,
    SourceLabel,
    lattice_join,
)


# ── Ed25519 签名后端 ──────────────────────────────────────────────────────────

def _ed25519_sign(payload: bytes, private_key_b64: str) -> str:
    """用 Ed25519 私钥签名，返回 base64 编码的签名字符串。"""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    raw = base64.b64decode(private_key_b64)
    key = Ed25519PrivateKey.from_private_bytes(raw)
    sig = key.sign(payload)
    return base64.b64encode(sig).decode("ascii")


def _ed25519_verify(payload: bytes, signature_b64: str, public_key_b64: str) -> bool:
    """验证 Ed25519 签名，返回 True 表示签名有效。"""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    try:
        raw = base64.b64decode(public_key_b64)
        key = Ed25519PublicKey.from_public_bytes(raw)
        sig = base64.b64decode(signature_b64)
        key.verify(sig, payload)
        return True
    except (InvalidSignature, Exception):
        return False


def _generate_ed25519_keypair() -> tuple[str, str]:
    """生成 Ed25519 密钥对，返回 (private_b64, public_b64)。"""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    priv = Ed25519PrivateKey.generate()
    pub = priv.public_key()
    priv_b64 = base64.b64encode(
        priv.private_bytes(
            serialization.Encoding.Raw,
            serialization.PrivateFormat.Raw,
            serialization.NoEncryption(),
        )
    ).decode("ascii")
    pub_b64 = base64.b64encode(
        pub.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    ).decode("ascii")
    return priv_b64, pub_b64


# ── HMAC 签名后端 ─────────────────────────────────────────────────────────────

def _hmac_sign(payload: bytes, secret: bytes) -> str:
    """用 HMAC-SHA256 签名，返回 base64 编码的 MAC。"""
    mac = _hmac.new(secret, payload, hashlib.sha256).digest()
    return base64.b64encode(mac).decode("ascii")


def _hmac_verify(payload: bytes, signature_b64: str, secret: bytes) -> bool:
    """验证 HMAC-SHA256 签名，使用常量时间比较防止时序攻击。"""
    expected = _hmac.new(secret, payload, hashlib.sha256).digest()
    try:
        actual = base64.b64decode(signature_b64)
        return _hmac.compare_digest(expected, actual)
    except Exception:
        return False


# ── 签名载荷构造 ──────────────────────────────────────────────────────────────

def build_sign_payload(
    content_hash: str,
    label: IntegrityLabel,
    triggering_query_hash: str,
    write_time: float,
) -> bytes:
    """
    构造签名载荷。

    纳入 triggering_query_hash 是 MINJA 防御相对于 MemGuard 原有签名的关键扩展：
    原有签名只覆盖内容本身，无法证明"是谁触发了这次写入"。
    加入触发查询哈希后，攻击者无法在不破坏签名的前提下伪造写入来源。
    """
    payload = {
        "content_hash": content_hash,
        "integrity_label": int(label),
        "triggering_query_hash": triggering_query_hash,
        "write_time": round(write_time, 3),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


# ── D4 溯源绑定器主类 ─────────────────────────────────────────────────────────

class D4ProvenanceBinder:
    """
    D4 IFC 溯源标签绑定器。

    职责：
    1. bind()   → 计算 IFC 标签、构造签名、返回带溯源的 CandidateEntry
    2. verify() → 验证已存储条目的签名完整性（检索路径调用）

    密钥管理：
    - Ed25519 模式：private_key_b64 用于签名，public_key_b64 用于验证
    - HMAC 模式：同一 secret 用于签名和验证
    - 若 config 中未提供密钥，自动生成并打印到 stderr（提示保存）
    """

    def __init__(self, config: D4Config):
        self.cfg = config
        self._private_key_b64: Optional[str] = None
        self._public_key_b64: Optional[str] = None
        self._hmac_secret: Optional[bytes] = None
        self._init_keys()

    def _init_keys(self) -> None:
        """初始化密钥，若未配置则自动生成。"""
        backend = self.cfg.signing_backend

        if backend == "ed25519":
            if self.cfg.private_key_b64:
                self._private_key_b64 = self.cfg.private_key_b64
                # 从私钥派生公钥
                priv_b64, pub_b64 = self._derive_public_key(self.cfg.private_key_b64)
                self._public_key_b64 = pub_b64
            else:
                priv_b64, pub_b64 = _generate_ed25519_keypair()
                self._private_key_b64 = priv_b64
                self._public_key_b64 = pub_b64
                import sys
                print(
                    f"[D4] 已自动生成 Ed25519 密钥对。请将以下私钥保存到配置文件：\n"
                    f"  D4Config.private_key_b64 = \"{priv_b64}\"",
                    file=sys.stderr,
                )

        elif backend == "hmac":
            if self.cfg.hmac_secret:
                self._hmac_secret = self.cfg.hmac_secret
            else:
                self._hmac_secret = os.urandom(32)
                import sys
                print(
                    f"[D4] 已自动生成 HMAC 密钥（32字节随机）。"
                    "重启后密钥会变化，届时历史签名将无法验证。",
                    file=sys.stderr,
                )

    @staticmethod
    def _derive_public_key(private_key_b64: str) -> tuple[str, str]:
        """从 Ed25519 私钥派生公钥，返回 (priv_b64, pub_b64)。"""
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        raw = base64.b64decode(private_key_b64)
        priv = Ed25519PrivateKey.from_private_bytes(raw)
        pub = priv.public_key()
        pub_b64 = base64.b64encode(
            pub.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        ).decode("ascii")
        return private_key_b64, pub_b64

    @property
    def public_key_b64(self) -> Optional[str]:
        """暴露公钥供外部（检索路径）进行验证。"""
        return self._public_key_b64

    # ── 核心方法：绑定溯源标签 ────────────────────────────────────────────────

    def bind(
        self,
        entry: CandidateEntry,
        source_labels: list[SourceLabel],
        triggering_query: str,
    ) -> CandidateEntry:
        """
        为候选条目绑定 IFC 溯源标签并签名。

        label = lattice_join(所有信息源的标签)  ← 最低完整性传染
        signature 覆盖 content_hash + label + triggering_query_hash

        修改 entry.provenance 字段并返回同一对象（in-place 更新）。
        """
        if not self.cfg.enabled:
            # D4 禁用时直接返回无溯源标签的条目
            return entry

        combined_label = lattice_join([sl.label for sl in source_labels])
        query_hash = hashlib.sha256(triggering_query.encode()).hexdigest()
        write_time = time.time()

        payload = build_sign_payload(
            entry.content_hash(), combined_label, query_hash, write_time
        )

        backend = self.cfg.signing_backend
        if backend == "ed25519":
            signature = _ed25519_sign(payload, self._private_key_b64)
        elif backend == "hmac":
            signature = _hmac_sign(payload, self._hmac_secret)
        else:
            raise ValueError(f"未知 D4 签名后端: {backend}")

        entry.provenance = ProvenanceTag(
            label=combined_label,
            triggering_query_hash=query_hash,
            source_types=[sl.source_type for sl in source_labels],
            write_time=write_time,
            signature=signature,
            sign_algo=backend,
        )
        return entry

    # ── 核心方法：验证溯源标签 ────────────────────────────────────────────────

    def verify(self, entry: CandidateEntry) -> tuple[bool, str]:
        """
        验证条目的溯源签名是否完整。

        返回 (is_valid, reason)。
        检索路径（D5 之前）调用此方法，过滤被篡改的条目。
        """
        if not self.cfg.enabled:
            return True, "D4 已禁用，跳过验证"

        if entry.provenance is None:
            return False, "条目缺少 ProvenanceTag，无法验证"

        prov = entry.provenance
        payload = build_sign_payload(
            entry.content_hash(),
            prov.label,
            prov.triggering_query_hash,
            prov.write_time,
        )

        backend = prov.sign_algo
        if backend == "ed25519":
            if self._public_key_b64 is None:
                return False, "Ed25519 公钥未加载，无法验证"
            valid = _ed25519_verify(payload, prov.signature, self._public_key_b64)
        elif backend == "hmac":
            if self._hmac_secret is None:
                return False, "HMAC 密钥未加载，无法验证"
            valid = _hmac_verify(payload, prov.signature, self._hmac_secret)
        else:
            return False, f"未知签名算法: {backend}"

        if valid:
            return True, f"签名验证通过 (algo={backend}, label={prov.label.name})"
        return False, "签名验证失败：条目内容或溯源标签已被篡改"
