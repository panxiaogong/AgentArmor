"""
Reflection provenance 绑定与签名验证。

目标：
1. 让每条被接受的反思记忆都带着可追溯来源标签；
2. 让未来检索阶段能够验证“这条记忆是不是被篡改过”；
3. 把写入时的风险、证据来源、判决链保留下来，便于事后取证。
"""
from __future__ import annotations

import base64
import hashlib
import hmac as _hmac
import json
import os
import time
from typing import List, Optional

from Reflection.config import ProvenanceConfig
from Reflection.types import (
    FactAssessment,
    IntegrityLabel,
    MemoryRecord,
    ProvenanceTag,
    ReflectionContext,
    SourceLabel,
    lattice_join,
)
from Reflection.utils import clamp


def _hmac_sign(payload: bytes, secret: bytes) -> str:
    mac = _hmac.new(secret, payload, hashlib.sha256).digest()
    return base64.b64encode(mac).decode("ascii")


def _hmac_verify(payload: bytes, signature_b64: str, secret: bytes) -> bool:
    expected = _hmac.new(secret, payload, hashlib.sha256).digest()
    try:
        actual = base64.b64decode(signature_b64)
        return _hmac.compare_digest(expected, actual)
    except Exception:
        return False


def _ed25519_sign(payload: bytes, private_key_b64: str) -> str:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    raw = base64.b64decode(private_key_b64)
    key = Ed25519PrivateKey.from_private_bytes(raw)
    signature = key.sign(payload)
    return base64.b64encode(signature).decode("ascii")


def _ed25519_verify(payload: bytes, signature_b64: str, public_key_b64: str) -> bool:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    try:
        raw = base64.b64decode(public_key_b64)
        key = Ed25519PublicKey.from_public_bytes(raw)
        signature = base64.b64decode(signature_b64)
        key.verify(signature, payload)
        return True
    except (InvalidSignature, Exception):
        return False


def _generate_ed25519_keypair() -> tuple[str, str]:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    private_b64 = base64.b64encode(
        private_key.private_bytes(
            serialization.Encoding.Raw,
            serialization.PrivateFormat.Raw,
            serialization.NoEncryption(),
        )
    ).decode("ascii")
    public_b64 = base64.b64encode(
        public_key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    ).decode("ascii")
    return private_b64, public_b64


def build_provenance_payload(
    record: MemoryRecord,
    label: IntegrityLabel,
    triggering_query_hash: str,
    write_time: float,
    evidence_turn_ids: List[str],
) -> bytes:
    payload = {
        "record_id": record.record_id,
        "fact_hash": record.content_hash(),
        "summary_hash": record.summary_hash(),
        "integrity_label": int(label),
        "triggering_query_hash": triggering_query_hash,
        "write_time": round(write_time, 3),
        "evidence_turn_ids": list(evidence_turn_ids),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


class ProvenanceBinder:
    """为被接受的反思记忆绑定签名 provenance，并支持未来检索验签。"""

    def __init__(self, config: ProvenanceConfig) -> None:
        self.cfg = config
        self._private_key_b64: Optional[str] = None
        self._public_key_b64: Optional[str] = None
        self._hmac_secret: Optional[bytes] = None
        self._init_keys()

    def _init_keys(self) -> None:
        if not self.cfg.enabled:
            return
        if self.cfg.signing_backend == "hmac":
            self._hmac_secret = self.cfg.hmac_secret or os.urandom(32)
            return

        if self.cfg.signing_backend == "ed25519":
            if self.cfg.private_key_b64:
                self._private_key_b64 = self.cfg.private_key_b64
                self._public_key_b64 = self._derive_public_key(self.cfg.private_key_b64)
            else:
                self._private_key_b64, self._public_key_b64 = _generate_ed25519_keypair()
            return

        raise ValueError(f"未知 provenance signing backend: {self.cfg.signing_backend}")

    @staticmethod
    def _derive_public_key(private_key_b64: str) -> str:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        raw = base64.b64decode(private_key_b64)
        private_key = Ed25519PrivateKey.from_private_bytes(raw)
        public_key = private_key.public_key()
        return base64.b64encode(
            public_key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        ).decode("ascii")

    @property
    def public_key_b64(self) -> Optional[str]:
        return self._public_key_b64

    def bind(
        self,
        record: MemoryRecord,
        assessment: FactAssessment,
        context: ReflectionContext,
    ) -> MemoryRecord:
        if not self.cfg.enabled:
            return record

        label = self._compute_label(assessment)
        query_hash = hashlib.sha256(context.triggering_query.encode("utf-8")).hexdigest()
        write_time = time.time()
        payload = build_provenance_payload(
            record=record,
            label=label,
            triggering_query_hash=query_hash,
            write_time=write_time,
            evidence_turn_ids=assessment.evidence_turn_ids,
        )

        if self.cfg.signing_backend == "hmac":
            signature = _hmac_sign(payload, self._hmac_secret)
        elif self.cfg.signing_backend == "ed25519":
            signature = _ed25519_sign(payload, self._private_key_b64)
        else:
            raise ValueError(f"未知 provenance signing backend: {self.cfg.signing_backend}")

        record.provenance = ProvenanceTag(
            label=label,
            source_types=[label_item.source_type for label_item in assessment.source_labels],
            evidence_turn_ids=list(assessment.evidence_turn_ids),
            triggering_query_hash=query_hash,
            summary_hash=record.summary_hash(),
            write_time=write_time,
            signature=signature,
            sign_algo=self.cfg.signing_backend,
            risk_at_write=assessment.final_risk,
            verdict_trace=[f"{verdict.node}:{verdict.action}" for verdict in assessment.verdicts],
        )
        return record

    def verify(self, record: MemoryRecord) -> tuple[bool, str]:
        if not self.cfg.enabled:
            return True, "provenance 已禁用"
        if record.provenance is None:
            return False, "记忆条目缺少 provenance 标签"

        provenance = record.provenance
        payload = build_provenance_payload(
            record=record,
            label=provenance.label,
            triggering_query_hash=provenance.triggering_query_hash,
            write_time=provenance.write_time,
            evidence_turn_ids=provenance.evidence_turn_ids,
        )

        if provenance.sign_algo == "hmac":
            if self._hmac_secret is None:
                return False, "HMAC 密钥缺失，无法验签"
            valid = _hmac_verify(payload, provenance.signature, self._hmac_secret)
        elif provenance.sign_algo == "ed25519":
            if self._public_key_b64 is None:
                return False, "Ed25519 公钥缺失，无法验签"
            valid = _ed25519_verify(payload, provenance.signature, self._public_key_b64)
        else:
            return False, f"未知签名算法: {provenance.sign_algo}"

        if valid:
            return True, f"签名验证通过 ({provenance.sign_algo}, {provenance.label.name})"
        return False, "签名验证失败：条目正文或 provenance 标签被篡改"

    def _compute_label(self, assessment: FactAssessment) -> IntegrityLabel:
        if not assessment.source_labels:
            return IntegrityLabel.UNTRUSTED

        label = lattice_join([item.label for item in assessment.source_labels])
        if self.cfg.integrity_mode == "source_lattice":
            return label
        if self.cfg.integrity_mode != "risk_aware":
            raise ValueError(f"未知 provenance integrity mode: {self.cfg.integrity_mode}")

        # risk_aware 模式下，如果写入时已经有明显注入/接地性风险，就主动降一档完整性。
        downgrade = 0
        if assessment.injection_score >= 0.35:
            downgrade += 1
        if assessment.provenance_score < 0.55:
            downgrade += 1
        if assessment.final_risk >= 0.35:
            downgrade += 1
        adjusted = int(label) - min(downgrade, 2)
        return IntegrityLabel(int(clamp(adjusted, IntegrityLabel.UNTRUSTED, IntegrityLabel.TRUSTED)))

