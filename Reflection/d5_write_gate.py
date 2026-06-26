"""
D5：最终写入闸门——综合 D1-D4 的信号做 accept / quarantine / reject 决策。
"""
from __future__ import annotations

from Reflection.config import D2Config, D5Config
from Reflection.types import DefenseVerdict, FactAssessment
from Reflection.utils import clamp


class D5WriteGate:
    """综合全局注入信号与逐事实证据，给出最终判决。"""

    def __init__(self, config: D5Config, grounding_config: D2Config) -> None:
        self.cfg = config
        self.grounding_cfg = grounding_config

    def check(self, assessment: FactAssessment) -> DefenseVerdict:
        if not self.cfg.enabled:
            return DefenseVerdict("D5", passed=True, score=0.0, reason="D5 已禁用", action="PASS")
        if self.cfg.strategy == "weighted_gate":
            return self._weighted_gate(assessment)
        if self.cfg.strategy == "strict_gate":
            return self._strict_gate(assessment)
        raise ValueError(f"未知 D5 strategy: {self.cfg.strategy}")

    def _weighted_gate(self, assessment: FactAssessment) -> DefenseVerdict:
        # D5 不重复发明规则，而是把前面节点的证据压缩成一个统一风险分数，
        # 这样做的好处是：后续做消融实验时只需要调权重，不必改节点逻辑。
        risk = self._risk(assessment)
        if assessment.contradiction_score >= 1.0:
            return DefenseVerdict(
                node="D5",
                passed=False,
                score=risk,
                reason="与已信任记忆发生强冲突，终止写入",
                action="BLOCK",
                metadata={"final_risk": risk},
            )

        if assessment.policy_score >= 1.0:
            return DefenseVerdict(
                node="D5",
                passed=False,
                score=risk,
                reason="命中存储策略硬约束，终止写入",
                action="BLOCK",
                metadata={"final_risk": risk},
            )

        if assessment.provenance_score < self.grounding_cfg.min_provenance_score:
            if risk >= self.cfg.rejection_threshold:
                return DefenseVerdict(
                    node="D5",
                    passed=False,
                    score=risk,
                    reason="证据接地性不足且综合风险过高，拒绝写入",
                    action="BLOCK",
                    metadata={"final_risk": risk},
                )
            return DefenseVerdict(
                node="D5",
                passed=False,
                score=risk,
                reason="证据接地性不足，隔离待审",
                action="FLAG",
                metadata={"final_risk": risk},
            )

        if risk >= self.cfg.rejection_threshold:
            return DefenseVerdict(
                node="D5",
                passed=False,
                score=risk,
                reason="综合风险超过拒绝阈值，终止写入",
                action="BLOCK",
                metadata={"final_risk": risk},
            )
        if risk >= self.cfg.quarantine_threshold:
            return DefenseVerdict(
                node="D5",
                passed=False,
                score=risk,
                reason="综合风险处于可疑区间，转入隔离区",
                action="FLAG",
                metadata={"final_risk": risk},
            )
        return DefenseVerdict(
            node="D5",
            passed=True,
            score=risk,
                reason="综合风险可接受，允许写入",
                action="PASS",
                metadata={"final_risk": risk},
            )

    def _strict_gate(self, assessment: FactAssessment) -> DefenseVerdict:
        risk = self._risk(assessment)
        witness_count = len(set(assessment.evidence_turn_ids))

        if assessment.contradiction_score >= 1.0 or assessment.policy_score >= 1.0:
            return DefenseVerdict(
                node="D5",
                passed=False,
                score=risk,
                reason="strict_gate 命中硬约束，终止写入",
                action="BLOCK",
                metadata={"final_risk": risk},
            )
        if assessment.injection_score >= 0.35 and assessment.provenance_score < 0.75:
            return DefenseVerdict(
                node="D5",
                passed=False,
                score=risk,
                reason="strict_gate 下，注入信号偏高且接地性不够，拒绝写入",
                action="BLOCK",
                metadata={"final_risk": risk},
            )
        if witness_count < self.cfg.min_accept_witnesses and assessment.provenance_score < 0.85:
            return DefenseVerdict(
                node="D5",
                passed=False,
                score=risk,
                reason="strict_gate 下，独立证据见证不足，转入隔离区",
                action="FLAG",
                metadata={"final_risk": risk},
            )
        if risk >= self.cfg.quarantine_threshold:
            return DefenseVerdict(
                node="D5",
                passed=False,
                score=risk,
                reason="strict_gate 下风险仍偏高，转入隔离区",
                action="FLAG",
                metadata={"final_risk": risk},
            )
        return DefenseVerdict(
            node="D5",
            passed=True,
            score=risk,
            reason="strict_gate 核查通过，允许写入",
            action="PASS",
            metadata={"final_risk": risk},
        )

    def _risk(self, assessment: FactAssessment) -> float:
        return clamp(
            (self.cfg.injection_weight * assessment.injection_score)
            + (self.cfg.provenance_weight * (1.0 - assessment.provenance_score))
            + (self.cfg.contradiction_weight * assessment.contradiction_score)
            + (self.cfg.policy_weight * assessment.policy_score)
        )
