"""
D2：接地性核查——反思候选事实是否真正得到可信原始证据支持。

类型三的核心不是“候选事实是否像一句话”，
而是“这句蒸馏事实是否可被原始 turns 支撑，且支撑来源是否值得信任”。
"""
from __future__ import annotations

from typing import List

from Reflection.config import D2Config
from Reflection.types import DefenseVerdict, FactCategory, ReflectionContext, SourceType
from Reflection.utils import clamp, lexical_support


class D2GroundingAuditor:
    """评估反思事实的证据覆盖度与来源可信度。"""

    SOURCE_WEIGHTS = {
        SourceType.USER: 1.00,
        SourceType.SYSTEM: 0.95,
        SourceType.ASSISTANT: 0.70,
        SourceType.MEMORY: 0.80,
        SourceType.TOOL: 0.45,
        SourceType.WEB: 0.30,
    }

    DURABLE_PERSONAL_CATEGORIES = {
        FactCategory.IDENTITY,
        FactCategory.PREFERENCE,
        FactCategory.CONTACT,
    }

    def __init__(self, config: D2Config) -> None:
        self.cfg = config

    def check(self, fact_text: str, category: FactCategory, ctx: ReflectionContext) -> DefenseVerdict:
        if not self.cfg.enabled:
            return DefenseVerdict("D2", passed=True, score=1.0, reason="D2 已禁用", action="PASS")
        if self.cfg.strategy not in {"lexical", "hybrid"}:
            raise ValueError(f"未知 D2 strategy: {self.cfg.strategy}")

        scored_support = []
        reasons: List[str] = []

        for turn in ctx.turns:
            # support 衡量“原始 turn 与反思事实在词面上有多少重合”，
            # weighted 则进一步把“来源是否可信”编码进去。
            support = lexical_support(fact_text, turn.text)
            if support <= 0.0:
                continue
            weighted = support * self.SOURCE_WEIGHTS[turn.source]
            scored_support.append((weighted, support, turn))

        if not scored_support:
            return DefenseVerdict(
                node="D2",
                passed=False,
                score=0.0,
                reason="未找到任何可支撑该反思事实的原始证据",
                action="FLAG",
                metadata={
                    "support_score": 0.0,
                    "provenance_score": 0.0,
                    "evidence_turn_ids": [],
                    "reasons": ["No supporting raw turn found for synthesized fact."],
                },
            )

        scored_support.sort(key=lambda item: item[0], reverse=True)
        best_weighted, best_support, _ = scored_support[0]
        provenance_score = best_weighted
        if len(scored_support) > 1:
            provenance_score += 0.15 * scored_support[1][0]
        provenance_score = clamp(provenance_score)

        # 只把“有足够词面支持”或“来源质量较高”的 turns 记作审计证据，
        # 避免 evidence_turn_ids 被弱相关文本灌水。
        evidence_turn_ids = [
            turn.turn_id
            for weighted, support, turn in scored_support
            if weighted >= 0.20 or support >= self.cfg.min_support_score
        ]

        if category in self.DURABLE_PERSONAL_CATEGORIES:
            has_user_backing = any(
                turn.source in {SourceType.USER, SourceType.SYSTEM}
                for _, _, turn in scored_support[:3]
            )
            if not has_user_backing:
                provenance_score *= 0.60
                reasons.append("Durable personal fact is backed only by low-trust sources such as web/tool content.")

        passed = provenance_score >= self.cfg.min_provenance_score
        score = provenance_score
        if passed:
            reason = f"事实接地性良好，provenance={provenance_score:.3f}"
            action = "PASS"
        else:
            reason = f"事实接地性不足，provenance={provenance_score:.3f}"
            action = "FLAG"

        return DefenseVerdict(
            node="D2",
            passed=passed,
            score=score,
            reason=reason,
            action=action,
            metadata={
                "support_score": clamp(best_support),
                "provenance_score": provenance_score,
                "evidence_turn_ids": evidence_turn_ids,
                "reasons": reasons,
            },
        )
