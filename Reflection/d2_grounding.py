"""
D2：接地性核查——反思候选事实是否真正得到可信原始证据支持。

类型三的关键不是“摘要像不像事实”，而是：
  这条蒸馏事实能否被原始 turns 中的具体证据句支撑，
  且这些证据句本身是否来自足够可信的来源。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

from Reflection.config import D2Config
from Reflection.types import (
    DefenseVerdict,
    FactCategory,
    ReflectionContext,
    SourceLabel,
    SourceType,
    default_integrity_for_source,
)
from Reflection.utils import (
    clamp,
    extract_canonical_value,
    extract_slot_key,
    lexical_support,
    split_sentences,
    tokenize,
    unique_tokens,
)


@dataclass
class EvidenceSentence:
    """句级证据候选。"""

    turn_id: str
    source: SourceType
    sentence: str
    lexical_score: float
    rarity_score: float
    slot_score: float
    selector_score: float


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
        if self.cfg.strategy == "lexical":
            return self._check_lexical(fact_text, category, ctx)
        if self.cfg.strategy in {"hybrid", "evidence_graph"}:
            return self._check_evidence_graph(fact_text, category, ctx)
        raise ValueError(f"未知 D2 strategy: {self.cfg.strategy}")

    def _check_lexical(
        self,
        fact_text: str,
        category: FactCategory,
        ctx: ReflectionContext,
    ) -> DefenseVerdict:
        """保留简单 lexical 模式，便于消融实验。"""

        scored_support = []
        reasons: List[str] = []
        for turn in ctx.turns:
            support = lexical_support(fact_text, turn.text)
            if support <= 0.0:
                continue
            weighted = support * self.SOURCE_WEIGHTS[turn.source]
            scored_support.append((weighted, support, turn))

        if not scored_support:
            return self._empty_verdict("未找到任何可支撑该反思事实的原始证据")

        scored_support.sort(key=lambda item: item[0], reverse=True)
        best_weighted, best_support, _ = scored_support[0]
        provenance_score = best_weighted
        if len(scored_support) > 1:
            provenance_score += 0.15 * scored_support[1][0]
        provenance_score = clamp(provenance_score)

        evidence_turn_ids = [
            turn.turn_id
            for weighted, support, turn in scored_support
            if weighted >= 0.20 or support >= self.cfg.min_support_score
        ]
        source_labels = [
            SourceLabel(
                turn_id=turn.turn_id,
                source_type=turn.source.value,
                label=default_integrity_for_source(turn.source),
                selector_score=weighted,
                excerpt=turn.text[:160],
            )
            for weighted, _, turn in scored_support[: self.cfg.max_evidence_sentences]
        ]
        evidence_snippets = [turn.text[:160] for _, _, turn in scored_support[: self.cfg.max_evidence_sentences]]

        if category in self.DURABLE_PERSONAL_CATEGORIES:
            has_user_backing = any(
                turn.source in {SourceType.USER, SourceType.SYSTEM}
                for _, _, turn in scored_support[:3]
            )
            if not has_user_backing:
                provenance_score *= 0.60
                reasons.append("Durable personal fact is backed only by low-trust sources such as web/tool content.")

        return self._verdict_from_scores(
            provenance_score=provenance_score,
            support_score=best_support,
            evidence_turn_ids=evidence_turn_ids,
            evidence_snippets=evidence_snippets,
            source_labels=source_labels,
            reasons=reasons,
        )

    def _check_evidence_graph(
        self,
        fact_text: str,
        category: FactCategory,
        ctx: ReflectionContext,
    ) -> DefenseVerdict:
        """
        句级证据选择：
        1. 把 turns 切成句子；
        2. 给每个句子计算 lexical / rarity / slot 三个信号；
        3. 选出 top-k 独立证据句，并依据 witness 数和来源可信度聚合 provenance。
        """

        evidence_sentences = self._collect_evidence_sentences(fact_text, category, ctx)
        if not evidence_sentences:
            return self._empty_verdict("未找到可支撑该摘要事实的句级证据")

        selected = [item for item in evidence_sentences if item.selector_score >= self.cfg.min_sentence_score]
        selected = selected[: self.cfg.max_evidence_sentences] or evidence_sentences[:1]

        support_score = clamp(sum(item.lexical_score for item in selected) / len(selected))
        exact_match_bonus = 0.15 if selected[0].lexical_score >= 0.95 else 0.0
        provenance_score = clamp(
            (0.50 * selected[0].selector_score)
            + (0.20 * (sum(item.selector_score for item in selected) / len(selected)))
            + (0.10 * self.SOURCE_WEIGHTS[selected[0].source])
            + exact_match_bonus
            + self._witness_bonus(selected)
            + self._source_diversity_bonus(selected)
        )
        reasons: List[str] = []

        if category in self.DURABLE_PERSONAL_CATEGORIES:
            has_user_backing = any(item.source in {SourceType.USER, SourceType.SYSTEM} for item in selected)
            if not has_user_backing:
                provenance_score *= 0.60
                reasons.append("Durable personal fact lacks user/system-backed evidence witnesses.")

        source_labels = [
            SourceLabel(
                turn_id=item.turn_id,
                source_type=item.source.value,
                label=default_integrity_for_source(item.source),
                selector_score=item.selector_score,
                excerpt=item.sentence[:160],
            )
            for item in selected
        ]

        return self._verdict_from_scores(
            provenance_score=provenance_score,
            support_score=support_score,
            evidence_turn_ids=[item.turn_id for item in selected],
            evidence_snippets=[item.sentence for item in selected],
            source_labels=source_labels,
            reasons=reasons,
        )

    def _collect_evidence_sentences(
        self,
        fact_text: str,
        category: FactCategory,
        ctx: ReflectionContext,
    ) -> List[EvidenceSentence]:
        sentences = []
        idf = self._build_token_idf(ctx)
        fact_tokens = tokenize(fact_text)
        total_fact_idf = sum(idf.get(token, 1.0) for token in fact_tokens) or 1.0

        for turn in ctx.turns:
            for sentence in split_sentences(turn.text):
                lexical = lexical_support(fact_text, sentence)
                if lexical <= 0.0:
                    continue
                rarity = self._rarity_score(fact_text, sentence, idf, total_fact_idf)
                slot_score = self._slot_score(fact_text, sentence, category)
                source_weight = self.SOURCE_WEIGHTS[turn.source]
                selector = clamp(
                    (0.40 * lexical)
                    + (0.20 * rarity)
                    + (0.20 * slot_score)
                    + (0.20 * source_weight)
                )
                sentences.append(
                    EvidenceSentence(
                        turn_id=turn.turn_id,
                        source=turn.source,
                        sentence=sentence,
                        lexical_score=lexical,
                        rarity_score=rarity,
                        slot_score=slot_score,
                        selector_score=selector,
                    )
                )

        sentences.sort(key=lambda item: item.selector_score, reverse=True)
        return sentences

    @staticmethod
    def _build_token_idf(ctx: ReflectionContext) -> Dict[str, float]:
        sentence_tokens = []
        for turn in ctx.turns:
            for sentence in split_sentences(turn.text):
                tokens = unique_tokens(sentence)
                if tokens:
                    sentence_tokens.append(tokens)
        sentence_count = len(sentence_tokens) or 1
        document_frequency: Dict[str, int] = {}
        for tokens in sentence_tokens:
            for token in tokens:
                document_frequency[token] = document_frequency.get(token, 0) + 1
        return {
            token: 1.0 + math.log((1.0 + sentence_count) / (1.0 + frequency))
            for token, frequency in document_frequency.items()
        }

    @staticmethod
    def _rarity_score(fact_text: str, sentence: str, idf: Dict[str, float], total_fact_idf: float) -> float:
        fact_tokens = tokenize(fact_text)
        sentence_tokens = unique_tokens(sentence)
        overlap = [token for token in fact_tokens if token in sentence_tokens]
        overlap_idf = sum(idf.get(token, 1.0) for token in overlap)
        return clamp(overlap_idf / total_fact_idf)

    @staticmethod
    def _slot_score(fact_text: str, sentence: str, category: FactCategory) -> float:
        fact_slot = extract_slot_key(fact_text, category)
        fact_value = extract_canonical_value(fact_text, category)
        sentence_slot = extract_slot_key(sentence, category)
        sentence_value = extract_canonical_value(sentence, category)
        if fact_slot == sentence_slot and fact_value and sentence_value and fact_value == sentence_value:
            return 1.0
        if fact_slot == sentence_slot:
            return 0.5
        return 0.0

    def _witness_bonus(self, selected: List[EvidenceSentence]) -> float:
        unique_turns = len({item.turn_id for item in selected})
        bonus = max(unique_turns - 1, 0) * self.cfg.independent_witness_bonus
        return min(0.20, bonus)

    @staticmethod
    def _source_diversity_bonus(selected: List[EvidenceSentence]) -> float:
        unique_sources = len({item.source for item in selected})
        return min(0.10, 0.03 * max(unique_sources - 1, 0))

    def _verdict_from_scores(
        self,
        provenance_score: float,
        support_score: float,
        evidence_turn_ids: List[str],
        evidence_snippets: List[str],
        source_labels: List[SourceLabel],
        reasons: List[str],
    ) -> DefenseVerdict:
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
                "support_score": clamp(support_score),
                "provenance_score": clamp(provenance_score),
                "evidence_turn_ids": evidence_turn_ids,
                "evidence_snippets": evidence_snippets,
                "source_labels": source_labels,
                "reasons": reasons,
                "strategy": self.cfg.strategy,
            },
        )

    @staticmethod
    def _empty_verdict(reason: str) -> DefenseVerdict:
        return DefenseVerdict(
            node="D2",
            passed=False,
            score=0.0,
            reason=reason,
            action="FLAG",
            metadata={
                "support_score": 0.0,
                "provenance_score": 0.0,
                "evidence_turn_ids": [],
                "evidence_snippets": [],
                "source_labels": [],
                "reasons": [reason],
            },
        )
