from __future__ import annotations

from typing import List

from Reflection.types import ConversationTurn, ReflectionCandidate
from Reflection.utils import classify_fact, split_sentences


class HeuristicReflector:
    """
    A deterministic stand-in for the background LLM reflection process.

    The goal is not to be linguistically perfect; it is to reproduce the
    dangerous path where raw content is condensed into a handful of durable
    memory candidates that later receive elevated trust.
    """

    def synthesize(self, turns: List[ConversationTurn]) -> ReflectionCandidate:
        fact_texts: List[str] = []
        for turn in turns:
            for sentence in split_sentences(turn.text):
                category = classify_fact(sentence)
                # 这里刻意保留“偏宽松”的候选提取逻辑，
                # 因为类型三作品要模拟的正是：后台反思器容易把危险句子也蒸馏成事实。
                if category.value != "other" or "remember" in sentence.lower():
                    fact_texts.append(sentence)

        if not fact_texts and turns:
            fallback_sentences = split_sentences(turns[-1].text)
            if fallback_sentences:
                fact_texts.append(fallback_sentences[0])

        summary_text = ". ".join(fact_texts)
        return ReflectionCandidate(
            summary_text=summary_text,
            fact_texts=fact_texts,
            metadata={"turn_count": len(turns)},
        )
