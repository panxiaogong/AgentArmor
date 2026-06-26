"""
D1：反思触发层——检测是否存在“面向反思器本身”的提示注入。

类型三的危险点在于：
  攻击者不一定要直接诱导主 Agent 行动，
  他只要能控制“后台摘要/反思该如何写”，
  就能把不可信内容洗成未来对话中的“可信记忆”。
"""
from __future__ import annotations

from typing import Iterable, Sequence, Tuple

from Reflection.config import D1Config
from Reflection.types import DefenseVerdict, ReflectionContext


class D1ReflectionIntentDetector:
    """对反思/摘要写入路径做第一层快速筛查。"""

    RULES: Sequence[Tuple[str, float, bool, Sequence[str]]] = (
        (
            "reflection_directive",
            0.45,
            True,
            (
                "when summarizing",
                "in the summary",
                "write this to memory",
                "save this to memory",
                "store this in memory",
                "for future conversations",
                "future summaries should",
                "must remember",
                "remember this forever",
                "反思时记住",
                "总结时写入记忆",
            ),
        ),
        (
            "authority_laundering",
            0.20,
            False,
            (
                "system note",
                "developer note",
                "policy update",
                "high priority instruction",
                "internal requirement",
                "系统说明",
                "开发者说明",
                "高优先级指令",
            ),
        ),
        (
            "exfiltration_workflow",
            0.25,
            False,
            (
                "forward to",
                "send to",
                "upload to",
                "synchronize to",
                "route to",
                "share with",
                "转发到",
                "同步到",
                "上传到",
            ),
        ),
        (
            "secret_material",
            0.55,
            True,
            (
                "api key",
                "access token",
                "password",
                "secret credential",
                "session cookie",
                "密钥",
                "口令",
                "访问令牌",
            ),
        ),
    )

    def __init__(self, config: D1Config) -> None:
        self.cfg = config

    def check(self, ctx: ReflectionContext) -> DefenseVerdict:
        if not self.cfg.enabled:
            return DefenseVerdict("D1", passed=True, score=0.0, reason="D1 已禁用", action="PASS")

        # 这里故意把原始 turns、summary 和 fact_texts 拼接起来：
        # 类型三攻击可能只在其中一个视角里暴露出“面向反思器”的操控意图。
        joined = "\n".join(
            [ctx.joined_turn_text(), ctx.candidate.summary_text] + list(ctx.candidate.fact_texts)
        ).lower()
        if self.cfg.strategy == "keyword":
            return self._check_keyword_only(joined)
        if self.cfg.strategy == "hybrid":
            return self._check_hybrid(joined)
        if self.cfg.strategy == "pattern_graph":
            return self._check_pattern_graph(joined)
        raise ValueError(f"未知 D1 strategy: {self.cfg.strategy}")

    def _check_keyword_only(self, joined: str) -> DefenseVerdict:
        matched_rules = []
        score = 0.0
        hard_block = False

        for rule_name, weight, blocks, phrases in self.RULES:
            if self._matches(joined, phrases):
                matched_rules.append(rule_name)
                score += weight
                hard_block = hard_block or blocks

        return self._to_verdict(score, hard_block, matched_rules)

    def _check_hybrid(self, joined: str) -> DefenseVerdict:
        matched_rules = []
        score = 0.0
        hard_block = False

        for rule_name, weight, blocks, phrases in self.RULES:
            rule_hit = self._matches(joined, phrases)
            workflow_hit = rule_name == "exfiltration_workflow" and self._matches_exfiltration(joined)
            if rule_hit or workflow_hit:
                matched_rules.append(rule_name)
                score += weight
                hard_block = hard_block or blocks

        return self._to_verdict(score, hard_block, matched_rules)

    def _check_pattern_graph(self, joined: str) -> DefenseVerdict:
        matched_rules = []
        score = 0.0
        hard_block = False
        for rule_name, weight, blocks, phrases in self.RULES:
            rule_hit = self._matches(joined, phrases)
            workflow_hit = rule_name == "exfiltration_workflow" and self._matches_exfiltration(joined)
            if rule_hit or workflow_hit:
                matched_rules.append(rule_name)
                score += weight
                hard_block = hard_block or blocks

        # graph 视角：如果“反思指令 + 权威伪装”或“反思指令 + 外传工作流”共同出现，
        # 说明攻击者不只是插入一个危险词，而是在构造完整的 memory steering 模板。
        if "reflection_directive" in matched_rules and "authority_laundering" in matched_rules:
            score += self.cfg.graph_bonus
        if "reflection_directive" in matched_rules and "exfiltration_workflow" in matched_rules:
            score += self.cfg.graph_bonus
        if "secret_material" in matched_rules and "reflection_directive" in matched_rules:
            score += self.cfg.graph_bonus
            hard_block = True
        return self._to_verdict(score, hard_block, matched_rules)

    def _to_verdict(self, score: float, hard_block: bool, matched_rules: list[str]) -> DefenseVerdict:
        score = min(score, 1.0)
        metadata = {"matched_rules": matched_rules, "hard_block": hard_block, "strategy": self.cfg.strategy}
        if hard_block or score >= self.cfg.block_threshold:
            return DefenseVerdict(
                node="D1",
                passed=False,
                score=score,
                reason=f"检测到面向反思器的注入意图：{matched_rules}",
                action="BLOCK",
                metadata=metadata,
            )
        if score >= self.cfg.flag_threshold:
            return DefenseVerdict(
                node="D1",
                passed=False,
                score=score,
                reason=f"存在可疑反思操控信号：{matched_rules}",
                action="FLAG",
                metadata=metadata,
            )
        return DefenseVerdict(
            node="D1",
            passed=True,
            score=score,
            reason="未发现明显的反思型提示注入特征",
            action="PASS",
            metadata=metadata,
        )

    @staticmethod
    def _matches(text: str, phrases: Iterable[str]) -> bool:
        return any(phrase in text for phrase in phrases)

    @staticmethod
    def _matches_exfiltration(text: str) -> bool:
        verbs = ("forward", "send", "upload", "synchronize", "route", "share", "转发", "发送", "上传", "同步")
        return any(verb in text for verb in verbs) and (" to " in text or "@" in text or "到" in text)
