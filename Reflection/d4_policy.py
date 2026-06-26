"""
D4：存储策略核查——什么事实值得进入长时反思记忆。

不是所有“可以被摘要出来”的内容都应该被永久记住。
类型三里尤其需要阻止：
  - 指令型文本进入事实记忆
  - 凭证/秘密进入长时记忆
  - 明显会过期的一次性任务上下文被长期固化
"""
from __future__ import annotations

from Reflection.config import D4Config
from Reflection.types import DefenseVerdict, FactCategory
from Reflection.utils import has_any


class D4PolicyAuditor:
    """执行长时记忆存储策略。"""

    EPHEMERAL_TERMS = {
        "today",
        "this run",
        "for this task only",
        "one-time",
        "temporary",
        "session only",
        "just for now",
        "今天",
        "本轮",
        "临时",
        "仅本次",
    }

    def __init__(self, config: D4Config) -> None:
        self.cfg = config

    def check(self, fact_text: str, category: FactCategory) -> DefenseVerdict:
        if not self.cfg.enabled:
            return DefenseVerdict("D4", passed=True, score=0.0, reason="D4 已禁用", action="PASS")
        if self.cfg.strategy == "rule_policy":
            return self._rule_policy_check(fact_text, category)
        if self.cfg.strategy == "strict_privacy":
            return self._strict_privacy_check(fact_text, category)
        raise ValueError(f"未知 D4 strategy: {self.cfg.strategy}")

    def _rule_policy_check(self, fact_text: str, category: FactCategory) -> DefenseVerdict:
        lower = fact_text.lower()
        if category == FactCategory.CREDENTIAL and self.cfg.block_credential:
            return DefenseVerdict(
                node="D4",
                passed=False,
                score=1.0,
                reason="凭证与秘密信息不应被写入反思型长时记忆",
                action="BLOCK",
                metadata={"policy_score": 1.0},
            )

        if category == FactCategory.INSTRUCTION and self.cfg.block_instruction:
            return DefenseVerdict(
                node="D4",
                passed=False,
                score=1.0,
                reason="指令型文本不应被当作可信事实写入长期记忆",
                action="BLOCK",
                metadata={"policy_score": 1.0},
            )

        if category == FactCategory.TASK and self.cfg.flag_ephemeral_task and has_any(lower, self.EPHEMERAL_TERMS):
            return DefenseVerdict(
                node="D4",
                passed=False,
                score=0.65,
                reason="该任务事实更像会话态临时上下文，不宜直接固化为长时记忆",
                action="FLAG",
                metadata={"policy_score": 0.65},
            )

        if len(fact_text.strip()) < 12:
            # 过短事实往往缺少上下文，直接放行会给后续检索埋下噪声。
            return DefenseVerdict(
                node="D4",
                passed=False,
                score=0.30,
                reason="候选事实过短，缺乏足够语义，建议隔离后再确认",
                action="FLAG",
                metadata={"policy_score": 0.30},
            )

        return DefenseVerdict(
            node="D4",
            passed=True,
            score=0.0,
            reason="满足长时记忆存储策略",
            action="PASS",
            metadata={"policy_score": 0.0},
        )

    def _strict_privacy_check(self, fact_text: str, category: FactCategory) -> DefenseVerdict:
        base_verdict = self._rule_policy_check(fact_text, category)
        if base_verdict.action != "PASS":
            return base_verdict

        if category == FactCategory.CONTACT:
            return DefenseVerdict(
                node="D4",
                passed=False,
                score=0.75,
                reason="strict_privacy 模式下，contact 类事实默认进入隔离区等待复核",
                action="FLAG",
                metadata={"policy_score": 0.75},
            )
        if category == FactCategory.TASK:
            return DefenseVerdict(
                node="D4",
                passed=False,
                score=0.45,
                reason="strict_privacy 模式下，task 类事实不默认固化为长期记忆",
                action="FLAG",
                metadata={"policy_score": 0.45},
            )
        return base_verdict
