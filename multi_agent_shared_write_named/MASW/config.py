"""集中配置。

这个文件只放可调参数，不放流程逻辑。后续做实验时，可以只改这里
来比较不同阈值、防御强度和工具策略对 ASR/Utility 的影响。
"""

from __future__ import annotations

from .types import TrustLevel


# 写入共享记忆前的风险阈值。超过该值的候选事实进入隔离区。
RISK_THRESHOLD_WRITE = 0.45

# 自动验证阶段使用更严格阈值，避免可疑事实被提升为 VERIFIED。
RISK_THRESHOLD_VERIFY = 0.30

# 工具执行前的上下文风险阈值。超过该值需要人工审批或拒绝。
RISK_THRESHOLD_EXECUTE = 0.30

# 允许进入 shared memory 的最低信任级别。
MIN_SHARED_TRUST = TrustLevel.VERIFIED

# 自动验证要求的最低来源信誉。
MIN_SOURCE_REPUTATION_FOR_AUTO_VERIFY = 0.40

# 冲突检测时的置信度差距阈值。
CONFLICT_EPSILON = 0.15

# 高影响工具默认不自动执行。
HIGH_IMPACT_ACTIONS = frozenset(
    {
        "email.send",
        "database.write",
        "secret.read",
        "deploy.production",
        "repo.commit",
    }
)
