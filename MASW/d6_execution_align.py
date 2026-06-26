"""d6: execution alignment.

LLM 或 Agent 只能产生 ActionProposal。真正的工具调用由确定性策略决定。
这一步用于阻断“低信任记忆 -> 高权限 Agent -> 高影响工具”的混淆代理链路。
"""

from __future__ import annotations

from .config import HIGH_IMPACT_ACTIONS, RISK_THRESHOLD_EXECUTE
from .d3_risk_filter import compute_context_risk
from .memory_store import AuditLog
from .types import (
    ActionDecision,
    ActionProposal,
    Agent,
    AuditEventType,
    MemoryContextItem,
    Task,
    ToolExecutor,
)


class SimplePlanner:
    """无 LLM 依赖的示例规划器。

    真实系统可替换为 LLM planner，但输出仍必须是 ActionProposal，
    并且必须经过 ActionMediator 才能执行。
    """

    def propose(self, agent: Agent, task: Task, context: list[MemoryContextItem]) -> ActionProposal:
        tool = task.requested_action or "noop"
        return ActionProposal(
            action=tool,
            tool=tool,
            resource=task.target_resource,
            rationale="Proposed from task request and retrieved memory evidence.",
            depends_on_memory_ids=tuple(item.memory_id for item in context),
        )


class ActionMediator:
    """工具调用仲裁器。"""

    def __init__(
        self,
        audit_log: AuditLog,
        tool_executors: dict[str, ToolExecutor] | None = None,
    ) -> None:
        self.audit_log = audit_log
        self.tool_executors = tool_executors or {}

    def evaluate_and_execute(
        self,
        agent: Agent,
        proposal: ActionProposal,
        context: list[MemoryContextItem],
    ) -> ActionDecision:
        self.audit_log.append(
            AuditEventType.ACTION_PROPOSED,
            actor=agent.id,
            proposal_id=proposal.id,
            tool=proposal.tool,
            resource=proposal.resource,
            depends_on_memory_ids=proposal.depends_on_memory_ids,
        )

        denial = self._validate(agent, proposal, context)
        if denial is not None:
            decision = ActionDecision(
                proposal_id=proposal.id,
                executed=False,
                reason=denial,
                requires_human_approval="approval" in denial.lower(),
            )
            self.audit_log.append(
                AuditEventType.ACTION_DENIED,
                actor=agent.id,
                proposal_id=proposal.id,
                reason=decision.reason,
            )
            return decision

        executor = self.tool_executors.get(proposal.tool)
        if executor is not None:
            executor(proposal)

        decision = ActionDecision(
            proposal_id=proposal.id,
            executed=True,
            reason="Action allowed by policy and context risk checks",
        )
        self.audit_log.append(
            AuditEventType.ACTION_EXECUTED,
            actor=agent.id,
            proposal_id=proposal.id,
            tool=proposal.tool,
            resource=proposal.resource,
            depends_on_memory_ids=proposal.depends_on_memory_ids,
        )
        return decision

    def _validate(
        self,
        agent: Agent,
        proposal: ActionProposal,
        context: list[MemoryContextItem],
    ) -> str | None:
        if proposal.tool == "noop":
            return None

        if not agent.can_use_tool(proposal.tool):
            return "Tool not allowed for agent"

        context_risk = compute_context_risk(context)
        if context_risk > RISK_THRESHOLD_EXECUTE:
            return f"Human approval required: risky memory context, risk={context_risk:.2f}"

        if proposal.tool in HIGH_IMPACT_ACTIONS:
            return "Human approval required: high-impact tool"

        return None
