"""Pipeline orchestration.

本文件只负责编排，不把具体防御逻辑塞进主流程。
每个 d*.py 文件对应一个防御节点，便于单独替换、测试和消融实验。
"""

from __future__ import annotations

from .d1_input_label import ingest_external_content
from .d2_candidate_extract import FactExtractor, RuleBasedFactExtractor, agent_process_external
from .d4_provenance_gate import MemoryVerifier, MemoryWriteGateway
from .d5_retrieval_audit import MemoryRetriever
from .d6_execution_align import ActionMediator, SimplePlanner
from .d7_revocation import MemoryRevoker
from .memory_store import AuditLog, MemoryStore, QuarantineStore
from .types import (
    ActionDecision,
    ActionProposal,
    Agent,
    MemoryContextItem,
    MemoryRecord,
    MemoryScope,
    PipelineResult,
    Task,
)


class SecureSharedWritePipeline:
    """多 Agent 共享写入防御体系。

    默认包含两条完整拦截路径：

    路径 A：外部输入 -> Agent A 候选事实 -> 验证器 -> 写入共享记忆
    路径 B：共享记忆检索 -> Agent B 动作提案 -> 工具调用仲裁
    """

    def __init__(
        self,
        memory_store: MemoryStore,
        quarantine: QuarantineStore,
        audit_log: AuditLog,
        extractor: FactExtractor,
        verifier: MemoryVerifier,
        write_gateway: MemoryWriteGateway,
        retriever: MemoryRetriever,
        planner: SimplePlanner,
        action_mediator: ActionMediator,
        revoker: MemoryRevoker,
    ) -> None:
        self.memory_store = memory_store
        self.quarantine = quarantine
        self.audit_log = audit_log
        self.extractor = extractor
        self.verifier = verifier
        self.write_gateway = write_gateway
        self.retriever = retriever
        self.planner = planner
        self.action_mediator = action_mediator
        self.revoker = revoker

    def path_external_to_shared_memory(
        self,
        agent_a: Agent,
        verifier_agent: Agent,
        raw_external_content: str,
        task: Task,
    ) -> list[MemoryRecord]:
        """路径 A：外部内容进入共享记忆前的完整拦截链。

        任何失败都会进入 quarantine 或被拒绝，不会静默写入 shared memory。
        """

        external_input = ingest_external_content(
            raw_content=raw_external_content,
            source_uri=task.source_uri,
            source_type=task.source_type,
            audit_log=self.audit_log,
        )

        candidates = agent_process_external(
            agent=agent_a,
            external_input=external_input,
            task=task,
            extractor=self.extractor,
            quarantine=self.quarantine,
            audit_log=self.audit_log,
        )

        written: list[MemoryRecord] = []

        for candidate in candidates:
            promoted = self.verifier.verify_and_promote(candidate, verifier_agent)
            if promoted is None:
                continue

            record = self.write_gateway.write(
                agent=verifier_agent,
                candidate=promoted,
                target_scope=MemoryScope.SHARED.value,
            )
            if record is not None:
                written.append(record)

        return written

    def path_shared_memory_to_action(
        self,
        agent_b: Agent,
        task: Task,
    ) -> tuple[list[MemoryContextItem], ActionProposal, ActionDecision]:
        """路径 B：共享记忆影响高权限动作前的完整拦截链。"""

        context = self.retriever.retrieve(
            agent=agent_b,
            query=task.query,
            task_context=task.context,
        )

        proposal = self.planner.propose(
            agent=agent_b,
            task=task,
            context=context,
        )

        decision = self.action_mediator.evaluate_and_execute(
            agent=agent_b,
            proposal=proposal,
            context=context,
        )

        return context, proposal, decision

    def full_secure_path(
        self,
        agent_a: Agent,
        agent_b: Agent,
        verifier_agent: Agent,
        raw_external_content: str,
        task: Task,
    ) -> PipelineResult:
        """端到端路径：外部内容 -> 共享记忆 -> 高权限动作。"""

        written = self.path_external_to_shared_memory(
            agent_a=agent_a,
            verifier_agent=verifier_agent,
            raw_external_content=raw_external_content,
            task=task,
        )

        context, proposal, decision = self.path_shared_memory_to_action(
            agent_b=agent_b,
            task=task,
        )

        return PipelineResult(
            written_memories=written,
            retrieved_context=context,
            action_proposal=proposal,
            action_decision=decision,
        )

    def revoke_memory(self, memory_id: str, reason: str) -> list[str]:
        """安全运营入口：确认污染后撤销记忆及派生记录。"""

        return self.revoker.revoke_poisoned_memory(memory_id, reason)


def build_default_pipeline() -> SecureSharedWritePipeline:
    """构建默认可运行管线。"""

    memory_store = MemoryStore()
    quarantine = QuarantineStore()
    audit_log = AuditLog()

    extractor = RuleBasedFactExtractor()
    verifier = MemoryVerifier(memory_store, quarantine, audit_log)
    write_gateway = MemoryWriteGateway(memory_store, quarantine, audit_log)
    retriever = MemoryRetriever(memory_store, audit_log)
    planner = SimplePlanner()
    action_mediator = ActionMediator(audit_log)
    revoker = MemoryRevoker(memory_store, audit_log)

    return SecureSharedWritePipeline(
        memory_store=memory_store,
        quarantine=quarantine,
        audit_log=audit_log,
        extractor=extractor,
        verifier=verifier,
        write_gateway=write_gateway,
        retriever=retriever,
        planner=planner,
        action_mediator=action_mediator,
        revoker=revoker,
    )
