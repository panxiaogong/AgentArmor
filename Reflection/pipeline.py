from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from typing import List

from Reflection.config import PipelineConfig
from Reflection.d1_reflection_intent import D1ReflectionIntentDetector
from Reflection.d2_grounding import D2GroundingAuditor
from Reflection.d3_consistency import D3ConsistencyAuditor
from Reflection.d4_policy import D4PolicyAuditor
from Reflection.d5_write_gate import D5WriteGate
from Reflection.memory_store import MemoryStore
from Reflection.provenance import ProvenanceBinder
from Reflection.reflector import HeuristicReflector
from Reflection.retrieval_guard import RetrievalDefenseGuard
from Reflection.types import (
    DefenseVerdict,
    DecisionAction,
    FactAssessment,
    InjectionAssessment,
    MemoryRecord,
    PipelineResult,
    ReflectionCandidate,
    ReflectionContext,
    RetrievalResult,
)
from Reflection.utils import classify_fact, first_non_empty, split_sentences


@dataclass
class WriteResult(PipelineResult):
    """一次反思写入路径的执行结果。"""


def _audit_log(path: str | None, event: dict) -> None:
    line = json.dumps(event, ensure_ascii=False, default=str)
    print(f"[REFLECTION-AUDIT] {line}", file=sys.stderr)
    if path:
        try:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError:
            pass


class ReflectionDefensePipeline:
    """
    Reflection（类型三）纵深防御主管线。

    写入路径：
      D1：反思意图筛查
      D2：接地性审计
      D3：一致性核查
      D4：存储策略核查
      D5：最终写入闸门
      Provenance：签名/来源标签绑定

    检索路径：
      Retrieval：验签 + trust ranking + 协调异常检测
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        reflector: HeuristicReflector | None = None,
        d1: D1ReflectionIntentDetector | None = None,
        d2: D2GroundingAuditor | None = None,
        d3: D3ConsistencyAuditor | None = None,
        d4: D4PolicyAuditor | None = None,
        d5: D5WriteGate | None = None,
        provenance_binder: ProvenanceBinder | None = None,
        retrieval_guard: RetrievalDefenseGuard | None = None,
    ) -> None:
        self.cfg = config or PipelineConfig()
        self.reflector = reflector or HeuristicReflector()
        self.d1 = d1 or D1ReflectionIntentDetector(self.cfg.d1)
        self.d2 = d2 or D2GroundingAuditor(self.cfg.d2)
        self.d3 = d3 or D3ConsistencyAuditor(self.cfg.d3)
        self.d4 = d4 or D4PolicyAuditor(self.cfg.d4)
        self.d5 = d5 or D5WriteGate(self.cfg.d5, self.cfg.d2)
        self.provenance = provenance_binder or ProvenanceBinder(self.cfg.provenance)
        self.retrieval = retrieval_guard or RetrievalDefenseGuard(self.cfg.retrieval, self.provenance)

    @classmethod
    def from_config(
        cls,
        config: PipelineConfig | None = None,
        reflector: HeuristicReflector | None = None,
    ) -> "ReflectionDefensePipeline":
        cfg = config or PipelineConfig()
        binder = ProvenanceBinder(cfg.provenance)
        return cls(
            config=cfg,
            reflector=reflector or HeuristicReflector(),
            d1=D1ReflectionIntentDetector(cfg.d1),
            d2=D2GroundingAuditor(cfg.d2),
            d3=D3ConsistencyAuditor(cfg.d3),
            d4=D4PolicyAuditor(cfg.d4),
            d5=D5WriteGate(cfg.d5, cfg.d2),
            provenance_binder=binder,
            retrieval_guard=RetrievalDefenseGuard(cfg.retrieval, binder),
        )

    def synthesize(self, turns: List["ConversationTurn"]) -> ReflectionCandidate:
        return self.reflector.synthesize(turns)

    def process(self, turns: List["ConversationTurn"], memory_store: MemoryStore | None = None) -> PipelineResult:
        candidate = self.synthesize(turns)
        return self.evaluate(turns, candidate, memory_store)

    def evaluate(
        self,
        turns: List["ConversationTurn"],
        candidate: ReflectionCandidate,
        memory_store: MemoryStore | None = None,
    ) -> PipelineResult:
        started_at = time.time()
        candidate = self._normalize_candidate(candidate)
        context = self._build_context(turns, candidate)
        d1_verdict = self.d1.check(context)
        injection = InjectionAssessment(
            score=d1_verdict.score,
            hard_block=d1_verdict.action == "BLOCK",
            matched_rules=list(d1_verdict.metadata.get("matched_rules", [])),
        )
        result = WriteResult(candidate=candidate, injection=injection, verdicts=[d1_verdict])

        if d1_verdict.action == "BLOCK":
            self._apply_global_block(result, candidate.fact_texts, d1_verdict)
            self._audit_write("write_blocked", result, started_at)
            return result

        for index, fact_text in enumerate(candidate.fact_texts, start=1):
            fact_result = self._evaluate_fact(fact_text, context, d1_verdict, memory_store)
            # 全局 verdict 只追加“新节点判决”，避免 D1 在每个 fact 上重复灌入。
            result.verdicts.extend(fact_result.verdicts[1:])
            self._store_fact_result(result, fact_result, candidate, context, index, memory_store)

        if result.rejected_facts and not result.accepted_records and not result.quarantined_facts:
            result.blocked_by = self._infer_blocked_by(result.verdicts)

        self._audit_write("write_completed", result, started_at)
        return result

    def on_retrieval(
        self,
        query: str,
        memory_store: MemoryStore,
        limit: int | None = None,
    ) -> RetrievalResult:
        raw_ranked = memory_store.rank(query)[: limit or self.cfg.retrieval.max_results]
        result = self.retrieval.check(query, raw_ranked)
        self._audit_retrieval(query, result)
        return result

    @staticmethod
    def _normalize_candidate(candidate: ReflectionCandidate) -> ReflectionCandidate:
        fact_texts = candidate.fact_texts or split_sentences(candidate.summary_text)
        return ReflectionCandidate(
            summary_text=candidate.summary_text,
            fact_texts=fact_texts,
            metadata=dict(candidate.metadata),
        )

    def _build_context(
        self,
        turns: List["ConversationTurn"],
        candidate: ReflectionCandidate,
    ) -> ReflectionContext:
        return ReflectionContext(
            turns=turns,
            candidate=candidate,
            triggering_query=self._infer_triggering_query(turns),
            metadata=dict(candidate.metadata),
        )

    def _evaluate_fact(
        self,
        fact_text: str,
        context: ReflectionContext,
        d1_verdict: DefenseVerdict,
        memory_store: MemoryStore | None,
    ) -> FactAssessment:
        category = classify_fact(fact_text)
        assessment = FactAssessment(
            fact_text=fact_text,
            category=category,
            injection_score=d1_verdict.score,
        )
        assessment.verdicts.append(d1_verdict)
        if d1_verdict.action == "FLAG":
            assessment.reasons.append(d1_verdict.reason)

        # D2 把“哪几句原始证据支持这条摘要事实”挑出来，并打上来源标签。
        d2_verdict = self.d2.check(fact_text, category, context)
        assessment.verdicts.append(d2_verdict)
        assessment.support_score = float(d2_verdict.metadata.get("support_score", 0.0))
        assessment.provenance_score = float(d2_verdict.metadata.get("provenance_score", 0.0))
        assessment.evidence_turn_ids = list(d2_verdict.metadata.get("evidence_turn_ids", []))
        assessment.source_labels = list(d2_verdict.metadata.get("source_labels", []))
        assessment.metadata["evidence_snippets"] = list(d2_verdict.metadata.get("evidence_snippets", []))
        assessment.reasons.extend(d2_verdict.metadata.get("reasons", []))
        if d2_verdict.action == "FLAG":
            assessment.reasons.append(d2_verdict.reason)

        d3_verdict = self.d3.check(fact_text, category, memory_store)
        assessment.verdicts.append(d3_verdict)
        assessment.contradiction_score = float(d3_verdict.metadata.get("contradiction_score", 0.0))
        if d3_verdict.action == "BLOCK":
            assessment.reasons.append(d3_verdict.reason)
            assessment.final_risk = 1.0
            assessment.action = DecisionAction.REJECT
            return assessment

        d4_verdict = self.d4.check(fact_text, category)
        assessment.verdicts.append(d4_verdict)
        assessment.policy_score = float(d4_verdict.metadata.get("policy_score", 0.0))
        if d4_verdict.action == "BLOCK":
            assessment.reasons.append(d4_verdict.reason)
            assessment.final_risk = 1.0
            assessment.action = DecisionAction.REJECT
            return assessment
        if d4_verdict.action == "FLAG":
            assessment.reasons.append(d4_verdict.reason)

        d5_verdict = self.d5.check(assessment)
        assessment.verdicts.append(d5_verdict)
        assessment.final_risk = float(d5_verdict.metadata.get("final_risk", assessment.final_risk))
        if d5_verdict.action == "BLOCK":
            assessment.reasons.append(d5_verdict.reason)
            assessment.action = DecisionAction.REJECT
        elif d5_verdict.action == "FLAG":
            assessment.reasons.append(d5_verdict.reason)
            assessment.action = DecisionAction.QUARANTINE
        else:
            assessment.action = DecisionAction.ACCEPT
        return assessment

    @staticmethod
    def _build_record(
        index: int,
        assessment: FactAssessment,
        candidate: ReflectionCandidate,
    ) -> MemoryRecord:
        return MemoryRecord(
            record_id=f"mem-{index:03d}",
            fact_text=assessment.fact_text,
            category=assessment.category,
            provenance_score=assessment.provenance_score,
            evidence_turn_ids=list(assessment.evidence_turn_ids),
            source_summary=first_non_empty([candidate.summary_text, assessment.fact_text]),
            metadata={
                "risk": assessment.final_risk,
                "verdict_trace": [verdict.node for verdict in assessment.verdicts],
                "evidence_snippets": assessment.metadata.get("evidence_snippets", []),
            },
        )

    def _store_fact_result(
        self,
        result: WriteResult,
        fact_result: FactAssessment,
        candidate: ReflectionCandidate,
        context: ReflectionContext,
        index: int,
        memory_store: MemoryStore | None,
    ) -> None:
        result.facts.append(fact_result)
        if fact_result.action == DecisionAction.ACCEPT:
            record = self._build_record(index, fact_result, candidate)
            self.provenance.bind(record, fact_result, context)
            result.accepted_records.append(record)
            if memory_store is not None:
                memory_store.commit(record)
            return
        if fact_result.action == DecisionAction.QUARANTINE:
            result.quarantined_facts.append(fact_result)
            if memory_store is not None:
                memory_store.quarantine_fact(fact_result)
            return
        result.rejected_facts.append(fact_result)

    @staticmethod
    def _apply_global_block(
        result: WriteResult,
        fact_texts: List[str],
        verdict: DefenseVerdict,
    ) -> None:
        result.blocked_by = verdict.node
        for fact_text in fact_texts:
            assessment = FactAssessment(
                fact_text=fact_text,
                category=classify_fact(fact_text),
                injection_score=verdict.score,
                action=DecisionAction.REJECT,
            )
            assessment.reasons.append(verdict.reason)
            assessment.verdicts.append(verdict)
            result.facts.append(assessment)
            result.rejected_facts.append(assessment)

    @staticmethod
    def _infer_triggering_query(turns: List["ConversationTurn"]) -> str:
        for turn in reversed(turns):
            if turn.source.name in {"USER", "WEB", "TOOL", "ASSISTANT"}:
                return turn.text
        return turns[-1].text if turns else ""

    @staticmethod
    def _infer_blocked_by(verdicts: List[DefenseVerdict]) -> str | None:
        for verdict in verdicts:
            if verdict.action == "BLOCK":
                return verdict.node
        return None

    def _audit_write(self, event_name: str, result: PipelineResult, started_at: float) -> None:
        _audit_log(
            self.cfg.audit_log_path,
            {
                "event": event_name,
                "elapsed_s": round(time.time() - started_at, 4),
                "accepted": len(result.accepted_records),
                "quarantined": len(result.quarantined_facts),
                "rejected": len(result.rejected_facts),
                "blocked_by": result.blocked_by,
                "nodes_run": [verdict.node for verdict in result.verdicts],
            },
        )

    def _audit_retrieval(self, query: str, result: RetrievalResult) -> None:
        _audit_log(
            self.cfg.audit_log_path,
            {
                "event": "retrieval_checked",
                "query_preview": query[:80],
                "returned": len(result.entries),
                "tampered": result.tampered_count,
                "flagged": sum(1 for entry in result.entries if entry.flagged),
                "nodes_run": [verdict.node for verdict in result.verdicts],
            },
        )
