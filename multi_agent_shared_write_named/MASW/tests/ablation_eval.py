"""Ablation evaluation for MASW defense nodes.

The experiment answers two questions:
1. Does each node defend the attack stage it is responsible for?
2. Does the full composition show strict 1+1>2 synergy on this minimum set?

Prediction definition:
- Positive means the system judged/intercepted the sample as attack at any
  defense stage: taint-only classification, risk quarantine, write denial,
  retrieval filtering, or action denial.
- For benign samples, any such interception is a false positive.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from time import perf_counter_ns

from MASW.config import RISK_THRESHOLD_WRITE
from MASW.d1_input_label import ingest_external_content
from MASW.d2_candidate_extract import RuleBasedFactExtractor
from MASW.d3_risk_filter import detect_injection_risk, looks_like_instruction
from MASW.d4_provenance_gate import MemoryVerifier, MemoryWriteGateway
from MASW.d5_retrieval_audit import MemoryRetriever
from MASW.d6_execution_align import ActionMediator, SimplePlanner
from MASW.memory_store import AuditLog, MemoryStore, QuarantineStore
from MASW.tests.build_dataset import build_samples, write_csv, write_jsonl
from MASW.tests.eval_masw import _agents
from MASW.types import (
    CandidateFact,
    ExternalInput,
    MemoryContextItem,
    MemoryRecord,
    MemoryScope,
    Task,
    TaskContext,
    TrustLevel,
)


RESULTS_DIR = Path(__file__).resolve().parent / "results"
ABLATION_JSON_PATH = RESULTS_DIR / "masw_ablation_results.json"
ABLATION_CSV_PATH = RESULTS_DIR / "masw_ablation_metrics.csv"
STAGE_REPORT_PATH = RESULTS_DIR / "masw_ablation_stage_report.md"


@dataclass(frozen=True)
class AblationConfig:
    name: str
    d1_input_label: bool = False
    d2_candidate_extract: bool = True
    d3_risk_filter: bool = False
    d4_provenance_gate: bool = False
    d5_retrieval_audit: bool = False
    d6_execution_align: bool = False
    d7_revocation: bool = False
    taint_as_detection: bool = False

    @property
    def enabled_nodes(self) -> str:
        nodes: list[str] = []
        if self.d1_input_label:
            nodes.append("d1")
        if self.d2_candidate_extract:
            nodes.append("d2")
        if self.d3_risk_filter:
            nodes.append("d3")
        if self.d4_provenance_gate:
            nodes.append("d4")
        if self.d5_retrieval_audit:
            nodes.append("d5")
        if self.d6_execution_align:
            nodes.append("d6")
        if self.d7_revocation:
            nodes.append("d7")
        return "+".join(nodes) if nodes else "none"


CONFIGS = [
    AblationConfig(name="BASELINE", d2_candidate_extract=False),
    AblationConfig(name="D1_INPUT_LABEL_ONLY", d1_input_label=True, d2_candidate_extract=False, taint_as_detection=True),
    AblationConfig(name="D2_EXTRACT_ONLY", d2_candidate_extract=True),
    AblationConfig(name="D3_RISK_FILTER_ONLY", d2_candidate_extract=True, d3_risk_filter=True),
    AblationConfig(name="D4_PROVENANCE_GATE", d1_input_label=True, d2_candidate_extract=True, d4_provenance_gate=True),
    AblationConfig(name="D5_RETRIEVAL_AUDIT", d1_input_label=True, d2_candidate_extract=True, d5_retrieval_audit=True),
    AblationConfig(name="D6_EXECUTION_ALIGN", d1_input_label=True, d2_candidate_extract=True, d6_execution_align=True),
    AblationConfig(name="D3_D4_COMBO", d1_input_label=True, d2_candidate_extract=True, d3_risk_filter=True, d4_provenance_gate=True),
    AblationConfig(name="D5_D6_COMBO", d1_input_label=True, d2_candidate_extract=True, d5_retrieval_audit=True, d6_execution_align=True),
    AblationConfig(
        name="ALL",
        d1_input_label=True,
        d2_candidate_extract=True,
        d3_risk_filter=True,
        d4_provenance_gate=True,
        d5_retrieval_audit=True,
        d6_execution_align=True,
        d7_revocation=True,
    ),
    AblationConfig(
        name="ALL_MINUS_D3",
        d1_input_label=True,
        d2_candidate_extract=True,
        d4_provenance_gate=True,
        d5_retrieval_audit=True,
        d6_execution_align=True,
        d7_revocation=True,
    ),
    AblationConfig(
        name="ALL_MINUS_D4",
        d1_input_label=True,
        d2_candidate_extract=True,
        d3_risk_filter=True,
        d5_retrieval_audit=True,
        d6_execution_align=True,
        d7_revocation=True,
    ),
    AblationConfig(
        name="ALL_MINUS_D5_D6",
        d1_input_label=True,
        d2_candidate_extract=True,
        d3_risk_filter=True,
        d4_provenance_gate=True,
        d7_revocation=True,
    ),
]


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((percentile / 100) * (len(ordered) - 1)))
    return ordered[index]


def _task_for_sample(sample: dict[str, object]) -> Task:
    requested_action = "ticket.create"
    if sample["category"] == "tool_misuse":
        requested_action = "email.send"
    if "deploy.production" in str(sample["content"]):
        requested_action = "deploy.production"
    if "database.write" in str(sample["content"]):
        requested_action = "database.write"
    if "repo.commit" in str(sample["content"]):
        requested_action = "repo.commit"

    return Task(
        id=str(sample["id"]),
        query=str(sample["user_query"]),
        source_uri=str(sample["source_uri"]),
        source_type=str(sample["source_type"]),
        context=TaskContext(min_required_trust=TrustLevel.VERIFIED),
        requested_action=requested_action,
        target_resource="deployment-board",
    )


def _raw_external_input(sample: dict[str, object]) -> ExternalInput:
    return ExternalInput(
        content=str(sample["content"]),
        source_uri=str(sample["source_uri"]),
        source_type=str(sample["source_type"]),
        trust=TrustLevel.TRUSTED,
        taint=False,
    )


def _naive_candidate(sample: dict[str, object], external_input: ExternalInput) -> CandidateFact:
    return CandidateFact(
        subject="external_content",
        predicate="states",
        object=str(sample["content"]),
        confidence=0.90,
        evidence_span=str(sample["content"]),
        source=external_input.source_uri,
        writer="agent-a",
        trust=external_input.trust,
        taint=external_input.taint,
        parent_ids=(external_input.id,),
    )


def _force_write(store: MemoryStore, candidate: CandidateFact) -> MemoryRecord:
    record = MemoryRecord(
        content=candidate,
        writer=candidate.writer,
        source=candidate.source,
        trust=candidate.trust,
        scope=MemoryScope.SHARED.value,
        evidence=(candidate.evidence_span,),
        taint=candidate.taint,
        parent_ids=candidate.parent_ids,
    )
    store.insert(record)
    return record


def _context_without_retrieval_audit(store: MemoryStore) -> list[MemoryContextItem]:
    return [
        MemoryContextItem(
            memory_id=record.id,
            content=record.as_text(),
            source=record.source,
            trust=record.trust,
            taint=record.taint,
            score=1.0,
        )
        for record in store.all()
    ]


def _simulate(config: AblationConfig, sample: dict[str, object]) -> dict[str, object]:
    started = perf_counter_ns()
    audit = AuditLog()
    quarantine = QuarantineStore()
    store = MemoryStore()
    agent_a, verifier_agent, agent_b = _agents()
    task = _task_for_sample(sample)
    blocked_stage = "none"

    if config.d1_input_label:
        external_input = ingest_external_content(
            raw_content=str(sample["content"]),
            source_uri=str(sample["source_uri"]),
            source_type=str(sample["source_type"]),
            audit_log=audit,
        )
        if config.taint_as_detection:
            blocked_stage = "d1_taint_label"
            elapsed_ms = (perf_counter_ns() - started) / 1_000_000
            return {
                "predicted_attack": True,
                "blocked_stage": blocked_stage,
                "elapsed_ms": elapsed_ms,
                "memory_written": 0,
                "context_count": 0,
                "action_executed": False,
            }
    else:
        external_input = _raw_external_input(sample)

    if config.d2_candidate_extract:
        candidates = RuleBasedFactExtractor().extract(agent_a, external_input, task)
    else:
        candidates = [_naive_candidate(sample, external_input)]

    if config.d3_risk_filter:
        risky = [
            candidate
            for candidate in candidates
            if detect_injection_risk(candidate) > RISK_THRESHOLD_WRITE or looks_like_instruction(candidate)
        ]
        if risky:
            for candidate in risky:
                quarantine.add(candidate, "d3 risk filter")
            blocked_stage = "d3_risk_filter"
            elapsed_ms = (perf_counter_ns() - started) / 1_000_000
            return {
                "predicted_attack": True,
                "blocked_stage": blocked_stage,
                "elapsed_ms": elapsed_ms,
                "memory_written": 0,
                "context_count": 0,
                "action_executed": False,
            }

    if config.d4_provenance_gate:
        verifier = MemoryVerifier(store, quarantine, audit)
        gateway = MemoryWriteGateway(store, quarantine, audit)
        for candidate in candidates:
            promoted = verifier.verify_and_promote(candidate, verifier_agent)
            if promoted is not None:
                gateway.write(verifier_agent, promoted, MemoryScope.SHARED.value)
        if quarantine.items:
            blocked_stage = "d4_provenance_gate"
    else:
        for candidate in candidates:
            _force_write(store, candidate)

    if config.d5_retrieval_audit:
        raw_retrieval_candidates = store.vector_search(task.query)
        context = MemoryRetriever(store, audit).retrieve(agent_b, task.query, task.context)
        if raw_retrieval_candidates and not context and blocked_stage == "none":
            blocked_stage = "d5_retrieval_audit"
    else:
        context = _context_without_retrieval_audit(store)

    action_executed = True
    if config.d6_execution_align:
        proposal = SimplePlanner().propose(agent_b, task, context)
        decision = ActionMediator(audit).evaluate_and_execute(agent_b, proposal, context)
        action_executed = decision.executed
        if not decision.executed and blocked_stage == "none":
            blocked_stage = "d6_execution_align"

    if config.d7_revocation and blocked_stage != "none":
        for record in store.all():
            store.mark_revoked(record.id)

    elapsed_ms = (perf_counter_ns() - started) / 1_000_000
    return {
        "predicted_attack": blocked_stage != "none",
        "blocked_stage": blocked_stage,
        "elapsed_ms": elapsed_ms,
        "memory_written": len(store.all(include_revoked=True)),
        "context_count": len(context),
        "action_executed": action_executed,
    }


def _metrics(rows: list[dict[str, object]], include_latency: bool) -> dict[str, object]:
    tp = sum(1 for row in rows if row["label"] == "attack" and row["predicted_attack"])
    fp = sum(1 for row in rows if row["label"] == "benign" and row["predicted_attack"])
    fn = sum(1 for row in rows if row["label"] == "attack" and not row["predicted_attack"])
    tn = sum(1 for row in rows if row["label"] == "benign" and not row["predicted_attack"])

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0

    elapsed = [float(row["elapsed_ms"]) for row in rows]
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "p50_ms": median(elapsed) if include_latency else None,
        "p95_ms": _percentile(elapsed, 95) if include_latency else None,
        "p99_ms": _percentile(elapsed, 99) if include_latency else None,
    }


def _stage_counts(rows: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        stage = str(row["blocked_stage"])
        counts[stage] = counts.get(stage, 0) + 1
    return counts


def _write_csv(metrics_rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "config",
        "enabled_nodes",
        "precision",
        "recall",
        "f1",
        "fpr",
        "tp",
        "fp",
        "fn",
        "tn",
        "p50_ms",
        "p95_ms",
        "p99_ms",
    ]
    with ABLATION_CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_rows)


def _format_metric(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _build_stage_report(metrics_rows: list[dict[str, object]], synergy: dict[str, object]) -> str:
    by_name = {row["config"]: row for row in metrics_rows}
    lines = [
        "# MASW Ablation Stage Report",
        "",
        "## Node-Level Answers",
        "",
        "| Node | Corresponding stage | Result | Evidence |",
        "|---|---|---|---|",
    ]

    rows = [
        (
            "d1_input_label",
            "外部输入信任边界",
            "部分成立：能标记所有外部输入，但不能区分攻击/良性。",
            by_name["D1_INPUT_LABEL_ONLY"],
        ),
        (
            "d2_candidate_extract",
            "结构化候选事实抽取",
            "不是独立拦截器；它为 d3/d4 提供可审计输入。",
            by_name["D2_EXTRACT_ONLY"],
        ),
        (
            "d3_risk_filter",
            "提示注入/工具误用/记忆投毒/劫持文本早期过滤",
            "成立：当前最小集上单节点已高召回。",
            by_name["D3_RISK_FILTER_ONLY"],
        ),
        (
            "d4_provenance_gate",
            "共享记忆写入前的验证、提升和网关",
            "成立：阻断低信任或高风险内容自动进入 shared memory。",
            by_name["D4_PROVENANCE_GATE"],
        ),
        (
            "d5_retrieval_audit",
            "读取阶段的 trust/scope/taint 过滤",
            "部分成立：能阻断被召回的污染记忆，但单独使用覆盖不足。",
            by_name["D5_RETRIEVAL_AUDIT"],
        ),
        (
            "d6_execution_align",
            "工具调用和高影响动作仲裁",
            "部分成立：能阻断高风险上下文/工具，但单独使用误伤高。",
            by_name["D6_EXECUTION_ALIGN"],
        ),
        (
            "d7_revocation",
            "事后污染撤销和派生记忆清理",
            "不是入口拦截器；用于已确认污染后的 containment。",
            by_name["ALL"],
        ),
    ]

    for node, stage, result, evidence_row in rows:
        evidence = (
            f"Prec={_format_metric(evidence_row['precision'])}, "
            f"Rec={_format_metric(evidence_row['recall'])}, "
            f"FPR={_format_metric(evidence_row['fpr'])}"
        )
        lines.append(f"| `{node}` | {stage} | {result} | {evidence} |")

    lines.extend(
        [
            "",
            "## Composition Answer",
            "",
            f"- ALL F1: {_format_metric(synergy['all_f1'])}",
            f"- Best single-node F1: {_format_metric(synergy['best_single_f1'])} (`{synergy['best_single_config']}`)",
            f"- Strict F1 synergy (`ALL > best single`): {synergy['strict_f1_synergy']}",
            f"- F1 gain over best single: {_format_metric(synergy['f1_gain_over_best_single'])}",
            "",
            "Interpretation: strict 1+1>2 is only claimed when ALL exceeds the best single-node F1. "
            "If this is false, the current minimum dataset shows saturation by one early node rather than mathematical synergy. "
            "The combined design is still defense-in-depth because leave-one-out rows measure robustness when a stage is bypassed.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_ablation() -> dict[str, object]:
    write_jsonl()
    write_csv()
    samples = build_samples()
    all_results: dict[str, list[dict[str, object]]] = {}
    metrics_rows: list[dict[str, object]] = []

    for config in CONFIGS:
        rows: list[dict[str, object]] = []
        for sample in samples:
            outcome = _simulate(config, sample)
            rows.append(
                {
                    "id": sample["id"],
                    "label": sample["label"],
                    "category": sample["category"],
                    **outcome,
                }
            )

        metrics = _metrics(rows, include_latency=config.name == "ALL")
        metrics_row = {
            "config": config.name,
            "enabled_nodes": config.enabled_nodes,
            **metrics,
        }
        all_results[config.name] = rows
        metrics_rows.append(metrics_row)

    single_names = {
        "D1_INPUT_LABEL_ONLY",
        "D2_EXTRACT_ONLY",
        "D3_RISK_FILTER_ONLY",
        "D4_PROVENANCE_GATE",
        "D5_RETRIEVAL_AUDIT",
        "D6_EXECUTION_ALIGN",
    }
    single_rows = [row for row in metrics_rows if row["config"] in single_names]
    best_single = max(single_rows, key=lambda row: float(row["f1"]))
    all_row = next(row for row in metrics_rows if row["config"] == "ALL")
    synergy = {
        "all_f1": all_row["f1"],
        "best_single_f1": best_single["f1"],
        "best_single_config": best_single["config"],
        "strict_f1_synergy": float(all_row["f1"]) > float(best_single["f1"]),
        "f1_gain_over_best_single": float(all_row["f1"]) - float(best_single["f1"]),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(metrics_rows)
    STAGE_REPORT_PATH.write_text(_build_stage_report(metrics_rows, synergy), encoding="utf-8")
    payload = {
        "metrics": metrics_rows,
        "stage_counts": {name: _stage_counts(rows) for name, rows in all_results.items()},
        "synergy": synergy,
        "rows": all_results,
    }
    ABLATION_JSON_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    report = run_ablation()
    printable = {
        "metrics": report["metrics"],
        "synergy": report["synergy"],
    }
    print(json.dumps(printable, ensure_ascii=False, indent=2))
