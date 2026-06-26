"""
Reflection 防御体系评测脚手架。

目标：
  - 提供比 `evaluation.py` 更像比赛实验的运行入口；
  - 支持写入路径消融和检索路径专项评估；
  - 输出可直接汇总到论文/作品报告的 CSV 和简表。

运行：
  python -m Reflection.tests.eval_reflection
"""
from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from Reflection import (
    D1Config,
    D2Config,
    D3Config,
    D4Config,
    D5Config,
    FactAssessment,
    FactCategory,
    IntegrityLabel,
    MemoryRecord,
    PipelineConfig,
    ReflectionCandidate,
    ReflectionContext,
    ReflectionDefensePipeline,
    RetrievalConfig,
    SourceLabel,
    SourceType,
)
from Reflection.memory_store import MemoryStore
from Reflection.tests.build_dataset import ALL_SAMPLES, RETRIEVAL_SCENARIOS, ReflectionSample, RetrievalScenario
from Reflection.types import ConversationTurn, DecisionAction


RESULTS_DIR = Path(__file__).resolve().parent / "results"


@dataclass
class EvalResult:
    sample_id: str
    subtype: str
    path_type: str
    config_name: str
    expected_label: str
    predicted_label: str
    decisive_node: str
    latency_ms: float
    detail: str


def build_config1() -> PipelineConfig:
    return PipelineConfig.unsafe()


def build_config2() -> PipelineConfig:
    cfg = PipelineConfig.unsafe()
    cfg.d1 = D1Config(enabled=True, strategy="keyword")
    return cfg


def build_config3() -> PipelineConfig:
    cfg = PipelineConfig.unsafe()
    cfg.d1 = D1Config(enabled=True, strategy="hybrid")
    cfg.d2 = D2Config(enabled=True, strategy="lexical")
    return cfg


def build_config4() -> PipelineConfig:
    cfg = PipelineConfig.unsafe()
    cfg.d1 = D1Config(enabled=True, strategy="pattern_graph")
    cfg.d2 = D2Config(enabled=True, strategy="evidence_graph")
    cfg.provenance.enabled = True
    cfg.d4 = D4Config(enabled=True, strategy="rule_policy")
    return cfg


def build_config5() -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.d2.strategy = "evidence_graph"
    cfg.provenance.integrity_mode = "risk_aware"
    cfg.retrieval.enabled = False
    return cfg


def build_config6() -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.d1.strategy = "pattern_graph"
    cfg.d2.strategy = "evidence_graph"
    cfg.d3.strategy = "consensus"
    cfg.d4.strategy = "strict_privacy"
    cfg.d5.strategy = "strict_gate"
    cfg.d5.min_accept_witnesses = 2
    cfg.retrieval = RetrievalConfig(enabled=True, strategy="hubness_cluster")
    return cfg


CONFIG_BUILDERS = {
    "Config-1": build_config1,
    "Config-2": build_config2,
    "Config-3": build_config3,
    "Config-4": build_config4,
    "Config-5": build_config5,
    "Config-6": build_config6,
}


def run_evaluation() -> List[EvalResult]:
    results: List[EvalResult] = []
    for config_name, builder in CONFIG_BUILDERS.items():
        pipeline = ReflectionDefensePipeline.from_config(builder())
        results.extend(_evaluate_write_samples(pipeline, config_name, ALL_SAMPLES))
        results.extend(_evaluate_retrieval_scenarios(pipeline, config_name, RETRIEVAL_SCENARIOS))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _save_raw_results(results, RESULTS_DIR / "eval_results.csv")
    _save_metrics_table(results, RESULTS_DIR / "metrics_table.txt")
    return results


def _evaluate_write_samples(
    pipeline: ReflectionDefensePipeline,
    config_name: str,
    samples: Iterable[ReflectionSample],
) -> List[EvalResult]:
    results = []
    for sample in samples:
        turns = [ConversationTurn(turn_id=sample.sample_id, source=SourceType(sample.source), text=sample.raw_text)]
        candidate = ReflectionCandidate(
            summary_text=sample.summary_candidate,
            fact_texts=[part.strip() for part in sample.summary_candidate.split(".") if part.strip()],
            metadata={"sample_id": sample.sample_id, "subtype": sample.subtype},
        )
        store = MemoryStore()
        started_at = time.perf_counter()
        result = pipeline.evaluate(turns, candidate, store)
        latency_ms = (time.perf_counter() - started_at) * 1000.0
        predicted = "attack" if all(item.action != DecisionAction.ACCEPT for item in result.facts) else "benign"
        decisive_node = result.blocked_by or _last_non_pass_node(result.verdicts)
        results.append(
            EvalResult(
                sample_id=sample.sample_id,
                subtype=sample.subtype,
                path_type="write",
                config_name=config_name,
                expected_label=sample.label,
                predicted_label=predicted,
                decisive_node=decisive_node,
                latency_ms=round(latency_ms, 2),
                detail=",".join(f"{verdict.node}:{verdict.action}" for verdict in result.verdicts),
            )
        )
    return results


def _evaluate_retrieval_scenarios(
    pipeline: ReflectionDefensePipeline,
    config_name: str,
    scenarios: Iterable[RetrievalScenario],
) -> List[EvalResult]:
    results = []
    for scenario in scenarios:
        store = MemoryStore()
        _inject_synthetic_poison_cluster(pipeline, store, scenario)
        started_at = time.perf_counter()
        retrieval = pipeline.on_retrieval(scenario.query, store)
        latency_ms = (time.perf_counter() - started_at) * 1000.0
        predicted = "attack" if retrieval.tampered_count > 0 or any(entry.flagged for entry in retrieval.entries) else "benign"
        decisive_node = _last_non_pass_node(retrieval.verdicts)
        results.append(
            EvalResult(
                sample_id=scenario.scenario_id,
                subtype="RETRIEVAL",
                path_type="retrieval",
                config_name=config_name,
                expected_label=scenario.expected_label,
                predicted_label=predicted,
                decisive_node=decisive_node,
                latency_ms=round(latency_ms, 2),
                detail=",".join(f"{verdict.node}:{verdict.action}" for verdict in retrieval.verdicts),
            )
        )
    return results


def _inject_synthetic_poison_cluster(
    pipeline: ReflectionDefensePipeline,
    store: MemoryStore,
    scenario: RetrievalScenario,
) -> None:
    for index, fact_text in enumerate(scenario.poison_facts, start=1):
        _commit_synthetic_record(
            pipeline=pipeline,
            store=store,
            record_id=f"{scenario.scenario_id}-poison-{index}",
            fact_text=fact_text,
            category=FactCategory.CONTACT if "@" in fact_text else FactCategory.OTHER,
            source_type=SourceType.WEB,
            provenance_score=0.35,
            risk=0.70,
            triggering_query=scenario.query,
        )

    for index, fact_text in enumerate(scenario.benign_facts, start=1):
        _commit_synthetic_record(
            pipeline=pipeline,
            store=store,
            record_id=f"{scenario.scenario_id}-benign-{index}",
            fact_text=fact_text,
            category=FactCategory.CONTACT if "@" in fact_text else FactCategory.PREFERENCE,
            source_type=SourceType.USER,
            provenance_score=0.88,
            risk=0.10,
            triggering_query=scenario.query,
        )


def _commit_synthetic_record(
    pipeline: ReflectionDefensePipeline,
    store: MemoryStore,
    record_id: str,
    fact_text: str,
    category: FactCategory,
    source_type: SourceType,
    provenance_score: float,
    risk: float,
    triggering_query: str,
) -> None:
    record = MemoryRecord(
        record_id=record_id,
        fact_text=fact_text,
        category=category,
        provenance_score=provenance_score,
        evidence_turn_ids=[f"{record_id}-t1"],
        source_summary=fact_text,
    )
    assessment = FactAssessment(
        fact_text=fact_text,
        category=category,
        provenance_score=provenance_score,
        evidence_turn_ids=[f"{record_id}-t1"],
        final_risk=risk,
        source_labels=[
            SourceLabel(
                turn_id=f"{record_id}-t1",
                source_type=source_type.value,
                label=IntegrityLabel.UNTRUSTED if source_type == SourceType.WEB else IntegrityLabel.CANDIDATE,
                selector_score=provenance_score,
                excerpt=fact_text[:120],
            )
        ],
    )
    assessment.verdicts = []
    context = ReflectionContext(
        turns=[ConversationTurn(turn_id=f"{record_id}-t1", source=source_type, text=fact_text)],
        candidate=ReflectionCandidate(summary_text=fact_text, fact_texts=[fact_text]),
        triggering_query=triggering_query,
    )
    pipeline.provenance.bind(record, assessment, context)
    store.commit(record)


def _save_raw_results(results: List[EvalResult], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "subtype",
                "path_type",
                "config_name",
                "expected_label",
                "predicted_label",
                "decisive_node",
                "latency_ms",
                "detail",
            ],
        )
        writer.writeheader()
        for item in results:
            writer.writerow(item.__dict__)


def _save_metrics_table(results: List[EvalResult], path: Path) -> None:
    lines = []
    header = f"{'Config':<10} {'Subtype':<12} {'Prec':>6} {'Rec':>6} {'F1':>6} {'FPR':>6}"
    lines.append(header)
    lines.append("-" * len(header))
    for config_name in CONFIG_BUILDERS:
        for subtype in ["RS-1", "RS-2", "RS-3", "RS-4", "RETRIEVAL", "ALL_ATTACK", "ALL"]:
            subset = [item for item in results if item.config_name == config_name and _match_subtype(item, subtype)]
            if not subset:
                continue
            precision, recall, f1, fpr = _metrics(subset)
            lines.append(f"{config_name:<10} {subtype:<12} {precision:>6.3f} {recall:>6.3f} {f1:>6.3f} {fpr:>6.3f}")
        lines.append("-" * len(header))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _match_subtype(item: EvalResult, subtype: str) -> bool:
    if subtype == "ALL":
        return True
    if subtype == "ALL_ATTACK":
        return item.expected_label == "attack"
    return item.subtype == subtype


def _metrics(results: List[EvalResult]) -> tuple[float, float, float, float]:
    tp = sum(1 for item in results if item.expected_label == "attack" and item.predicted_label == "attack")
    tn = sum(1 for item in results if item.expected_label != "attack" and item.predicted_label != "attack")
    fp = sum(1 for item in results if item.expected_label != "attack" and item.predicted_label == "attack")
    fn = sum(1 for item in results if item.expected_label == "attack" and item.predicted_label != "attack")
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return precision, recall, f1, fpr


def _last_non_pass_node(verdicts) -> str:
    for verdict in reversed(verdicts):
        if verdict.action != "PASS":
            return verdict.node
    return "PASS"


if __name__ == "__main__":
    run_evaluation()
