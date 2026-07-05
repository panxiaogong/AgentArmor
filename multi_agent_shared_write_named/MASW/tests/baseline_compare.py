"""Horizontal baseline comparison for MASW.

The experiment runs MASW and external-style baselines on the same dataset:

- MASW_ALL_RECOMMENDED: D3 HybridDetector + D4 RuleBasedDetector + D5/D6.
- DEEPSEEK_V4_FLASH: remote LLM-as-judge baseline, disabled unless explicitly
  configured through environment variables or local `.env`.
- PROMPTINJECT_STYLE_TOOL: narrow prompt-injection keyword guard.
- LLM_GUARD_STYLE_TOOL: multi-scanner local guard approximation.
- REBUFF_STYLE_TOOL: layered signature/canary-style local guard approximation.

Prediction definition:
Positive means the detector judges the row as an attack. Metrics are computed
against the dataset label, not against MASW's decision.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import median

from MASW.baselines.common import BaselineDetector, BaselinePrediction
from MASW.baselines.deepseek import DeepSeekV4FlashBaseline
from MASW.baselines.local_tools import local_tool_baselines
from MASW.tests.ablation_eval import _metrics, _percentile
from MASW.tests.tech_selection_eval import SelectionConfig, _run_sample, build_selection_samples


RESULTS_DIR = Path(__file__).resolve().parent / "results"
BASELINE_METRICS_CSV_PATH = RESULTS_DIR / "masw_baseline_comparison_metrics.csv"
BASELINE_RESULTS_JSON_PATH = RESULTS_DIR / "masw_baseline_comparison_results.json"
BASELINE_REPORT_PATH = RESULTS_DIR / "masw_baseline_comparison_report.md"


MASW_RECOMMENDED_CONFIG = SelectionConfig(
    name="MASW_ALL_RECOMMENDED",
    d3_detector="hybrid",
    d4_detector="rule",
    include_d5_d6=True,
)


def baseline_detectors() -> list[BaselineDetector]:
    """Return non-MASW baseline detectors.

    DeepSeek is included first but may report `skipped=True` if the local user
    has not opted into remote execution.
    """

    return [DeepSeekV4FlashBaseline(), *local_tool_baselines()]


def _masw_prediction(sample: dict[str, object]) -> BaselinePrediction:
    outcome = _run_sample(MASW_RECOMMENDED_CONFIG, sample)
    return BaselinePrediction(
        baseline=MASW_RECOMMENDED_CONFIG.name,
        predicted_attack=bool(outcome["predicted_attack"]),
        risk=1.0 if bool(outcome["predicted_attack"]) else 0.0,
        reason=str(outcome["blocked_stage"]),
        elapsed_ms=float(outcome["elapsed_ms"]),
    )


def _row_from_prediction(sample: dict[str, object], prediction: BaselinePrediction) -> dict[str, object]:
    return {
        "baseline": prediction.baseline,
        "id": sample["id"],
        "label": sample["label"],
        "category": sample["category"],
        "predicted_attack": prediction.predicted_attack,
        "risk": prediction.risk,
        "reason": prediction.reason,
        "elapsed_ms": prediction.elapsed_ms,
        "skipped": prediction.skipped,
    }


def _metric_row(baseline_name: str, rows: list[dict[str, object]]) -> dict[str, object]:
    evaluated_rows = [row for row in rows if not bool(row["skipped"])]
    skipped_count = len(rows) - len(evaluated_rows)

    if not evaluated_rows:
        return {
            "baseline": baseline_name,
            "status": "skipped",
            "evaluated_samples": 0,
            "skipped_samples": skipped_count,
            "precision": None,
            "recall": None,
            "f1": None,
            "fpr": None,
            "tp": None,
            "fp": None,
            "fn": None,
            "tn": None,
            "p50_ms": None,
            "p95_ms": None,
            "p99_ms": None,
        }

    metrics = _metrics(evaluated_rows, include_latency=True)
    elapsed = [float(row["elapsed_ms"]) for row in evaluated_rows]
    metrics["p50_ms"] = median(elapsed)
    metrics["p95_ms"] = _percentile(elapsed, 95)
    metrics["p99_ms"] = _percentile(elapsed, 99)

    status = "ok" if skipped_count == 0 else "partial"
    return {
        "baseline": baseline_name,
        "status": status,
        "evaluated_samples": len(evaluated_rows),
        "skipped_samples": skipped_count,
        **metrics,
    }


def _write_metrics_csv(metrics_rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "baseline",
        "status",
        "evaluated_samples",
        "skipped_samples",
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
    with BASELINE_METRICS_CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_rows)


def _fmt(value: object) -> str:
    if value is None:
        return "skipped"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _build_report(metrics_rows: list[dict[str, object]], total_samples: int) -> str:
    lines = [
        "# MASW Baseline Comparison Report",
        "",
        "## Dataset",
        "",
        f"- Total samples: {total_samples}",
        "- Source: `build_selection_samples()` = MASW minimum dataset plus subtle sync/exfiltration hard set.",
        "- Labels: attack vs benign.",
        "",
        "## Baselines",
        "",
        "- `MASW_ALL_RECOMMENDED`: our system, using D3 HybridDetector + D4 RuleBasedDetector + D5/D6.",
        "- `DEEPSEEK_V4_FLASH`: remote LLM-as-judge baseline. It is skipped unless local env enables it.",
        "- `PROMPTINJECT_STYLE_TOOL`: prompt injection keyword guard.",
        "- `LLM_GUARD_STYLE_TOOL`: local multi-scanner guard approximation.",
        "- `REBUFF_STYLE_TOOL`: local layered signature/canary-style guard approximation.",
        "",
        "## Metrics",
        "",
        "| Baseline | Status | Precision | Recall | F1 | FPR | P95 ms |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in metrics_rows:
        lines.append(
            f"| `{row['baseline']}` | {row['status']} | {_fmt(row['precision'])} | {_fmt(row['recall'])} | "
            f"{_fmt(row['f1'])} | {_fmt(row['fpr'])} | {_fmt(row['p95_ms'])} |"
        )

    deepseek_row = next((row for row in metrics_rows if row["baseline"] == "DEEPSEEK_V4_FLASH"), None)
    if deepseek_row is not None and deepseek_row["status"] == "skipped":
        lines.extend(
            [
                "",
                "## DeepSeek Run Note",
                "",
                "DeepSeek was not executed in this run. To enable it locally, create `.env` from `.env.example`, "
                "set `DEEPSEEK_API_KEY`, and set `MASW_RUN_REMOTE_BASELINES=1`.",
            ]
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The local tool baselines are deterministic adapters for comparison. They are intentionally weaker than a full MASW pipeline because they only classify text; they do not enforce provenance, trust promotion, retrieval filtering, or action mediation.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_baseline_comparison() -> dict[str, object]:
    samples = build_selection_samples()
    by_baseline: dict[str, list[dict[str, object]]] = {}
    metrics_rows: list[dict[str, object]] = []

    masw_rows: list[dict[str, object]] = []
    for sample in samples:
        masw_rows.append(_row_from_prediction(sample, _masw_prediction(sample)))
    by_baseline[MASW_RECOMMENDED_CONFIG.name] = masw_rows
    metrics_rows.append(_metric_row(MASW_RECOMMENDED_CONFIG.name, masw_rows))

    for detector in baseline_detectors():
        rows: list[dict[str, object]] = []
        for sample in samples:
            rows.append(_row_from_prediction(sample, detector.predict(sample)))
        by_baseline[detector.name] = rows
        metrics_rows.append(_metric_row(detector.name, rows))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_metrics_csv(metrics_rows)
    BASELINE_REPORT_PATH.write_text(_build_report(metrics_rows, len(samples)), encoding="utf-8")
    payload = {"metrics": metrics_rows, "rows": by_baseline}
    BASELINE_RESULTS_JSON_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    report = run_baseline_comparison()
    print(json.dumps({"metrics": report["metrics"]}, ensure_ascii=False, indent=2))

