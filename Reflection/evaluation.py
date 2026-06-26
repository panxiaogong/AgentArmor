from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from Reflection.config import D1Config, D2Config, D3Config, D4Config, D5Config, PipelineConfig
from Reflection.memory_store import MemoryStore
from Reflection.pipeline import ReflectionDefensePipeline
from Reflection.types import ConversationTurn, DecisionAction, ReflectionCandidate, SourceType
from Reflection.utils import split_sentences


@dataclass
class DatasetSample:
    """最小评测样本格式，和 seed CSV 一一对应。"""

    sample_id: str
    source: SourceType
    raw_text: str
    summary_candidate: str
    label: str
    attack_goal: str
    notes: str


def load_dataset(path: str | Path) -> List[DatasetSample]:
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            DatasetSample(
                sample_id=row["sample_id"],
                source=SourceType(row["source"]),
                raw_text=row["raw_text"],
                summary_candidate=row["summary_candidate"],
                label=row["label"],
                attack_goal=row["attack_goal"],
                notes=row["notes"],
            )
            for row in reader
        ]


def compute_metrics(gold_attack: Sequence[bool], predicted_attack: Sequence[bool], latencies_ms: Sequence[float]) -> Dict[str, float]:
    """计算作品报告里常用的拦截效果指标。"""

    tp = sum(1 for gold, pred in zip(gold_attack, predicted_attack) if gold and pred)
    tn = sum(1 for gold, pred in zip(gold_attack, predicted_attack) if not gold and not pred)
    fp = sum(1 for gold, pred in zip(gold_attack, predicted_attack) if not gold and pred)
    fn = sum(1 for gold, pred in zip(gold_attack, predicted_attack) if gold and not pred)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    sorted_latencies = sorted(latencies_ms) if latencies_ms else [0.0]
    return {
        "Prec": precision,
        "Rec": recall,
        "F1": f1,
        "FPR": fpr,
        "P50ms": _percentile(sorted_latencies, 0.50),
        "P95ms": _percentile(sorted_latencies, 0.95),
        "P99ms": _percentile(sorted_latencies, 0.99),
        "TP": float(tp),
        "TN": float(tn),
        "FP": float(fp),
        "FN": float(fn),
    }


def build_ablation_configs() -> Dict[str, PipelineConfig]:
    """构造从无防护到全量防护的最小消融配置。"""

    return {
        "Config-1-unsafe": PipelineConfig.unsafe(),
        "Config-2-injection-only": PipelineConfig(
            d1=D1Config(enabled=True),
            d2=D2Config(enabled=False, min_support_score=0.0, min_provenance_score=0.0),
            d3=D3Config(enabled=False),
            d4=D4Config(enabled=False),
            d5=D5Config(enabled=True),
        ),
        "Config-3-injection-provenance": PipelineConfig(
            d1=D1Config(enabled=True),
            d2=D2Config(enabled=True),
            d3=D3Config(enabled=False),
            d4=D4Config(enabled=False),
            d5=D5Config(enabled=True),
        ),
        "Config-4-with-consistency": PipelineConfig(
            d1=D1Config(enabled=True),
            d2=D2Config(enabled=True),
            d3=D3Config(enabled=True),
            d4=D4Config(enabled=False),
            d5=D5Config(enabled=True),
        ),
        "Config-5-full": PipelineConfig(),
    }


def evaluate_dataset(pipeline: ReflectionDefensePipeline, samples: Iterable[DatasetSample]) -> Dict[str, float]:
    gold_attack: List[bool] = []
    predicted_attack: List[bool] = []
    latencies_ms: List[float] = []

    for sample in samples:
        turns = [ConversationTurn(turn_id=sample.sample_id, source=sample.source, text=sample.raw_text)]
        candidate = ReflectionCandidate(
            summary_text=sample.summary_candidate,
            fact_texts=split_sentences(sample.summary_candidate),
            metadata={"sample_id": sample.sample_id},
        )
        start = time.perf_counter()
        result = pipeline.evaluate(turns, candidate, MemoryStore())
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

        attack = sample.label.lower() == "attack"
        # 只要本条样本下所有事实都没被 ACCEPT，就把它视为“防御成功阻断”。
        blocked = all(assessment.action != DecisionAction.ACCEPT for assessment in result.facts)
        gold_attack.append(attack)
        predicted_attack.append(blocked)

    return compute_metrics(gold_attack, predicted_attack, latencies_ms)


def _percentile(values: Sequence[float], q: float) -> float:
    """小样本场景下的轻量百分位实现，足够支持当前 seed dataset。"""

    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    index = round(q * (len(values) - 1))
    return float(values[index])
