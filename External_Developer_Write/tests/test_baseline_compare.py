"""
test_baseline_compare.py — Type 4 基线横向对比测试

将 200 文档数据集跑在 3 个外部基线上，与 SP1-SP7 消融结果对比。

Baselines:
  1. KeywordRuleBaseline — Regex 关键词匹配
  2. PerplexityBaseline  — 字符 n-gram 困惑度
  3. DeepSeekBaseline    — LLM-as-judge API

指标: Precision / Recall / F1 / FPR / latency
"""

import os
import sys
import pytest
from typing import Dict

from External_Developer_Write.tests.baseline_compare import (
    build_all_documents,
    build_ground_truth,
    build_attack_type_map,
    KeywordRuleBaseline,
    PerplexityBaseline,
    DeepSeekBaseline,
    run_baseline_comparison,
    generate_baseline_report,
    skip_if_no_api,
    BaselineResult,
    REPORT_DIR,
    BASELINE_REPORT_PATH,
)


class TestBaselineCompare:
    """基线横向对比测试套件。"""

    @pytest.fixture(scope="class")
    def dataset(self):
        """构建完整 200 文档数据集。"""
        docs = build_all_documents()
        gt = build_ground_truth(docs)
        benign_docs = [d for d in docs if not gt[d.doc_id]]
        return docs, gt, benign_docs

    # ── 数据完整性 ───────────────────────────────────────────

    def test_dataset_integrity(self, dataset):
        """验证数据集完整性：200 文档（120 attack + 80 benign）。"""
        docs, gt, benign = dataset
        assert len(docs) == 200, f"Expected 200 docs, got {len(docs)}"
        n_attack = sum(1 for v in gt.values() if v)
        n_benign = sum(1 for v in gt.values() if not v)
        assert n_attack == 120, f"Expected 120 attack, got {n_attack}"
        assert n_benign == 80, f"Expected 80 benign, got {n_benign}"
        assert len(benign) == 80, f"Expected 80 benign docs, got {len(benign)}"

    # ── 基线 1: Keyword Rule ────────────────────────────────

    def test_keyword_baseline(self, dataset):
        """BC-01: 关键词规则基线。"""
        docs, gt, _ = dataset
        kw = KeywordRuleBaseline()
        result = kw.evaluate(docs, gt)

        print(f"\n[BC-01] Keyword Rule Baseline:")
        print(f"  Precision={result.metrics['precision']:.4f}")
        print(f"  Recall={result.metrics['recall']:.4f}")
        print(f"  F1={result.metrics['f1']:.4f}")
        print(f"  FPR={result.metrics['fpr']:.4f}")
        print(f"  TP={result.metrics['tp']}, FP={result.metrics['fp']}, "
              f"TN={result.metrics['tn']}, FN={result.metrics['fn']}")
        print(f"  P50={result.latency_stats['p50']:.4f}ms")

        # 关键词规则至少能检测一部分攻击
        assert result.metrics["precision"] >= 0.8, "关键词基线精确率异常偏低"
        assert result.metrics["tp"] + result.metrics["fn"] > 0

        # 打印分类型
        print(f"  Per-type F1:")
        for atype in sorted(result.per_type_metrics.keys()):
            f1 = result.per_type_metrics[atype]["f1"]
            print(f"    {atype}: {f1:.4f}")

    # ── 基线 2: Perplexity ─────────────────────────────────

    def test_perplexity_baseline(self, dataset):
        """BC-02: 困惑度基线。"""
        docs, gt, benign_docs = dataset
        ppl = PerplexityBaseline(n=5)
        ppl.train(benign_docs)

        # 打印训练信息
        info = ppl._details
        print(f"\n[BC-02] Perplexity Training:")
        print(f"  Training docs: {info['train_docs']}")
        print(f"  Vocab size: {info['vocab_size']}")
        print(f"  Unique n-grams: {info['unique_ngrams']}")
        print(f"  Mean perplexity: {info['mean_perplexity']:.4f}")
        print(f"  Std perplexity: {info['std_perplexity']:.4f}")
        print(f"  Threshold (mean+2*std): {info['threshold']:.4f}")

        result = ppl.evaluate(docs, gt)
        print(f"\n[BC-02] Perplexity Baseline:")
        print(f"  Precision={result.metrics['precision']:.4f}")
        print(f"  Recall={result.metrics['recall']:.4f}")
        print(f"  F1={result.metrics['f1']:.4f}")
        print(f"  FPR={result.metrics['fpr']:.4f}")
        print(f"  TP={result.metrics['tp']}, FP={result.metrics['fp']}, "
              f"TN={result.metrics['tn']}, FN={result.metrics['fn']}")

        # 困惑度基线应对大部分攻击有效（攻击文本包含非常规字符模式）
        assert result.metrics["f1"] >= 0.5, "困惑度基线 F1 异常偏低"

        # 打印分类型
        print(f"  Per-type F1:")
        for atype in sorted(result.per_type_metrics.keys()):
            f1 = result.per_type_metrics[atype]["f1"]
            print(f"    {atype}: {f1:.4f}")

    # ── 基线 3: DeepSeek ───────────────────────────────────

    def test_deepseek_baseline(self, dataset):
        """BC-03: DeepSeek LLM-as-judge 基线。

        需要 API key（从 .env 读取），无 key 时 skip。
        """
        if skip_if_no_api():
            pytest.skip("DEEPSEEK_API_KEY 未配置，跳过 DeepSeek 基线测试")

        docs, gt, _ = dataset

        # 使用子集测试（全部 200 文档，API 调用全部）
        dsk = DeepSeekBaseline()
        result = dsk.evaluate(docs, gt)

        print(f"\n[BC-03] DeepSeek Baseline:")
        print(f"  Model: {dsk.model}")
        print(f"  Precision={result.metrics['precision']:.4f}")
        print(f"  Recall={result.metrics['recall']:.4f}")
        print(f"  F1={result.metrics['f1']:.4f}")
        print(f"  FPR={result.metrics['fpr']:.4f}")
        print(f"  TP={result.metrics['tp']}, FP={result.metrics['fp']}, "
              f"TN={result.metrics['tn']}, FN={result.metrics['fn']}")

        # DeepSeek 应能识别大部分攻击
        assert result.metrics["f1"] >= 0.3, "DeepSeek 基线 F1 异常偏低"

    # ── 统一运行（不含 DeepSeek） ──────────────────────────

    def test_run_all_baselines(self, dataset):
        """BC-04: 统一运行所有基线（不含 DeepSeek API）。"""
        docs, gt, benign = dataset

        # 运行关键词基线
        kw = KeywordRuleBaseline()
        r_kw = kw.evaluate(docs, gt)
        assert isinstance(r_kw, BaselineResult)
        assert r_kw.metrics["f1"] > 0

        # 运行困惑度基线
        ppl = PerplexityBaseline(n=5)
        ppl.train(benign)
        r_ppl = ppl.evaluate(docs, gt)
        assert isinstance(r_ppl, BaselineResult)
        assert r_ppl.metrics["f1"] > 0

        print(f"\n[BC-04] 所有基线运行完成:")
        print(f"  Keyword Rule: F1={r_kw.metrics['f1']:.4f}")
        print(f"  Perplexity: F1={r_ppl.metrics['f1']:.4f}")

    # ── 报告生成 ───────────────────────────────────────────

    def test_baseline_report(self, dataset):
        """BC-05: 生成基线对比报告。"""
        docs, gt, benign = dataset

        # 计算所有基线结果
        kw = KeywordRuleBaseline()
        r_kw = kw.evaluate(docs, gt)

        ppl = PerplexityBaseline(n=5)
        ppl.train(benign)
        r_ppl = ppl.evaluate(docs, gt)

        results = {
            kw.name: r_kw,
            ppl.name: r_ppl,
        }

        # 尝试 DeepSeek（如果可用）
        if not skip_if_no_api():
            try:
                dsk = DeepSeekBaseline()
                r_dsk = dsk.evaluate(docs, gt)
                results[dsk.name] = r_dsk
            except Exception as e:
                print(f"  DeepSeek skipped during report: {e}")

        report = generate_baseline_report(results)

        assert os.path.exists(BASELINE_REPORT_PATH), "报告文件未生成"
        assert len(report) > 500, "报告内容过短"
        assert "# Type 4 基线对比评估报告" in report, "报告缺少标题"

        print(f"\n[BC-05] 基线对比报告已生成: {BASELINE_REPORT_PATH}")
        print(f"  大小: {len(report)} 字符")

        # 打印对比表摘要
        print(f"\n  对比摘要:")
        for name in sorted(results.keys()):
            r = results[name]
            print(f"    {name}: F1={r.metrics['f1']:.4f}, "
                  f"P={r.metrics['precision']:.4f}, "
                  f"R={r.metrics['recall']:.4f}, "
                  f"FPR={r.metrics['fpr']:.4f}, "
                  f"P50={r.latency_stats['p50']:.4f}ms")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
