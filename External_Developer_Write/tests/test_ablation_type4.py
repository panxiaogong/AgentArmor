"""
test_ablation_type4.py — Type 4 消融评估测试

执行完整的消融评估实验:
  1. 单节点消融 (TA-01 ~ TA-12)
  2. 组合配置对比 (CF-01 ~ CF-05)
  3. 延迟基准
  4. 生成 Markdown 报告

输出: pytest 终端 + ablation_report_type4.md
"""

import os
import pytest
from typing import List, Dict, Tuple

from External_Developer_Write.types import (
    Document, DocumentEmbedding, ChunkInfo,
)
from External_Developer_Write.tests.ablation_evaluation import AblationEvaluator
from External_Developer_Write.tests.csv_dataset_exporter import export_datasets_to_csv


class TestType4Ablation:
    """Type 4 完整消融评估实验套件。"""

    @pytest.fixture(autouse=True)
    def setup(
        self,
        benign_docs, benign_embeddings,
        poisonedrag_docs, poisonedrag_embeddings,
        agentpoison_docs, agentpoison_embeddings,
        prompt_injection_docs, tool_misuse_docs,
        memory_poisoning_docs, agent_hijacking_docs,
        enhanced_poisonedrag_docs, enhanced_agentpoison_docs,
        enhanced_semantic_confusion_groups,
        hybrid_attack_docs, extra_benign_docs,
        benign_chunk_groups, semantic_confusion_groups,
        mock_models,
    ):
        self.evaluator = AblationEvaluator(mock_models)
        self.benign_docs = benign_docs
        self.benign_embeds = benign_embeddings
        self.poisonedrag_embeds = poisonedrag_embeddings
        self.agentpoison_embeds = agentpoison_embeddings
        self.pi_docs = prompt_injection_docs
        self.tm_docs = tool_misuse_docs
        self.mp_docs = memory_poisoning_docs
        self.ah_docs = agent_hijacking_docs
        self.epr_docs = enhanced_poisonedrag_docs
        self.eap_docs = enhanced_agentpoison_docs
        self.semconf_groups = semantic_confusion_groups
        self.enhanced_semconf = enhanced_semantic_confusion_groups
        self.hybrid_docs = hybrid_attack_docs
        self.extra_benign = extra_benign_docs
        self.benign_chunks = benign_chunk_groups

    # ═══════════════════════════════════════════════════════════════
    # CSV Dataset Export
    # ═══════════════════════════════════════════════════════════════

    def test_export_csv_dataset(self):
        """导出结构化 CSV 数据集。"""
        csv_dir = os.path.join(os.path.dirname(__file__), "datasets")
        os.makedirs(csv_dir, exist_ok=True)
        csv_path = os.path.join(csv_dir, "type4_ablation_dataset.csv")
        result = export_datasets_to_csv(csv_path)
        assert os.path.exists(result), f"CSV 文件未生成: {result}"
        print(f"\n[CSV] 数据集已导出: {result}")

    # ═══════════════════════════════════════════════════════════════
    # SP1: Embedding Anomaly Detection
    # ═══════════════════════════════════════════════════════════════

    def test_ta_01_sp1_vs_poisonedrag(self):
        """TA-01: SP1 vs PoisonedRAG (basic)。"""
        results = self.evaluator.evaluate_sp1(
            self.benign_embeds, self.poisonedrag_embeds
        )
        r = results[0]
        print(f"\n[TA-01] SP1 vs PoisonedRAG (basic):")
        print(f"  Precision={r.metrics['precision']:.4f}, "
              f"Recall={r.metrics['recall']:.4f}, "
              f"F1={r.metrics['f1']:.4f}, FPR={r.metrics['fpr']:.4f}")
        assert r.metrics["f1"] >= 0.0  # 信息性输出，不硬性要求

    def test_ta_02_sp1_vs_enhanced_pr(self):
        """TA-02: SP1 vs PoisonedRAG (enhanced)。"""
        results = self.evaluator.evaluate_sp1(
            self.benign_embeds, self.poisonedrag_embeds
        )
        r = results[1]
        print(f"\n[TA-02] SP1 vs PoisonedRAG (enhanced):")
        print(f"  Precision={r.metrics['precision']:.4f}, "
              f"Recall={r.metrics['recall']:.4f}, "
              f"F1={r.metrics['f1']:.4f}, FPR={r.metrics['fpr']:.4f}")

    # ═══════════════════════════════════════════════════════════════
    # SP2: Content Perplexity
    # ═══════════════════════════════════════════════════════════════

    def test_ta_03_sp2_vs_prompt_injection(self):
        """TA-03: SP2 vs Prompt Injection (30 samples)。"""
        r = self.evaluator.evaluate_sp2(self.benign_docs, self.pi_docs)
        print(f"\n[TA-03] SP2 vs Prompt Injection:")
        print(f"  Precision={r.metrics['precision']:.4f}, "
              f"Recall={r.metrics['recall']:.4f}, "
              f"F1={r.metrics['f1']:.4f}, FPR={r.metrics['fpr']:.4f}")
        print(f"  Subtype breakdown:")
        for st_name, st_m in sorted(r.per_subtype_metrics.items()):
            print(f"    {st_name}: F1={st_m['f1']:.3f}, "
                  f"P={st_m['precision']:.3f}, R={st_m['recall']:.3f}")

    # ═══════════════════════════════════════════════════════════════
    # SP3: Cross-Chunk Coherence
    # ═══════════════════════════════════════════════════════════════

    def test_ta_04_sp3_vs_semconf_basic(self):
        """TA-04: SP3 vs Semantic Confusion (basic)。"""
        results = self.evaluator.evaluate_sp3(
            self.benign_chunks, self.semconf_groups, self.enhanced_semconf
        )
        if results:
            r = results[0]
            print(f"\n[TA-04] SP3 vs Semantic Confusion (basic):")
            print(f"  Precision={r.metrics['precision']:.4f}, "
                  f"Recall={r.metrics['recall']:.4f}, "
                  f"F1={r.metrics['f1']:.4f}, FPR={r.metrics['fpr']:.4f}")

    def test_ta_05_sp3_vs_semconf_enhanced(self):
        """TA-05: SP3 vs Semantic Confusion (enhanced)。"""
        results = self.evaluator.evaluate_sp3(
            self.benign_chunks, self.semconf_groups, self.enhanced_semconf
        )
        if len(results) > 1:
            r = results[1]
            print(f"\n[TA-05] SP3 vs Semantic Confusion (enhanced):")
            print(f"  Precision={r.metrics['precision']:.4f}, "
                  f"Recall={r.metrics['recall']:.4f}, "
                  f"F1={r.metrics['f1']:.4f}, FPR={r.metrics['fpr']:.4f}")

    # ═══════════════════════════════════════════════════════════════
    # SP4: Trigger Region Detection
    # ═══════════════════════════════════════════════════════════════

    def test_ta_06_sp4_vs_agentpoison_basic(self):
        """TA-06: SP4 vs AgentPoison (basic)。"""
        results = self.evaluator.evaluate_sp4(
            self.benign_embeds, self.agentpoison_embeds
        )
        r = results[0]
        print(f"\n[TA-06] SP4 vs AgentPoison (basic):")
        print(f"  Precision={r.metrics['precision']:.4f}, "
              f"Recall={r.metrics['recall']:.4f}, "
              f"F1={r.metrics['f1']:.4f}, FPR={r.metrics['fpr']:.4f}")

    def test_ta_07_sp4_vs_agentpoison_enhanced(self):
        """TA-07: SP4 vs AgentPoison (enhanced)。"""
        results = self.evaluator.evaluate_sp4(
            self.benign_embeds, self.agentpoison_embeds
        )
        r = results[1]
        print(f"\n[TA-07] SP4 vs AgentPoison (enhanced):")
        print(f"  Precision={r.metrics['precision']:.4f}, "
              f"Recall={r.metrics['recall']:.4f}, "
              f"F1={r.metrics['f1']:.4f}, FPR={r.metrics['fpr']:.4f}")

    # ═══════════════════════════════════════════════════════════════
    # SP5: Robust Aggregation
    # ═══════════════════════════════════════════════════════════════

    def test_ta_08_sp5_vs_tool_misuse(self):
        """TA-08: SP5 vs Tool Misuse。"""
        r = self.evaluator.evaluate_sp5(self.tm_docs)
        print(f"\n[TA-08] SP5 vs Tool Misuse:")
        print(f"  Keyword Suppression Rate: "
              f"{r.metrics.get('keyword_suppression_rate', 'N/A')}")
        print(f"  Robust (benign-only): "
              f"{r.details.get('robust_benign_only', 'N/A')}")
        print(f"  Robust (minority attack): "
              f"{r.details.get('robust_minority_attack', 'N/A')}")
        print(f"  Robust (majority attack): "
              f"{r.details.get('robust_majority_attack', 'N/A')}")

    # ═══════════════════════════════════════════════════════════════
    # SP6: Post-Retrieval Verifier
    # ═══════════════════════════════════════════════════════════════

    def test_ta_09_sp6_vs_memory_poisoning(self):
        """TA-09: SP6 vs Memory Poisoning。"""
        results = self.evaluator.evaluate_sp6(self.mp_docs, self.ah_docs)
        r = results[0]
        print(f"\n[TA-09] SP6 vs Memory Poisoning:")
        print(f"  Precision={r.metrics['precision']:.4f}, "
              f"Recall={r.metrics['recall']:.4f}, "
              f"F1={r.metrics['f1']:.4f}, FPR={r.metrics['fpr']:.4f}")

    def test_ta_10_sp6_vs_agent_hijacking(self):
        """TA-10: SP6 vs Agent Hijacking。"""
        results = self.evaluator.evaluate_sp6(self.mp_docs, self.ah_docs)
        r = results[1]
        print(f"\n[TA-10] SP6 vs Agent Hijacking:")
        print(f"  Precision={r.metrics['precision']:.4f}, "
              f"Recall={r.metrics['recall']:.4f}, "
              f"F1={r.metrics['f1']:.4f}, FPR={r.metrics['fpr']:.4f}")

    # ═══════════════════════════════════════════════════════════════
    # SP7: Semantic Dependency Graph
    # ═══════════════════════════════════════════════════════════════

    def test_ta_11_sp7_vs_semantic_confusion(self):
        """TA-11: SP7 vs Semantic Confusion (cross-doc)。"""
        results = self.evaluator.evaluate_sp7(
            self.benign_docs, self.enhanced_semconf, self.hybrid_docs
        )
        if results:
            r = results[0]
            print(f"\n[TA-11] SP7 vs Semantic Confusion:")
            print(f"  Precision={r.metrics['precision']:.4f}, "
                  f"Recall={r.metrics['recall']:.4f}, "
                  f"F1={r.metrics['f1']:.4f}, FPR={r.metrics['fpr']:.4f}")
            print(f"  Benign density={r.details.get('benign_density', 'N/A')}, "
                  f"Attack density={r.details.get('attack_density', 'N/A')}")

    def test_ta_12_sp7_vs_hybrid(self):
        """TA-12: SP7 vs Hybrid Attacks。"""
        results = self.evaluator.evaluate_sp7(
            self.benign_docs, self.enhanced_semconf, self.hybrid_docs
        )
        if len(results) > 1:
            r = results[1]
            print(f"\n[TA-12] SP7 vs Hybrid Attacks:")
            print(f"  Precision={r.metrics['precision']:.4f}, "
                  f"Recall={r.metrics['recall']:.4f}, "
                  f"F1={r.metrics['f1']:.4f}, FPR={r.metrics['fpr']:.4f}")

    # ═══════════════════════════════════════════════════════════════
    # 组合配置评估
    # ═══════════════════════════════════════════════════════════════

    def test_cf_all_configs(self):
        """CF-01~05: 全部 5 种配置组合对比。"""
        results = self.evaluator.evaluate_configs(
            self.benign_docs, self.benign_embeds,
            self.pi_docs, self.epr_docs,
            self.tm_docs, self.mp_docs, self.ah_docs,
        )
        print(f"\n{'='*60}")
        print(f"  [CF] 组合配置评估")
        print(f"{'='*60}")
        for cr in results:
            print(f"\n  {cr.config_name} ({cr.config_label}):")
            print(f"    活跃节点: {len(cr.active_sps)} SPs")
            print(f"    综合告警率: {cr.overall_metrics.get('overall_alert_rate', 0):.2%}")
            for atk, rate in cr.attack_coverage.items():
                print(f"    {atk}: {rate:.0%}")
            if cr.latency_ms:
                print(f"    延迟: P50={cr.latency_ms.get('p50', '-')}ms, "
                      f"P95={cr.latency_ms.get('p95', '-')}ms")

    # ═══════════════════════════════════════════════════════════════
    # 延迟基准
    # ═══════════════════════════════════════════════════════════════

    def test_latency_benchmark(self):
        """延迟基准测试。"""
        benchmarks = self.evaluator.benchmark_latency(
            self.benign_embeds, self.benign_docs, n_iter=30
        )
        print(f"\n{'='*60}")
        print(f"  [LATENCY] 延迟基准")
        print(f"{'='*60}")
        print(f"  {'Operation':<20} {'Mean(ms)':<10} {'P50':<10} {'P95':<10} {'P99':<10}")
        print(f"  {'-'*60}")
        for b in benchmarks:
            print(f"  {b.sp_name:<20} {b.mean_ms:<10.4f} {b.p50_ms:<10.4f} "
                  f"{b.p95_ms:<10.4f} {b.p99_ms:<10.4f}")

    # ═══════════════════════════════════════════════════════════════
    # 报告生成
    # ═══════════════════════════════════════════════════════════════

    def test_generate_report(self):
        """生成完整的 Markdown 消融评估报告。"""
        # 确保数据已被收集
        self.evaluator.evaluate_sp1(self.benign_embeds, self.poisonedrag_embeds)
        self.evaluator.evaluate_sp2(self.benign_docs, self.pi_docs)
        self.evaluator.evaluate_sp3(
            self.benign_chunks, self.semconf_groups, self.enhanced_semconf
        )
        self.evaluator.evaluate_sp4(self.benign_embeds, self.agentpoison_embeds)
        self.evaluator.evaluate_sp5(self.tm_docs)
        self.evaluator.evaluate_sp6(self.mp_docs, self.ah_docs)
        self.evaluator.evaluate_sp7(
            self.benign_docs, self.enhanced_semconf, self.hybrid_docs
        )
        self.evaluator.evaluate_configs(
            self.benign_docs, self.benign_embeds,
            self.pi_docs, self.epr_docs,
            self.tm_docs, self.mp_docs, self.ah_docs,
        )
        self.evaluator.benchmark_latency(
            self.benign_embeds, self.benign_docs, n_iter=30
        )

        report_dir = os.path.join(os.path.dirname(__file__), "reports")
        report_path = os.path.join(report_dir, "ablation_report_type4.md")
        report = self.evaluator.generate_markdown_report(report_path)

        assert os.path.exists(report_path), f"报告文件未生成: {report_path}"

        # 打印摘要到终端
        print(f"\n{'='*60}")
        print(f"  [REPORT] 消融评估报告已生成")
        print(f"{'='*60}")
        print(f"  文件: {report_path}")
        print(f"  大小: {len(report)} 字符")
        print(f"\n  最后 10 行:")
        report_lines = report.strip().split("\n")
        for line in report_lines[-10:]:
            print(f"    {line}")
