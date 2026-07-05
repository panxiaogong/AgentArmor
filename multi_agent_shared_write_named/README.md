# Multi-Agent Shared Write

源码放在 `MASW/` 包内。这样既保留参考图里的 `types.py` 命名，又避免顶层
`types.py` 遮蔽 Python 标准库导致测试无法运行。

## 阶段报告

当前阶段设计与测试记录：

```text
MASW/report_stage_current.md
```

论文风格架构图集：

```text
MASW/figures/
```

## 测试与评估

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s MASW/tests
PYTHONDONTWRITEBYTECODE=1 python3 -m MASW.tests.build_dataset
PYTHONDONTWRITEBYTECODE=1 python3 -m MASW.tests.eval_masw
PYTHONDONTWRITEBYTECODE=1 python3 -m MASW.tests.ablation_eval
PYTHONDONTWRITEBYTECODE=1 python3 -m MASW.tests.tech_selection_eval
PYTHONDONTWRITEBYTECODE=1 python3 -m MASW.tests.baseline_compare
PYTHONDONTWRITEBYTECODE=1 python3 -m MASW.tests.e2e_attack_validation
```

当前最小数据集位于：

```text
MASW/tests/data/masw_min_dataset.jsonl
```

规模：

- prompt_injection: 20
- tool_misuse: 20
- memory_poisoning: 20
- agent_hijacking: 20
- benign: 45

评估结果会写入：

```text
MASW/tests/results/masw_eval_results.json
```

消融评估输出：

```text
MASW/tests/results/masw_ablation_metrics.csv
MASW/tests/results/masw_ablation_results.json
MASW/tests/results/masw_ablation_stage_report.md
```

二轮技术选型评估输出：

```text
MASW/tests/results/masw_tech_selection_metrics.csv
MASW/tests/results/masw_tech_selection_results.json
MASW/tests/results/masw_tech_selection_report.md
```

当前选型结论：`d3_risk_filter` 使用 `HybridDetector` 提升隐蔽同步/外传样本的召回；
`d4_provenance_gate` 内部仍使用 `RuleBasedDetector`，因为 D4 的验证阈值更低，
在该位置使用 hybrid 会增加良性内部同步样本的误报。

横向基线对比输出：

```text
MASW/tests/results/masw_baseline_comparison_metrics.csv
MASW/tests/results/masw_baseline_comparison_results.json
MASW/tests/results/masw_baseline_comparison_report.md
```

DeepSeek baseline 默认跳过远程调用，避免误用 API key 或产生费用。需要本地补跑时：

```bash
cp .env.example .env
# 编辑 .env：填入 DEEPSEEK_API_KEY，并把 MASW_RUN_REMOTE_BASELINES 改成 1
PYTHONDONTWRITEBYTECODE=1 python3 -m MASW.tests.baseline_compare
```

端到端攻击链验证输出：

```text
MASW/tests/results/masw_e2e_attack_validation_results.csv
MASW/tests/results/masw_e2e_attack_validation_results.json
MASW/tests/results/masw_e2e_attack_validation_report.md
```

该实验搭建了一个最小脆弱 Agent，复现
`Q_inject -> 毒化共享记忆写入 -> Q_target -> 检索 -> 危险工具执行`。
当前 8 个场景中，脆弱路径危险执行 `8/8`；MASW 毒化记忆写入 `0/8`，
MASW 危险执行 `0/8`。
