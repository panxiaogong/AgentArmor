# MASW: Multi-Agent Shared Write

这个目录采用和参考图一致的命名方式：`config.py`、`types.py`、`pipeline.py`
加 `d1_*.py` 到 `d7_*.py` 的防御节点文件。

## 阶段报告

当前阶段完整报告：

```text
MASW/report_stage_current.md
```

论文风格架构图集：

```text
MASW/figures/
```

## 文件说明

- `d1_input_label.py`: 外部输入低信任标记。
- `d2_candidate_extract.py`: Agent A 候选事实抽取。
- `d3_risk_filter.py`: 注入与指令型载荷风险检测。
- `d4_provenance_gate.py`: provenance、验证提升、共享记忆写入网关。
- `d5_retrieval_audit.py`: 信任感知检索与审计。
- `d6_execution_align.py`: 动作提案与工具调用仲裁。
- `d7_revocation.py`: 污染记忆追踪与撤销。
- `e2e_vulnerable_agent.py`: 仅用于实验的脆弱共享写入 Agent，复现攻击链。

## 运行

从 `Multi-Agent Shared Write` 的父目录或当前目录执行：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s MASW/tests
PYTHONDONTWRITEBYTECODE=1 python3 -m MASW.tests.build_dataset
PYTHONDONTWRITEBYTECODE=1 python3 -m MASW.tests.eval_masw
PYTHONDONTWRITEBYTECODE=1 python3 -m MASW.tests.ablation_eval
PYTHONDONTWRITEBYTECODE=1 python3 -m MASW.tests.tech_selection_eval
PYTHONDONTWRITEBYTECODE=1 python3 -m MASW.tests.baseline_compare
PYTHONDONTWRITEBYTECODE=1 python3 -m MASW.tests.e2e_attack_validation
```

## 测试覆盖

- `test_d1_input_label.py`: 外部输入标记、taint、审计事件。
- `test_d2_candidate_extract.py`: 候选事实抽取、低信任继承、恶意内容隔离。
- `test_d3_risk_filter.py`: prompt injection、tool misuse、memory poisoning、agent hijacking 风险检测。
- `test_d4_provenance_gate.py`: 验证提升、共享记忆写入网关、冲突检测。
- `test_d5_retrieval_audit.py`: trust/scope/taint 检索过滤与审计。
- `test_d6_execution_align.py`: 动作提案、工具权限、高风险上下文、高影响工具审批。
- `test_d7_revocation.py`: 污染记忆及派生记忆撤销。
- `test_memory_store_and_types.py`: 共享数据结构、记忆仓库、隔离区、审计日志。
- `test_dataset_against_pipeline.py`: 125 条样本的端到端回归。
- `test_ablation_eval.py`: 消融指标公式、ALL 配置和协同结论字段。
- `test_tech_selection_eval.py`: 二轮检测器技术选型与 hard set 回归。
- `test_baseline_compare.py`: MASW 与 DeepSeek/工具基线的横向比较入口。
- `test_e2e_attack_validation.py`: `Q_inject -> Q_target` 两轮攻击链复现与 MASW 对照。

## 数据集

最小可用数据集：

```text
MASW/tests/data/masw_min_dataset.jsonl
```

构成：

- `prompt_injection`: 20
- `tool_misuse`: 20
- `memory_poisoning`: 20
- `agent_hijacking`: 20
- `benign`: 45

外部参考来源见：

```text
MASW/tests/data/DATASET_SOURCES.md
```

当前评估目标是先得到可复现的最小测试集。后续可以把 JailbreakBench、
PromptInject、AgentDojo 等公开 benchmark 的样本按同一 schema 导入。

## 消融评估

消融脚本：

```text
MASW/tests/ablation_eval.py
```

输出：

```text
MASW/tests/results/masw_ablation_metrics.csv
MASW/tests/results/masw_ablation_results.json
MASW/tests/results/masw_ablation_stage_report.md
```

指标：

- `precision = TP / (TP + FP)`
- `recall = TP / (TP + FN)`
- `f1 = 2 * precision * recall / (precision + recall)`
- `fpr = FP / (FP + TN)`
- `p50_ms/p95_ms/p99_ms`: 只在 `ALL` 行显示

当前最小集上的结论：`ALL` 达到 `Precision=1.0`、`Recall=1.0`、`F1=1.0`、`FPR=0.0`。
严格 F1 协同 `ALL > best single` 为 false，因为 `D3_RISK_FILTER_ONLY` 与
`D4_PROVENANCE_GATE` 在当前最小数据集上已经达到饱和。这个结论应解释为：
当前数据集还不够难，不能证明数学意义上的 `1+1>2`；组合价值主要体现为
defense-in-depth 和绕过单点后的鲁棒性。

## 二轮技术选型

选型脚本：

```text
MASW/tests/tech_selection_eval.py
```

输出：

```text
MASW/tests/results/masw_tech_selection_metrics.csv
MASW/tests/results/masw_tech_selection_results.json
MASW/tests/results/masw_tech_selection_report.md
```

该脚本在最小数据集上额外加入 10 条隐蔽同步/外传攻击和 10 条良性内部同步样本，
用来检验第一轮规则检测是否被“知识库同步、归档、路由”等语义绕过。

当前结论：

- `D3_RULE` 在 hard set 上漏检隐蔽外传，整体 `Recall=0.8889`。
- `D3_HYBRID` 通过规则 + 语义 rubric 补足召回，`Recall=1.0000` 且 `FPR=0.0000`。
- `D4_GATE_HYBRID` 因 D4 验证阈值更低，会误伤部分良性内部同步，`FPR=0.0364`。
- 推荐组合是 `D3_HYBRID_PLUS_D4_RULE`，即 D3 用 `HybridDetector`，D4 保持 `RuleBasedDetector`。

## 横向基线对比

基线脚本：

```text
MASW/tests/baseline_compare.py
```

输出：

```text
MASW/tests/results/masw_baseline_comparison_metrics.csv
MASW/tests/results/masw_baseline_comparison_results.json
MASW/tests/results/masw_baseline_comparison_report.md
```

当前比较对象：

- `MASW_ALL_RECOMMENDED`: 本系统，D3 HybridDetector + D4 RuleBasedDetector + D5/D6。
- `DEEPSEEK_V4_FLASH`: DeepSeek LLM-as-judge baseline，通过本地环境变量或 `.env` 启用。
- `PROMPTINJECT_STYLE_TOOL`: PromptInject 风格关键词工具基线。
- `LLM_GUARD_STYLE_TOOL`: LLM Guard 风格多扫描器工具基线。
- `REBUFF_STYLE_TOOL`: Rebuff 风格签名/canary 工具基线。

DeepSeek baseline 默认跳过，避免把 API key 写入代码或结果文件。本地补跑方式：

```bash
cp .env.example .env
# 编辑 .env：填入 DEEPSEEK_API_KEY，并设置 MASW_RUN_REMOTE_BASELINES=1
PYTHONDONTWRITEBYTECODE=1 python3 -m MASW.tests.baseline_compare
```

`.env` 已在 `.gitignore` 中忽略，`.env.example` 只保留占位符。

## 端到端攻击链验证

验证脚本：

```text
MASW/tests/e2e_attack_validation.py
```

输出：

```text
MASW/tests/results/masw_e2e_attack_validation_results.csv
MASW/tests/results/masw_e2e_attack_validation_results.json
MASW/tests/results/masw_e2e_attack_validation_report.md
```

该实验单独搭建了一个最小脆弱 Agent，用于复现真实失败路径：

```text
Q_inject -> 毒化共享记忆写入 -> Q_target -> 检索 -> 危险工具执行
```

脆弱 Agent 故意采用扁平信任模型：外部输入直接写入 shared memory，标记为
`TRUSTED` 且 `taint=False`，后续检索到的记忆被当成可执行指令。危险工具不会
真的执行外部副作用，只写入 dry-run `DangerousToolLog`。

当前 8 个端到端场景覆盖 `email.send`、`database.write`、`repo.commit`、
`deploy.production`、`secret.read`。结果：脆弱路径危险执行 `8/8`；
MASW 毒化记忆写入 `0/8`；MASW 危险执行 `0/8`。
