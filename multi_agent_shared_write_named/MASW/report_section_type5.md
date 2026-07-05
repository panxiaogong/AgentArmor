# Type 5: Multi-Agent Shared Write

本目录对应“多 Agent 共享写入”防御原型。

当前阶段完整报告见：

```text
MASW/report_stage_current.md
```

配套英文架构图见：

```text
MASW/figures/
```

命名规则：

- `d1_input_label.py`: 外部输入低信任标记与 spotlighting。
- `d2_candidate_extract.py`: Agent A 只抽取候选事实，不直接写共享记忆。
- `d3_risk_filter.py`: 注入风险、指令型载荷、上下文风险检测。
- `d4_provenance_gate.py`: 冲突检测、验证提升、共享记忆写入网关。
- `d5_retrieval_audit.py`: 信任感知检索与审计。
- `d6_execution_align.py`: 动作提案与工具调用仲裁。
- `d7_revocation.py`: 污染记忆追踪与撤销。

核心不变量：

```text
No automatic trust promotion.
```

任何来自外部内容的候选事实，不能因为“被某个 Agent 写入”就自动变成共享可信记忆。

测试现状：

- 单元测试：覆盖 `d1` 到 `d7` 每个防御节点，以及 `types.py`、`memory_store.py`。
- 数据集：4 类攻击各 20 条，共 80 条；良性样本 45 条。
- 回归评估：`attack_shared_memory_writes = 0`，`attack_quarantine_rate = 1.0`，`benign_write_rate = 1.0`。
- 消融评估：`ALL` 行 `Precision=1.0`、`Recall=1.0`、`F1=1.0`、`FPR=0.0`；
  严格 F1 协同为 false，因为当前最小集上 D3/D4 单节点已经饱和。
- 二轮技术选型：额外加入 10 条隐蔽同步/外传攻击和 10 条良性内部同步样本。
  `D3_RULE` 的整体 `Recall=0.8889`，`D3_HYBRID` 的整体 `Recall=1.0` 且 `FPR=0.0`；
  但 `D4_GATE_HYBRID` 在更低验证阈值下产生 `FPR=0.0364`。因此当前推荐为
  `D3_HYBRID_PLUS_D4_RULE`：D3 用 HybridDetector 做早期语义召回，D4 保持
  RuleBasedDetector 作为保守 provenance gate。
- 横向基线对比：同一 145 条数据集上，`MASW_ALL_RECOMMENDED` 得到
  `Precision=1.0`、`Recall=1.0`、`F1=1.0`、`FPR=0.0`；本地工具基线
  `PROMPTINJECT_STYLE_TOOL`、`LLM_GUARD_STYLE_TOOL`、`REBUFF_STYLE_TOOL`
  召回分别为 `0.3222`、`0.4333`、`0.4222`。`DEEPSEEK_V4_FLASH` 远程基线
  已接入，但默认跳过，需本地 `.env` 显式设置 API key 和运行开关后补跑。
- 端到端攻击链验证：搭建最小脆弱 Agent，复现
  `Q_inject -> 毒化共享记忆写入 -> Q_target -> 检索 -> 危险工具执行`。
  8 个场景中脆弱路径危险执行 `8/8`；MASW 毒化记忆写入 `0/8`，危险执行 `0/8`。
