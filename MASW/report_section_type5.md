# Type 5: Multi-Agent Shared Write

本目录对应“多 Agent 共享写入”防御原型。

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
