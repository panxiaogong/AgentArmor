# MASW Ablation Stage Report

## Node-Level Answers

| Node | Corresponding stage | Result | Evidence |
|---|---|---|---|
| `d1_input_label` | 外部输入信任边界 | 部分成立：能标记所有外部输入，但不能区分攻击/良性。 | Prec=0.6400, Rec=1.0000, FPR=1.0000 |
| `d2_candidate_extract` | 结构化候选事实抽取 | 不是独立拦截器；它为 d3/d4 提供可审计输入。 | Prec=0.0000, Rec=0.0000, FPR=0.0000 |
| `d3_risk_filter` | 提示注入/工具误用/记忆投毒/劫持文本早期过滤 | 成立：当前最小集上单节点已高召回。 | Prec=1.0000, Rec=1.0000, FPR=0.0000 |
| `d4_provenance_gate` | 共享记忆写入前的验证、提升和网关 | 成立：阻断低信任或高风险内容自动进入 shared memory。 | Prec=1.0000, Rec=1.0000, FPR=0.0000 |
| `d5_retrieval_audit` | 读取阶段的 trust/scope/taint 过滤 | 部分成立：能阻断被召回的污染记忆，但单独使用覆盖不足。 | Prec=1.0000, Rec=0.0625, FPR=0.0000 |
| `d6_execution_align` | 工具调用和高影响动作仲裁 | 部分成立：能阻断高风险上下文/工具，但单独使用误伤高。 | Prec=0.6400, Rec=1.0000, FPR=1.0000 |
| `d7_revocation` | 事后污染撤销和派生记忆清理 | 不是入口拦截器；用于已确认污染后的 containment。 | Prec=1.0000, Rec=1.0000, FPR=0.0000 |

## Composition Answer

- ALL F1: 1.0000
- Best single-node F1: 1.0000 (`D3_RISK_FILTER_ONLY`)
- Strict F1 synergy (`ALL > best single`): False
- F1 gain over best single: 0.0000

Interpretation: strict 1+1>2 is only claimed when ALL exceeds the best single-node F1. If this is false, the current minimum dataset shows saturation by one early node rather than mathematical synergy. The combined design is still defense-in-depth because leave-one-out rows measure robustness when a stage is bypassed.
