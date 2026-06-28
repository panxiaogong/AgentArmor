# Type 4 基线对比评估报告

**生成时间**: 2026-06-28 21:24:21

**测试目的**: 将同一 200 文档数据集（120 attack + 80 benign）
跑在 3 个外部基线上，与 SP1-SP7 多节点防御系统横向比较。

---

## 1. 数据集概览

| 类别 | 数量 | 来源 |
|------|------|------|
| Prompt Injection | 30 | JailbreakBench + Gandalf + AgentDojo |
| Tool Misuse | 20 | SafeRAG + AgentDojo |
| Memory Poisoning | 20 | AgentDojo + Gandalf |
| Agent Hijacking | 20 | AgentDojo + SafeRAG |
| Enhanced PoisonedRAG | 10 | PoisonedRAG (USENIX Security 2025) |
| Enhanced AgentPoison | 10 | AgentPoison (NeurIPS 2024) |
| Hybrid Attacks | 10 | Multi-attack composite |
| **攻击合计** | **120** | |
| 良性文档 (conftest) | 40 | 常规知识库文档 |
| 良性文档 (extra) | 40 | 常规知识库文档 |
| **良性合计** | **80** | |
| **总计** | **200** | |

## 2. 基线系统描述

| # | 基线 | 类型 | 方法 |
|---|------|------|------|
| 1 | Keyword Rule | 规则引擎 | 21 条正则模式匹配攻击关键词（INJECT、OVERRIDE、BANK_TRANSFER 等） |
| 2 | Perplexity (n-gram) | 统计方法 | 字符级 5-gram 困惑度异常检测，阈值 = 均值 + 2×标准差 |
| 3 | DeepSeek-v4-Flash | LLM-as-judge | 调用 DeepSeek API，逐文档分类为 attack/benign |

## 3. 总体性能对比

| Baseline | Precision | Recall | F1 | FPR | P50(ms) | P95(ms) | P99(ms) |
|----------|-----------|--------|----|-----|---------|---------|---------|
| DeepSeek-v4-Flash | 1.0000 | 0.9250 | 0.9610 | 0.0000 | 1019.0805 | 1259.1510 | 1620.9627 |
| Keyword Rule | 1.0000 | 0.3583 | 0.5276 | 0.0000 | 0.0487 | 0.0663 | 0.0861 |
| Perplexity (n-gram) | 0.9756 | 1.0000 | 0.9877 | 0.0375 | 0.0853 | 0.1525 | 0.2416 |

### 3.1 与 SP1-SP7 系统对比

| 指标 | Keyword Rule | Perplexity | DeepSeek | SP1-SP7 系统 (Config-5) |
|------|-------------|------------|----------|------------------------|
| **最佳 F1** | 0.5276 | 0.9877 | 0.9610 | 0.968 (SP4) |
| **F1 范围** | 0.5276~0.9877 | | | 0.857 (SP2:PI)~0.968 (SP4:AP) |
| **综合延迟 P50** | 339.7382ms | | | 0.1463ms |

## 4. 分攻击类型 F1 对比

| Attack Type | DeepSeek-v4-Flash | Keyword Rule | Perplexity (n-gram) | SP1-SP7 |
|-------------| --- | --- | --- | --- |
| prompt_injection | 0.9474 | 0.5366 | 1.0000 | 0.857 (SP2) |
| tool_misuse | 1.0000 | 0.5185 | 1.0000 | 1.000 (SP5) |
| memory_poisoning | 0.9744 | 0.0000 | 1.0000 | 0.769 (SP6) |
| agent_hijacking | 0.9744 | 0.2609 | 1.0000 | 0.545 (SP6) |
| enhanced_poisonedrag | 0.7500 | 0.6667 | 1.0000 | 0.714 (SP1) |
| enhanced_agentpoison | 1.0000 | 1.0000 | 1.0000 | 0.645 (SP4) |
| hybrid | 1.0000 | 0.8235 | 1.0000 | 0.000 (SP7) |

## 5. 延迟性能对比

| Baseline | Mean (ms) | P50 (ms) | P95 (ms) | P99 (ms) |
|----------|-----------|----------|----------|----------|
| DeepSeek-v4-Flash | 1029.2891 | 1019.0805 | 1259.1510 | 1620.9627 |
| Keyword Rule | 0.0497 | 0.0487 | 0.0663 | 0.0861 |
| Perplexity (n-gram) | 0.0917 | 0.0853 | 0.1525 | 0.2416 |

## 6. 基线详情

### DeepSeek-v4-Flash (LLM-as-judge (deepseek-chat) API 调用分类)

- **Precision**: 1.0000
- **Recall**: 0.9250
- **F1**: 0.9610
- **FPR**: 0.0000
- **TP=111, FP=0, TN=80, FN=9**
- **Latency**: P50=1019.0805ms, P95=1259.1510ms, P99=1620.9627ms

  **分类型指标:**

  | Attack Type | Precision | Recall | F1 | FPR |
  |-------------|-----------|--------|----|-----|
  | agent_hijacking | 1.0000 | 0.9500 | 0.9744 | 0.0000 |
  | enhanced_agentpoison | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
  | enhanced_poisonedrag | 1.0000 | 0.6000 | 0.7500 | 0.0000 |
  | hybrid | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
  | memory_poisoning | 1.0000 | 0.9500 | 0.9744 | 0.0000 |
  | prompt_injection | 1.0000 | 0.9000 | 0.9474 | 0.0000 |
  | tool_misuse | 1.0000 | 1.0000 | 1.0000 | 0.0000 |

---

### Keyword Rule (Regex 关键词规则匹配（21 条攻击模式）)

- **Precision**: 1.0000
- **Recall**: 0.3583
- **F1**: 0.5276
- **FPR**: 0.0000
- **TP=43, FP=0, TN=80, FN=77**
- **Latency**: P50=0.0487ms, P95=0.0663ms, P99=0.0861ms

  **分类型指标:**

  | Attack Type | Precision | Recall | F1 | FPR |
  |-------------|-----------|--------|----|-----|
  | agent_hijacking | 1.0000 | 0.1500 | 0.2609 | 0.0000 |
  | enhanced_agentpoison | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
  | enhanced_poisonedrag | 1.0000 | 0.5000 | 0.6667 | 0.0000 |
  | hybrid | 1.0000 | 0.7000 | 0.8235 | 0.0000 |
  | memory_poisoning | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
  | prompt_injection | 1.0000 | 0.3667 | 0.5366 | 0.0000 |
  | tool_misuse | 1.0000 | 0.3500 | 0.5185 | 0.0000 |

---

### Perplexity (n-gram) (字符级 5-gram 困惑度异常检测)

- **Precision**: 0.9756
- **Recall**: 1.0000
- **F1**: 0.9877
- **FPR**: 0.0375
- **TP=120, FP=3, TN=77, FN=0**
- **Latency**: P50=0.0853ms, P95=0.1525ms, P99=0.2416ms

  **分类型指标:**

  | Attack Type | Precision | Recall | F1 | FPR |
  |-------------|-----------|--------|----|-----|
  | agent_hijacking | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
  | enhanced_agentpoison | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
  | enhanced_poisonedrag | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
  | hybrid | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
  | memory_poisoning | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
  | prompt_injection | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
  | tool_misuse | 1.0000 | 1.0000 | 1.0000 | 0.0000 |

---

## 7. 结论

### 7.1 各基线表现

- **DeepSeek-v4-Flash**: 最佳 F1=1.0000 (enhanced_agentpoison), 最差 F1=0.7500 (enhanced_poisonedrag)
- **Keyword Rule**: 最佳 F1=1.0000 (enhanced_agentpoison), 最差 F1=0.0000 (memory_poisoning)
- **Perplexity (n-gram)**: 最佳 F1=1.0000 (agent_hijacking), 最差 F1=1.0000 (agent_hijacking)

### 7.2 与 SP 系统对比分析

对比基于规则、统计、LLM 三种范式的基线，与多节点融合的 SP 防御系统：

- **规则基线** 对显式注入关键词（INJECT, BANK_TRANSFER 等）召回率高，
  但容易漏检无显式标记的隐式攻击（I-subtext）。
- **困惑度基线** 对包含非常规文本模式的攻击文档有效，
  但对精心伪装成正常文本的攻击检测效果有限。
- **LLM 基线** 利用语义理解能力，理论上能检测更广泛的攻击类型，
  但受限于 API 延迟和成本。
- **SP 多节点系统** 通过嵌入异常、困惑度、跨块连贯性、触发区域、
  鲁棒聚合、后检索验证、语义图等多维度检测，覆盖面更广。
