# AgentArmor

**AgentArmor** 是一套面向 LLM Agent 记忆系统的纵深防御研究原型。

系统针对攻击者通过操控 Agent 读写路径来污染记忆存储的威胁，构建了覆盖写入、检索、工具执行三个关键环节的多层防御体系。项目按 5 种攻击类型分模块实现，每个模块均包含独立的防御流水线、消融实验框架和评估数据集。

---

## 威胁模型

攻击者无法直接访问记忆存储，但可通过以下路径间接投毒：

| 攻击类型 | 攻击向量 |
|---|---|
| **Type 1 — AutoWrite** | 利用框架自动写入机制（如 LangChain `save_context()`），绕过人工过滤直接污染记忆 |
| **Type 2 — MINJA** | 在查询中嵌入隐藏指令，诱使 Agent 调用 `archival_memory_insert()` 写入恶意记忆 |
| **Type 3 — Reflection** | 劫持反思/合成写入路径，使恶意内容以"精炼事实"身份获得高信任度持久化 |
| **Type 4 — External Developer Write** | 在文档上传至 RAG 知识库时注入投毒内容（PoisonedRAG、AgentPoison 等） |
| **Type 5 — Multi-Agent Shared Write** | Agent A 读取不可信外部内容后写入共享记忆，Agent B/C 将其作为可信上下文执行高权限工具 |

---

## 项目结构

```
AgentArmor-main-3/
├── MINJA/                          # Type 2 防御模块
├── AutoWrite/                      # Type 1 防御模块
├── Reflection/                     # Type 3 防御模块
├── External_Developer_Write/       # Type 4 防御模块
├── MASW/                           # Type 5 简化原型
├── multi_agent_shared_write_named/ # Type 5 完整实现（含论文、评估、可视化）
│   ├── MASW/                       # 核心防御流水线
│   ├── figures/                    # 实验结果图表（PNG/PDF）
│   ├── document.tex / .pdf         # LaTeX 论文
│   └── artifacts/                  # 实验计划、审计日志、案例分析
└── multi_agent_shared_write_code/  # Type 5 早期代码原型
```

---

## 各模块防御机制

### Type 1 — AutoWrite

防御自动写入场景，在 MINJA D1-D6 管线基础上新增 6 个节点：

| 节点 | 功能 |
|---|---|
| D-A `TokenSanitizer` | Token 级清洗，过滤控制字符和注入 token |
| D-B `SelectiveWrite` | 选择性写入策略：新颖度过滤 + 写入频率限制 |
| D-C `IntegrityChain` | 密码学链式哈希，跨会话篡改检测 |
| D-D `TemporalDecay` | 检索时对陈旧记忆降权 |
| D-E `RetrievalAlign` | 双通道检索对齐（嵌入相似度 + LLM 判断） |
| D-F `DistributionMonitor` | KL 散度扫描，监控记忆分布健康状态 |

**消融结果（mock 模式，~120 样本）：**

| 配置 | F1 |
|---|---|
| Baseline（无防御） | 0.000 |
| +D-A | 0.049 |
| +D-A +D-B | 0.462 |
| Full（D-A 至 D-F） | 0.476 |

---

### Type 2 — MINJA

防御 MINJA 攻击（arXiv:2503.03704），6 节点流水线：

| 节点 | 功能 |
|---|---|
| D1 `QueryIntent` | 写意图检测：关键词正则 / 语义子空间距离 / LLM 分类器 |
| D2 `CausalWrite` | 反事实因果归因：计算 indication_prompt 对写入动作的 Dominance Shift |
| D3 `ProspectiveSim` | 前瞻行为模拟：生成对抗激活上下文，预判候选记忆是否导致危险工具调用 |
| D4 `Provenance` | IFC 溯源签名：HMAC/Ed25519 绑定内容哈希 + 完整性标签 + 触发查询哈希 |
| D5 `RetrievalAudit` | 检索时 Hub 异常检测（HubScan）+ Bron-Kerbosch 最大团检测协同攻击 |
| D6 `ExecutionAlign` | 工具调用与原始用户任务的余弦相似度对齐守卫 |

**消融结果（mock 模式，120 样本，MI-1/2/3/4 × 20 + 40 正常）：**

| 配置 | All F1 | MI-4 F1 |
|---|---|---|
| D1+D2+D3+D4 | 0.857 | 0.000（D5 未启用） |
| Full（D1-D6） | **1.000** | **1.000** |

---

### Type 3 — Reflection

防御反思合成写入，5 写入节点 + 溯源绑定器 + 检索守卫：

| 节点 | 功能 |
|---|---|
| D1 | 原始对话轮次中的反思目标注入检测 |
| D2 `Grounding` | 溯源锚定审计：验证合成事实是否有可信原始轮次支撑 |
| D3 `Consistency` | 与已信任记忆的槽位冲突检测 |
| D4 `Policy` | 策略审计：阻止存储密钥、凭证、可执行指令 |
| D5 `WriteGate` | 加权风险门控：`risk = 0.35×注入风险 + 0.35×(1-溯源分) + 0.20×矛盾分 + 0.10×策略分` |

---

### Type 4 — External Developer Write

防御文档上传时的 RAG 知识库投毒，7 节点流水线（SP1-SP7）：

| 节点 | 功能 |
|---|---|
| SP1 | 嵌入空间异常检测（Mahalanobis 距离 + LOF） |
| SP2 | 内容困惑度分析，检测注入指令的非自然语言模式 |
| SP3 | 跨块语义连贯性验证，检测拼接注入引入的语义断点 |
| SP4 | 触发区域检测，识别 AgentPoison 触发 token 的紧凑嵌入簇 |
| SP5 | 鲁棒聚合检索，稀释投毒文档对答案的影响 |
| SP6 | 检索后验证：NLI + 交叉编码器一致性校验 |
| SP7 | 语义依赖图分析：实体关系图检测跨文档碎片化注入 |

**评估结果（200 文档，Config-5）：**

| 方法 | F1 | P50 延迟 |
|---|---|---|
| 关键词规则基线 | 0.528 | — |
| DeepSeek-v4-Flash | 0.961 | ~1019 ms |
| SP1-SP7（无 API） | 0.645–1.000 | **~0.13 ms** |

---

### Type 5 — Multi-Agent Shared Write（MASW）

**核心不变量：外部内容不得仅凭流经 Agent 便自动获得信任，写入共享记忆。**

7 节点流水线（D1-D7）：

| 节点 | 功能 |
|---|---|
| D1 `InputLabel` | 低信任标记 + Spotlighting：用 `[UNTRUSTED_EXTERNAL_DATA_BEGIN]...[END]` 定界外部内容 |
| D2 `CandidateExtract` | Agent A 仅产出 `CandidateFact` 对象，禁止直接写入共享记忆 |
| D3 `RiskFilter` | 注入风险评分：覆盖短语、工具命令、策略记忆模式、数据外泄模式的加权检测 |
| D4 `ProvenanceGate` | 冲突检测 + 验证 + 信任提升 + 共享记忆写入网关 |
| D5 `RetrievalAudit` | 信任感知检索：按信任级别、作用域、污点标记过滤 |
| D6 `ExecutionAlign` | 行动提案 + 工具调用仲裁：高影响力工具需人工审批 |
| D7 `Revocation` | 投毒记忆撤销 + 经由 parent_ids 的级联清理 |

**端到端验证结果（8 攻击场景）：**

| 指标 | 脆弱 Agent | MASW |
|---|---|---|
| 危险工具执行 | 8/8 | **0/8** |
| 投毒记忆写入 | 8/8 | **0/8** |

**基线对比（145 样本，含困难集）：**

| 方法 | Precision | Recall | F1 | FPR |
|---|---|---|---|---|
| PromptInject-style | — | 0.322 | 0.487 | — |
| LLM-Guard-style | — | 0.433 | 0.605 | — |
| Rebuff-style | — | 0.422 | 0.594 | — |
| **MASW（推荐配置）** | **1.000** | **1.000** | **1.000** | **0.000** |

---

## 核心技术

**信任格（Trust Lattice）**：整数格 + 格汇运算（`min` 语义），任何不可信来源都会污染衍生结果。MINJA 使用 `IntegrityLabel`（UNTRUSTED / CANDIDATE / TRUSTED），MASW 使用四级 `TrustLevel`（0-3）。

**溯源签名**：MINJA D4 和 Reflection `ProvenanceBinder` 用 HMAC-SHA256 或 Ed25519 将内容哈希、完整性标签、触发查询哈希绑定为签名记录，检索时验证签名，篡改条目被过滤。

**Spotlighting**：MASW D1 用定界符显式标注数据/指令边界，防止下游模型将外部数据误认为系统指令。

**防御裁决协议**：所有流水线使用统一的 `DefenseVerdict`（PASS / FLAG / BLOCK / ASK），FLAG 升级至更昂贵的下游节点，ASK 暂停等待人工确认（用于高影响力工具调用）。

**消融友好设计**：每个防御节点均可通过类型化配置数据类独立开关，支持系统性消融实验。

**审计日志**：所有流水线输出结构化 JSONL 审计日志，覆盖写入、检索、工具调用事件。

---

## 环境依赖

核心流水线为纯 Python，无强制外部依赖，均可在 mock/启发式模式下运行。

| 依赖 | 用途 | 是否必须 |
|---|---|---|
| Python 3.10+ | 运行环境 | 是 |
| `numpy` | 嵌入向量计算（D5、AutoWrite） | 否 |
| `cryptography` | Ed25519 签名 | 否 |
| OpenAI / DeepSeek API Key | D2 因果归因、D3 行为模拟等 LLM 节点 | 否 |

---

## 快速开始

### Type 5 — MASW（简化版）

```bash
# 单元测试
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s MASW/tests

# 评估
PYTHONDONTWRITEBYTECODE=1 python3 -m MASW.tests.eval_masw
```

### Type 5 — MASW（完整版）

```bash
cd multi_agent_shared_write_named

# 单元测试
python3 -m unittest discover -s MASW/tests

# 消融评估
python3 -m MASW.tests.eval_masw_ablation

# 端到端攻击验证
python3 -m MASW.tests.e2e_attack_validation
```

### Type 2 — MINJA

```bash
cd MINJA

python3 -m unittest discover -s tests

# 消融评估
python3 -m tests.eval_minja_ablation
```

### Type 1 — AutoWrite

```bash
cd AutoWrite

python3 -m unittest discover -s tests

python3 -m tests.eval_autowrite_ablation
```

### Type 3 — Reflection

```bash
cd Reflection

python3 -m unittest discover -s tests
```

### Type 4 — External Developer Write

```bash
cd External_Developer_Write

python3 -m unittest discover -s tests
```

---

## 参考文献

- MINJA 攻击：arXiv:2503.03704
- CausalArmor：arXiv:2602.07918
- CaMeL：arXiv:2503.18813
- FIDES：arXiv:2505.23643
- HubScan：arXiv:2602.22427
