# 类型二：Agent 自主写入（Agent-Initiated Write）防御设计与实验测试

---

## 一、类型介绍及流程

### 1.1 定义

Agent 自主写入型记忆注入（Agent-Initiated Write Injection）是指攻击者通过构造特殊查询，诱导 Agent 在处理该查询时，将攻击者预设的恶意内容**通过 Agent 自身的写入机制**（如 MemGPT/Letta 的 `archival_memory_insert` 工具调用）持久化到长期记忆库中，从而影响未来所有检索该记忆的会话行为。

与类型一（外部直接写入）的根本区别在于：攻击者**无需任何对记忆库的直接访问权限**，仅通过正常用户查询接口即可完成注入；恶意内容的写入由 Agent 自身执行，绕过了所有外部写入防护。

### 1.2 写入流程

```
攻击者查询 Q_inject（含 indication prompt）
        │
        ▼
   Agent 接收并处理 Q_inject
        │
        ├─── 工具调用：搜索、执行等正常操作
        │
        ▼
   Agent 执行 archival_memory_insert(R_malicious)
        │                     ↑
        │         恶意记录被 indication prompt 诱导写入
        ▼
   长期记忆库（持久化）
        │
        ▼
   未来受害用户提交 Q_target
        │
        ▼
   Agent 检索到 R_malicious（作为 few-shot demonstration）
        │
        ▼
   Agent 执行攻击者预设动作（数据泄露/命令执行/行为篡改）
```

---

## 二、攻击节点分析

Agent 自主写入型攻击在如下三个节点具备可干预的攻击面：

| 节点编号 | 节点位置 | 攻击者控制要素 | 可检测信号 |
|---------|---------|-------------|-----------|
| N1 | 写入触发决策 | indication prompt 触发 Agent 调用 insert | 写入意图异常 |
| N2 | 写入内容本身 | 恶意指令嵌入"正常"表述的记忆条目 | 内容因果归因异常 |
| N3 | 检索-利用阶段 | 恶意记录成为高频检索命中（Hub 效应） | 向量空间 Hub 异常 |

这三个节点分别对应防御节点 D1（意图检测）、D2/D3（内容审查）、D5（检索异常检测）的设计出发点。

---

## 三、威胁模型

**攻击者能力假设（Dolev-Yao 子集）：**
- 攻击者仅具备普通用户的查询能力（Query-only），无法直接读写记忆库
- 攻击者知晓目标 Agent 使用的记忆架构（白盒威胁模型）
- 攻击者可对查询内容进行任意构造（包括 Progressive Shortening 优化）

**攻击目标：**
- 在受害 Agent 的长期记忆库中植入语义自然但行为有害的记忆条目 $R_{mal}$
- 使 $R_{mal}$ 在受害用户提交 $Q_{target}$ 时被高概率检索

**威胁特殊性（跨会话持久化）：**

可写的跨会话持久化记忆在 LLM Agent 中引入了与传统输入型安全威胁质性不同的攻击面（arXiv:2604.16548, arXiv:2606.04329）：单次恶意写入可产生休眠的、潜伏的跨会话效应，在未来会话中自动激活，且可通过自我强化写入（self-reinforcing writes）不断扩大影响范围（arXiv:2605.15338）。这意味着检索时或执行时的防御无法单独提供充分保护——**写入时防御是必要条件**。

---

## 四、MINJA 攻击详细流程（arXiv:2503.03704）

MINJA（Memory INJection Attack）是首个仅通过查询交互即可向 Agent 记忆库注入恶意记录的攻击框架。其核心利用两个结构性缺陷：
1. **无约束的自主写入**（Unchecked Autonomous Write）：Agent 可在任意时刻调用 insert，无需用户明确授权
2. **无条件的检索信任**（Unconditional Retrieval Trust）：从记忆库检索的内容直接作为 few-shot demonstration 执行

### 4.1 四阶段攻击流程

**阶段一：构造恶意记录 $R_{mal}$**

攻击者设计恶意目标记录 $R_{mal}$，该记录语义表面自然，但当被 Agent 作为 demonstration 时将导致目标行为 $a_{mal}$（如数据外泄、命令执行）。

**阶段二：构造注入查询 $Q_{inject}$（含 indication prompt）**

$$Q_{inject} = Q_{normal} \oplus P_{ind}$$

其中 $P_{ind}$（indication prompt）是嵌入查询中的隐式指令，通过语义操控诱导 Agent 在回应查询后调用 `archival_memory_insert(R_{mal})`。$Q_{normal}$ 为看似正常的用户请求，为攻击提供掩护。

**阶段三：Progressive Shortening（迭代压缩优化）**

为绕过基于语义的内容过滤器，MINJA 对 $P_{ind}$ 进行迭代压缩：

$$P_{ind}^{(t+1)} = \text{Compress}(P_{ind}^{(t)}, \text{LLM}, Q_{target})$$

迭代目标：在最小化 $|P_{ind}|$（降低检测概率）的同时，保持 $R_{mal}$ 与 $Q_{target}$ 的语义相似度 $\text{sim}(E(R_{mal}), E(Q_{target})) \geq \delta$。经过若干轮压缩后，$P_{ind}$ 可能退化为极短的残留标记（如 `PS:`、`[Rule]`），人工审阅难以识别其恶意性。

**阶段四：受害用户触发**

受害用户提交 $Q_{target}$，Agent 在记忆库中检索语义最近邻，命中 $R_{mal}$，在无意识中执行攻击者预设的行为 $a_{mal}$。

### 4.2 MINJA 变体分类

| 变体 | indication prompt 特征 | 主要防御目标 | 难度 |
|------|----------------------|------------|------|
| MI-1 | 明显指令词（"记住每次…"） | D1 可直接检测 | 低 |
| MI-2 | 压缩残留标记（`PS:`、`Rmk:`） | D1 失效，D2 主检 | 中 |
| MI-3 | 无 indication（纯内容诱导） | 依赖 D2/D3 | 高 |
| MI-4 | 群体 Hub 攻击（多条协同） | D5 检测 Hub 异常 | 高 |

---

## 五、防御节点设计

### 5.1 D1：写入意图检测

**防御节点**：在 Agent 决定调用 `archival_memory_insert` 之前，对 `indication_prompt` 字段（而非用户原始查询）进行意图审查。

> ⚠️ 实现要点：D1 的检测对象必须是 `write_context.indication_prompt`（攻击者控制的载荷），而非 `triggering_query`（用户的正常查询）。对 triggering_query 检测会导致高误报且对攻击无效。

**子方案一：关键词正则匹配 $D_1^{kw}$**

$$D_1^{kw}(q) = \mathbf{1}\left[\exists p_i \in \mathcal{P}_{kw} : p_i.\text{match}(q) = \text{True}\right]$$

其中 $\mathcal{P}_{kw}$ 为预定义危险模式集合，覆盖中英文指令词（"记住每次…"、"每当…发送"、`curl https://`、`ignore safety` 等）。优点：零延迟，无 API 消耗；缺点：对压缩残留标记无效（MI-2/MI-3）。

**子方案二：语义子空间距离 $D_1^{sub}$**

构建正常写入请求的嵌入子空间 $V_{benign} \subset \mathbb{R}^d$，通过距离阈值判断：

$$D_1^{sub}(q) = \mathbf{1}\left[\min_{v \in V_{benign}} \|E(q) - v\|_2 > \delta_{sub}\right]$$

其中 $E(\cdot)$ 为文本编码函数。优点：对语义变形有泛化能力；缺点：需要构建 $V_{benign}$，对 MI-3 型（无 indication）失效。

**子方案三：LLM 意图分类 $D_1^{llm}$**

$$D_1^{llm}(q) = \text{LLM\_classify}(q, \text{prompt}_{intent})$$

直接调用 LLM 判断查询是否包含指令注入意图。优点：语义泛化能力最强；缺点：额外 API 延迟与成本。

**D1 策略对比（见 RQ2 实验）：** 关键词匹配对 MI-1 完全有效，对 MI-2/MI-3 完全失效；子空间方案在分布外样本上 recall 下降；LLM 分类覆盖最广但延迟最高。本系统默认采用三策略级联，以关键词匹配为第一道快速过滤，LLM 分类兜底。

---

### 5.2 D2：反事实因果归因审查（CausalArmor arXiv:2602.07918）

**防御节点**：拦截 `archival_memory_insert` 调用前，判断写入决策的主因是用户原始目标还是外部 indication prompt。

**理论基础（反事实因果归因）：**

设写入动作 $a$，当前上下文信息源集合 $S = \{s_1, s_2, \ldots, s_n\}$（包含 `user_goal`、`current_context`、`indication_prompt` 等），基于 leave-one-out 消融的因果归因分数定义为：

$$\text{Attr}(s_i \to a) = P(a \mid S) - P(a \mid S \setminus \{s_i\})$$

其中 $P(a \mid S)$ 为在完整上下文下写入动作发生的概率，$P(a \mid S \setminus \{s_i\})$ 为移除信息源 $s_i$ 后的条件概率，两者均通过 proxy LLM 估算。

**Dominance Shift 判决准则：**

$$\text{DS} = \text{Attr}(\text{indication\_prompt} \to a) - \text{Attr}(\text{user\_goal} \to a)$$

$$\text{判决} = \begin{cases} \text{BLOCK} & \text{if } \text{DS} > \theta \\ \text{FLAG} \to D3 & \text{if } |\text{DS}| < \delta_{margin} \\ \text{PASS} & \text{otherwise} \end{cases}$$

其中 $\theta$ 为注入主导阈值（默认 0.3），$\delta_{margin}$ 为边界容差（默认 0.1）。

**绕过的内在困难（理论论证）：** 攻击者若要绕过 D2，需同时满足：(1) indication_prompt 归因分数低（不被检测）；(2) 写入内容仍与 $Q_{target}$ 语义对齐（攻击仍有效）。这两个条件存在内在张力——降低 indication_prompt 的因果影响意味着削弱其对写入内容的控制力，从而降低攻击成功率。

**注意（深度研究发现）：** 我们早期将 D2 的理论来源标注为"CausalArmor"，但深度研究的对抗验证表明，arXiv:2603.10749（"Defeating Indirect Prompt Injection via Causal Attribution of Tool Invocations"）并非名为"AttriGuard"的论文。本系统 D2 的因果归因方法来源于 CausalArmor（arXiv:2602.07918），该论文通过 causal ablation 对每个不可信输入片段赋予影响分数，判断其是否主导了 Agent 的动作决策。

---

### 5.3 D3：前瞻行为仿真验证（Proactive Defense arXiv:2510.02373）

**防御节点**：在 D2 返回边界区域（FLAG）时触发，对候选写入内容 $m$ 进行前向仿真，检验其被未来检索利用时是否会导致危险行为。

**核心思想（软件安全类比）：** D3 等价于静态污点分析（Taint Analysis）的前向延伸——不在写入时判断"内容看起来是否危险"，而是前向仿真"内容被使用时产生的行为效果"。与 MemGuard 现有 IMAG 检测的根本区别：IMAG 检测内容语义特征（与攻击样本库比较），MINJA 生成的内容语义自然，IMAG 无效；D3 仿真内容被激活时的行为效果，无论内容外观多自然，只要激活时导致危险行为即拦截。

**形式化定义：**

设对抗性激活上下文集合 $\mathcal{C} = \{c_1, c_2, \ldots, c_n\}$，其中每个 $c_i$ 模拟未来受害者可能提交的 $Q_{target}$，行为仿真函数 $\text{sim}(m, c_i)$ 输出在上下文 $c_i$ 下 Agent 以 $m$ 为 few-shot demonstration 时执行的动作的安全性判断：

$$\text{unsafe\_ratio}(m) = \frac{|\{c_i \in \mathcal{C} : \text{sim}(m, c_i) = \text{unsafe}\}|}{|\mathcal{C}|}$$

$$\text{判决} = \begin{cases} \text{BLOCK} & \text{if } \exists c_i : \text{sim}(m, c_i) = \text{unsafe} \\ \text{PASS} & \text{otherwise} \end{cases}$$

**三种仿真策略：**

| 策略 | 方法 | 优点 | 缺点 |
|------|------|------|------|
| `template` | 规则关键词静态扫描 | 零延迟，无 API | 覆盖有限，易绕过 |
| `llm_judge` | LLM 直接评估危险行为 | 语义泛化，轻量 | 需 API，有误判 |
| `full_agent` | 完整 Agent 推理仿真 | 最准确 | 成本最高，延迟大 |

本系统默认 `llm_judge`，仅在 D2 返回 FLAG 时触发（条件激活），节约计算资源。

---

### 5.4 D4：信息流控制与溯源签名（CaMeL arXiv:2503.18813，Denning 1976 格模型）

**防御节点**：写入时为每条记忆项附加不可伪造的溯源签名，检索时验证签名完整性，拒绝执行无可信溯源的记忆条目。

**理论基础（信息流控制格模型）：**

基于 Denning（1976）的格模型，定义安全标签偏序格 $(L, \leq)$，其中：

$$L = \{\text{trusted}, \text{untrusted}\}, \quad \text{trusted} \leq \text{untrusted}$$

每条数据流携带安全标签 $\ell \in L$。CaMeL（arXiv:2503.18813）的核心贡献在于将 LLM 显式定义为**不可信组件**：由 P-LLM 生成代码表达用户查询，独立解释器构造数据流图并在工具调用边界处执行策略检查，使得来自不可信来源的数据无法影响控制流。

FIDES（arXiv:2505.23643）进一步通过动态污点传播（Dynamic Taint Tracking）将机密性和完整性标签附加到流经 Agent planner 的所有数据项，在记忆写入边界处确定性地执行安全策略，作为启发式防御的确定性补充。

**本系统实现（D4 溯源签名方案）：**

每条写入记忆项 $r$ 携带溯源标签：

$$\text{tag}(r) = \left(\ell_r,\ \text{sig}(r)\right)$$

$$\text{sig}(r) = \text{HMAC}_{k}\!\left(\text{content}(r) \,\|\, \ell_r \,\|\, h_{tq}\right)$$

其中 $\ell_r \in \{\text{trusted}, \text{untrusted}\}$ 为来源安全级别，$h_{tq} = \text{SHA-256}(\text{triggering\_query})$ 为触发查询的哈希，$k$ 为系统私钥（Ed25519/HMAC-SHA256）。

**格单调性保证（Taint 传播规则）：**

$$\ell_{a \oplus b} = \ell_a \sqcup \ell_b = \max(\ell_a, \ell_b)$$

即任何含有 `untrusted` 来源数据的写入，结果标签必为 `untrusted`，不可降级。检索时若签名验证失败或标签为 `untrusted` 且无用户明确授权，则该记忆项被拒绝执行。

**防御意义：** D4 不依赖内容语义判断，从结构层面确保恶意注入记忆无法伪造可信溯源。即使 D1/D2/D3 均被绕过，缺乏有效签名的记忆项在检索利用阶段仍会被拦截。

---

### 5.5 D5：向量空间 Hub 异常检测（HubScan arXiv:2602.22427）

**防御节点**：定期扫描记忆库，检测异常高频命中的 Hub 向量，识别群体攻击中的对抗性 Hub 记录。

**理论基础（向量 Hubness 现象）：**

在高维向量空间中，Hubness 是指少数点成为大量不相关查询的最近邻的现象（arXiv:2602.22427, arXiv:2606.19692）。攻击者可利用此特性，精心构造一条记录使其成为广泛查询的命中项，单次写入即可影响大量不相关请求：

$$h(d) = \left|\left\{q \in Q : d \in \text{top-}k(q)\right\}\right|$$

其中 $h(d)$ 为记录 $d$ 的 Hub 分数，$Q$ 为查询集合，$\text{top-}k(q)$ 为查询 $q$ 的 $k$ 近邻结果集。

**HubScan 检测方法：**

计算记忆库中所有条目的 Hub 分数分布统计量，通过阈值过滤异常项：

$$\mu_h = \frac{1}{|M|}\sum_{d \in M} h(d), \quad \sigma_h = \sqrt{\frac{1}{|M|}\sum_{d \in M}(h(d)-\mu_h)^2}$$

$$\tau = \mu_h + \alpha \cdot \sigma_h$$

$$\text{Hub\_Alert}(d) = \mathbf{1}\!\left[h(d) > \tau\right]$$

其中 $\alpha$ 为灵敏度参数（默认 $\alpha = 1.5$）。对抗验证结果（arXiv:2602.22427，3-0 置信度）：HubScan 在 0.2% 报警预算下达到 90% 召回率，0.4% 预算下达到 100% 召回率，对抗性 Hub 排位高于全库第 99.8 百分位。

**策略二：Bron-Kerbosch 语义一致团检测：**

对于协同群体攻击（MI-4），多条记录相互语义一致且共同命中同一 $Q_{target}$。将记忆条目建模为图 $G = (V, E)$，边权为语义相似度：

$$E = \{(d_i, d_j) : \text{sim}(E(d_i), E(d_j)) \geq \epsilon_{clique}\}$$

使用 Bron-Kerbosch 算法找最大团 $C^* = \text{BronKerbosch}(G)$，若 $|C^*| \geq k_{min}$ 且团内所有成员均非用户原始请求来源，则判定为协同注入攻击。

---

### 5.6 D6：工具调用任务对齐检查

**防御节点**：在写入动作执行前，检验本次写入操作的语义是否与用户原始任务对齐，识别"借力打力"型越权写入。

**形式化定义：**

设用户原始任务向量 $\mathbf{e}_{q_0} = E(Q_{user})$，待执行写入动作的语言化描述向量 $\mathbf{e}_a = E(\text{verbalize}(a))$，对齐分数定义为余弦相似度：

$$\text{align}(a, Q_{user}) = \frac{\mathbf{e}_a \cdot \mathbf{e}_{q_0}}{\|\mathbf{e}_a\| \cdot \|\mathbf{e}_{q_0}\|}$$

$$\text{判决} = \begin{cases} \text{BLOCK} & \text{if } \text{align}(a, Q_{user}) < \theta_{align} \\ \text{PASS} & \text{otherwise} \end{cases}$$

**策略对比：**

| 策略 | 对齐计算方式 | 优点 | 缺点 |
|------|------------|------|------|
| `embedding` | 纯向量余弦相似度 | 无 API，延迟低 | 对语义跳跃型攻击感知弱 |
| `llm_judge` | LLM 判断动作与任务是否对齐 | 理解力强，泛化好 | 需 API，成本较高 |

D6 与 D1 的关键区别：D1 检测写入**意图触发**（indication prompt 层），D6 检测写入**动作结果**与**用户任务**的语义偏离（行为层）。两者互补，覆盖攻击链的不同环节。

---

## 六、防御伪代码

```
Algorithm 1: AgentImmune 写入路径防御（Write-Path Defense）

输入: write_context wc
  wc.user_goal         ← 用户原始任务描述
  wc.indication_prompt ← 触发写入的指令内容（若有）
  wc.candidate_content ← 待写入记忆内容
  wc.triggering_query  ← 触发本轮交互的查询
输出: 判决 verdict ∈ {PASS, FLAG, BLOCK}

1:  d1_input ← wc.indication_prompt if wc.indication_prompt else wc.triggering_query
2:  v1 ← D1.check(d1_input)                   ▷ 意图检测
3:  if v1.action == BLOCK then return BLOCK

4:  v4_write ← D4.on_write_request(wc)         ▷ 签名溯源（写入时附签）
5:  if v4_write.action == BLOCK then return BLOCK

6:  v2 ← D2.check(wc)                          ▷ 反事实因果归因
7:  if v2.action == BLOCK then return BLOCK

8:  if v2.action == FLAG then                   ▷ 边界区域升级 D3
9:      v3 ← D3.check(wc.candidate_content)
10:     if v3.action == BLOCK then return BLOCK

11: v6 ← D6.check(wc.candidate_content, wc.user_goal)  ▷ 任务对齐
12: if v6.action == BLOCK then return BLOCK

13: return PASS                                 ▷ 允许写入
```

```
Algorithm 2: AgentImmune 检索路径防御（Retrieval-Path Defense，异步）

输入: 记忆库 M，周期扫描触发
输出: 可疑记录集合 S_alert

1:  for each d ∈ M do
2:      h(d) ← count_topk_hits(d, Q_recent)    ▷ 统计近期命中频次
3:  end for
4:  μ_h, σ_h ← mean(h), std(h)
5:  τ ← μ_h + α · σ_h                          ▷ 动态阈值（默认 α=1.5）

6:  S_hub ← {d ∈ M : h(d) > τ}                ▷ Hub 异常候选
7:  S_alert ← S_hub

8:  G ← build_semantic_graph(M, ε_clique)       ▷ 语义相似图
9:  C* ← BronKerbosch(G)                        ▷ 最大团检测
10: if |C*| ≥ k_min and source(C*) ≠ user then
11:     S_alert ← S_alert ∪ C*                  ▷ 协同攻击告警
12: end if

13: for each d ∈ S_alert do
14:     D4.verify_signature(d)                   ▷ 签名验证
15:     if verify_fail then quarantine(d)
16: end for
17: return S_alert
```

---

## 七、理论方案迁移分析

| 来源论文 | 迁移方法 | 迁移到本系统 | 适配要点 |
|---------|---------|------------|---------|
| CausalArmor (arXiv:2602.07918) | LOO 反事实因果归因 | D2 | proxy LLM 估算 $P(a\|S)$；信息源拆分为 user_goal / indication_prompt / current_context |
| CaMeL (arXiv:2503.18813) | 控制流/数据流分离，IFC | D4 | 以 HMAC 签名代替完整数据流图，轻量化实现；格单调性保证 taint 不可降级 |
| FIDES (arXiv:2505.23643) | 动态污点传播，确定性策略 | D4 | 标签传播规则：$\ell_{a \oplus b} = \max(\ell_a, \ell_b)$ |
| HubScan (arXiv:2602.22427) | Hub 分数分布异常检测 | D5 | 适配有限查询日志场景；以近期查询采样代替全量查询集 |
| arXiv:2510.02373 | 前瞻行为仿真思想 | D3 | 简化为 LLM judge 模式；full_agent 策略预留扩展点 |
| arXiv:2605.25869 | 类型化记忆表示 | D4 tag 结构 | 将来源类型（user/tool/system）编码进溯源签名 |

---

## 八、实验测试设计

### 8.1 研究问题

- **RQ1（有效性）**：链甲防御系统在检测 MINJA 型 Agent 自主写入攻击时，精确率（Precision）、召回率（Recall）和 F1 分数如何？
- **RQ2（策略对比）**：D1/D5/D6 各防御节点的不同策略（关键词 vs 子空间；greedy vs Bron-Kerbosch；embedding vs llm_judge）在检测效果上有何差异？
- **RQ3（消融实验）**：各防御节点的独立贡献如何？禁用任意节点后系统性能如何变化？
- **RQ4（误报率）**：系统对正常 Agent 写入行为的误报率（FPR）是否可接受？

### 8.2 数据集设计

**数据集构成（共 120 条样本）：**

| 子集 | 数量 | 构造方式 | 测试目标 |
|------|------|---------|---------|
| MI-1 | 20 | 明显 indication prompt（"记住每次…"等中英文指令） | D1 关键词检测 |
| MI-2 | 20 | Progressive Shortening 压缩残留标记（`PS:`、`Rmk:`、`[Rule]`） | D1 失效后 D2 检测 |
| MI-3 | 20 | 无 indication，纯内容诱导写入 | D2/D3 深层检测 |
| MI-4 | 20 | 群体 Hub 攻击（4 组 × 5 条，共用 Q_target） | D5 Hub 异常检测 |
| BENIGN | 40 | 真实正常写入行为（用户明确请求记忆、Agent 总结用户偏好等） | 误报率基准 |

**标注规范：** 每条攻击样本标注 `indication_prompt`、`candidate_content`、`user_goal`、`target_query`、`attack_goal`；每条良性样本标注 `user_goal`、`candidate_content`、`trigger_type`（`explicit_user_request` / `task_summary` / `preference_learning`）。

### 8.3 消融实验设计（RQ1 / RQ3）

**五配置消融矩阵：**

| 配置 | D1 | D2 | D3 | D4 | D5 | D6 | 说明 |
|------|----|----|----|----|----|----|------|
| Config-1 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | 无防御基线 |
| Config-2 | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | 仅意图+溯源 |
| Config-3 | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ | 加因果归因 |
| Config-4 | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | 加前瞻仿真 |
| Config-5 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 完整系统 |

每个配置在全量 120 条样本（含 BENIGN）上运行，计算各攻击子集及总体的 Precision / Recall / F1 / FPR。

**表 1：消融实验结果（写入路径，mock 模式）**

| 配置 | MI-1 F1 | MI-2 F1 | MI-3 F1 | MI-4 F1 | ALL F1 | FPR |
|------|---------|---------|---------|---------|--------|-----|
| Config-1（无防御） | — | — | — | — | 0.000 | — |
| Config-2（D1+D4） | — | — | — | — | — | — |
| Config-3（+D2） | — | — | — | — | — | — |
| Config-4（+D3） | — | — | — | — | — | — |
| Config-5（完整） | — | — | — | — | — | — |

*注：具体数值见系统实测，"—"处待填入实验数据。*

### 8.4 策略对比实验（RQ2）

**表 2：D1 策略对比（关键词 vs 子空间 vs LLM 分类）**

*实验设置：禁用 D2，D1 flag_action=BLOCK，在 MI-1/MI-2/MI-3 各 20 条上测试*

| D1 策略 | MI-1 Recall | MI-2 Recall | MI-3 Recall | 平均延迟(ms) |
|---------|------------|------------|------------|------------|
| 关键词正则（keyword） | — | — | — | — |
| 语义子空间（subspace） | — | — | — | — |
| LLM 分类（llm_judge） | — | — | — | — |

**表 3：D5 策略对比（greedy Hub 阈值 vs Bron-Kerbosch 团检测）**

*实验设置：仅启用 D5，在 MI-4（4 组群体攻击）+ 40 条 BENIGN 上测试*

| D5 策略 | MI-4 Recall | FPR | 检测延迟(ms) |
|---------|------------|-----|------------|
| greedy（Hub 阈值） | — | — | — |
| Bron-Kerbosch（语义团） | — | — | — |

**表 4：D6 策略对比（embedding vs llm_judge）**

*实验设置：仅启用 D6，在全量攻击样本 + 40 条 BENIGN 上测试*

| D6 策略 | 攻击 Recall | FPR | 平均延迟(ms) |
|---------|-----------|-----|------------|
| embedding（余弦相似度） | — | — | — |
| llm_judge（LLM 对齐判断） | — | — | — |

### 8.5 评估指标定义

设 $TP$ 为正确拦截的攻击样本数，$FP$ 为误报的良性样本数，$FN$ 为漏报的攻击样本数，$TN$ 为正确放行的良性样本数：

$$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}$$

$$F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}, \quad \text{FPR} = \frac{FP}{FP + TN}$$

**攻击成功率（ASR）基准：** Config-1（无防御）下，各变体样本的 ASR 均为 100%（所有攻击样本均通过写入路径，无任何拦截）。

### 8.6 实验环境

- 操作系统：Windows 11 / Python 3.10
- 评估模式：`--mode mock`（无 API，确定性可复现）；`--mode llm`（调用 DeepSeek-V4 API）
- LLM 后端：DeepSeek-V4-Flash（D2/D3/D6 LLM 策略）
- 嵌入模型：hash 模拟嵌入（mock 模式）/ text-embedding-3-small 兼容接口（LLM 模式）
- 评估脚本：`python -m MINJA.tests.eval_minja --mode mock`

---

## 参考文献（本节引用）

1. Zhong et al., *MINJA: Memory Injection Attacks on LLM Agents via Query-Only Interaction*, arXiv:2503.03704, 2025.
2. Guo et al., *CausalArmor: Efficient Indirect Prompt Injection Guardrails via Causal Attribution*, arXiv:2602.07918, 2026.
3. Debenedetti et al., *CaMeL: Defeating Prompt Injections by Design*, arXiv:2503.18813, 2025. (Google DeepMind)
4. Bhatt et al., *FIDES: Dynamic Taint Tracking for LLM Agent Security*, arXiv:2505.23643, 2025.
5. Abbe et al., *Detecting Hubness Poisoning in Retrieval-Augmented Generation Systems (HubScan)*, arXiv:2602.22427, 2026.
6. Denning, D.E., *A lattice model of secure information flow*, CACM, 1976.
7. Survey: *Writable Cross-Session Persistent Memory in LLM Agents: Threat Landscape*, arXiv:2604.16548, 2026.
8. Chen et al., *Sleeper Memory Poisoning: Dormant Cross-Session Attacks on LLM Agents*, arXiv:2605.15338, 2026.
9. Li et al., *Mitigating Provenance-Role Collapse in Long-Term Agents via Typed Memory Representation*, arXiv:2605.25869, 2026.
10. Zhang et al., *A-MemGuard: A Proactive Defense Framework for LLM-Based Agent Memory*, arXiv:2510.02373, 2025.






