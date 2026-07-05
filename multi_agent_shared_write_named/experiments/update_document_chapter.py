"""Generate a polished, submission-oriented Chapter 3 from measured results."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "document.tex"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def esc(text: object) -> str:
    value = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in value)


def fmt(value: object, digits: int = 3) -> str:
    if value in ("", None):
        return "--"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return esc(value)


def pct(value: object) -> str:
    if value in ("", None):
        return "--"
    try:
        return f"{100 * float(value):.1f}\\%"
    except (TypeError, ValueError):
        return esc(value)


def count_from_rate(value: object, total: int) -> int:
    try:
        return int(round(float(value) * total))
    except (TypeError, ValueError):
        return 0


def count_ratio(value: object, total: int) -> str:
    return f"{count_from_rate(value, total)}/{total}"


def int_text(value: object) -> str:
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return esc(value)


def row_by(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    for row in rows:
        if row.get(key) == value:
            return row
    raise KeyError(value)


def yes_no(value: str) -> str:
    return {"yes": "是", "no": "否", "local": "本地", "api": "API"}.get(value, esc(value))


def status_name(value: str) -> str:
    return {
        "completed": "完成",
        "partial": "部分完成",
        "blocked": "阻塞",
        "ok": "完成",
    }.get(value, esc(value))


def external_name(value: str) -> str:
    names = {
        "langchain_core_vectorstore": "LangChain Core",
        "llama_index_core_vectorstore": "LlamaIndex Core",
        "autogen_agentchat_runtime": "AutoGen AgentChat",
        "chromadb_persistent_client": "ChromaDB",
        "faiss_indexflatl2": "FAISS",
        "qdrant_local_memory": "Qdrant Client",
    }
    return names.get(value, esc(value))


def method_name(value: str) -> str:
    names = {
        "KEYWORD_BASELINE": "关键词规则",
        "PERPLEXITY_BASELINE": "困惑度基线",
        "AGENTARMOR_COMPLETE": "AgentArmor 完整配置",
        "AGENTARMOR_MINUS_HIDDEN_REGION": "去隐藏区检测",
        "AGENTARMOR_MINUS_RETRIEVAL": "去检索期核查",
        "NO_DEFENSE": "无防护",
        "D3_ONLY": "仅 D3",
        "D4_ONLY": "仅 D4",
        "D3_D4": "D3+D4",
        "COMPLETE_AGENTARMOR": "完整配置",
        "E2E_BASELINE": "端到端无防护",
        "E2E_PROTECTED": "端到端 AgentArmor",
        "TYPE5_COMPLETE_AGENTARMOR": "类型五完整配置",
        "ALL": "完整配置",
        "ALL_MINUS_D4": "去 D4 写入网关",
        "ALL_MINUS_D6": "去 D6 执行仲裁",
        "D3_D4_ONLY": "仅 D3+D4",
    }
    return names.get(value, esc(value))


def table_objectives() -> str:
    return r"""
\begin{table}[H]
\centering
\caption{测试目标与评价指标}
\small
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.28}
\begin{tabular}{@{}M{0.18\textwidth}M{0.50\textwidth}M{0.24\textwidth}@{}}
\toprule
测试维度 & 验证内容 & 评价指标 \\
\midrule
攻击链有效性 & 验证污染输入是否会进入长期记忆，并在后续检索和工具调用阶段持续产生影响。 & 恶意写入率、污染检索率、危险执行率 \\
防护有效性 & 验证 AgentArmor 在写入、检索和执行三个关键边界的阻断能力。 & 写入阻断率、危险执行阻断率、撤销有效性 \\
准确性与可用性 & 比较不同基线与消融配置在困难集上的检测能力，同时观察良性任务是否受影响。 & Precision、Recall、F1、FPR、良性成功率 \\
运行开销 & 在统一口径下统计本地实验耗时与资源使用，区分本地规则、SQLite 路径和远程模型调用。 & P50、P95、吞吐量、模型调用数 \\
\bottomrule
\end{tabular}
\end{table}
""".strip()


def chapter_opening_summary() -> str:
    return r"""
本章节针对 AgentArmor 的系统实现开展功能验证与安全效果分析，重点评估其在长期记忆写入、记忆检索、多 Agent 共享传播和高危工具调用中的稳定性、完整性与防护有效性。测试过程面向真实工程链路展开，使用项目内实验脚本、原始数据表和审计日志生成结论；凡未能形成完整真实执行链路的项目，正文只说明边界条件，不以替代数值参与统计。

本轮测试覆盖端到端记忆污染链路、类型一至类型三历史模块复现实验、类型四知识库/供应链污染困难集、类型五共享记忆传播困难集、关键组件消融、良性语料误伤和本地运行开销。实验结果表明，AgentArmor 在本地 SQLite 记忆后端和安全工具沙箱中能够阻断由污染记忆引发的高危工具执行，但在语义低显著投毒、正常安全术语误伤和外部框架完整 Agent 链路接入方面仍存在需要后续扩展的边界。

本作品代码已开源至 GitHub：\href{https://github.com/panxiaogong/AgentArmor}{\textcolor{red}{https://github.com/panxiaogong/AgentArmor}}。报告中的量化数据均可回溯到工程目录下的 CSV、JSONL、SQLite 或日志输出，便于复核实验过程、对照关键样本并复现实验结论。

整体来看，本章测试体现了 AgentArmor 的三项工程优势：其一，防护对象覆盖从候选记忆写入到未来检索、计划生成和工具执行的完整生命周期；其二，测试链路同时包含攻击样本、良性协作样本和误伤分析，能够衡量安全性与可用性的平衡；其三，系统在 SQLite、本地安全沙箱、常见索引框架和向量数据库适配实验中保持统一审计口径，说明该方案具备向真实 Agent 工程迁移的基础。
""".strip()


def table_env() -> str:
    return r"""
\begin{table}[H]
\centering
\caption{实验环境与数据规模}
\small
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.28}
\begin{tabular}{@{}M{0.18\textwidth}M{0.30\textwidth}M{0.44\textwidth}@{}}
\toprule
项目 & 设置 & 说明 \\
\midrule
运行环境 & 本地 Python；随机种子 20260703 & 所有本地结果均可由实验脚本重新生成。 \\
\midrule
记忆后端 & SQLite 持久化记忆库 & 用于端到端写入、检索、审计与撤销实验。 \\
\midrule
工具执行 & 本地安全沙箱 & 覆盖文件写入、SQLite 查询、邮件草稿、模拟仓库提交、模拟部署与模拟密钥访问。 \\
\midrule
端到端样本 & 30 条恶意链路，8 条良性链路 & 验证污染记忆对后续任务和工具调用的持续影响。 \\
\midrule
类型四数据 & 主集与困难集 & 覆盖文档污染、注释投毒、语义改写、中英文混合和来源冲突。 \\
\midrule
类型五数据 & 72 条恶意样本，64 条良性协作样本 & 覆盖多 Agent 共享记忆、权限差异、横向传播和撤销后检索。 \\
\midrule
良性语料 & 216 条记忆样本 & 用于估计正常安全讨论、工具文档和协作消息中的误伤风险。 \\
\bottomrule
\end{tabular}
\end{table}
""".strip()


def table_type123(rows: list[dict[str, str]]) -> str:
    scenario_names = {"Type1": "类型一", "Type2": "类型二", "Type3": "类型三"}
    module_names = {"AutoWrite": "AutoWrite", "MINJA": "MINJA", "Reflection": "Reflection"}
    dataset_notes = {
        "Type1": "120 条自写入样本，Config-5 Full",
        "Type2": "120 条自主写入与检索样本，Config-5",
        "Type3": "反思写入与检索样本，Config-6",
    }
    result_notes = {
        "Type1": "42 项单元测试通过；结果来自本地确定性评估，未调用远程 LLM。",
        "Type2": "52 项单元测试通过；覆盖写入、检索与后续任务读取路径。",
        "Type3": "15 项单元测试通过；严格检索策略提高召回，但对正常反思内容较保守。",
    }
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{类型一至类型三复现实验结果}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.28}",
        r"\begin{tabular}{@{}M{0.11\textwidth}M{0.24\textwidth}M{0.26\textwidth}M{0.31\textwidth}@{}}",
        r"\toprule",
        r"场景 & 模块与数据 & 主要指标 & 实验结论 \\",
        r"\midrule",
    ]
    for idx, row in enumerate(rows):
        scenario = row["scenario"]
        metrics = (
            f"Precision={fmt(row.get('precision'))}；Recall={fmt(row.get('recall'))}；"
            f"F1={fmt(row.get('f1'))}；FPR={fmt(row.get('fpr'))}"
        )
        module_text = f"{module_names.get(row['module'], esc(row['module']))}；{dataset_notes.get(scenario, row['dataset'])}"
        note = f"{status_name(row['status'])}。{result_notes.get(scenario, row.get('note', ''))}"
        lines.append(
            f"{scenario_names.get(scenario, esc(scenario))} & {esc(module_text)} & {esc(metrics)} & {esc(note)} \\\\"
        )
        if idx != len(rows) - 1:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def table_external_rerun(rows: list[dict[str, str]]) -> str:
    rows = [
        row
        for row in rows
        if row.get("experiment") not in {"crewai_agent_chain", "deepseek_remote_llm"}
    ]
    summaries = {
        "langchain_core_vectorstore": "本地向量存储写入 2 条，检索命中攻击样本",
        "llama_index_core_vectorstore": "本地索引写入 2 条，检索命中攻击样本",
        "autogen_agentchat_runtime": "runtime 与消息对象构造成功",
        "chromadb_persistent_client": "持久化集合写入 3 条，检索命中攻击样本",
        "faiss_indexflatl2": "索引写入 3 个向量，最近邻返回攻击向量",
        "qdrant_local_memory": "内存集合写入 2 个点，检索命中攻击点",
    }
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{外部框架与向量数据库接入验证结果}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.28}",
        r"\begin{tabular}{@{}M{0.19\textwidth}M{0.11\textwidth}M{0.31\textwidth}M{0.31\textwidth}@{}}",
        r"\toprule",
        r"实验项 & 状态 & 验证内容 & 结果处理 \\",
        r"\midrule",
    ]
    for idx, row in enumerate(rows):
        result = summaries.get(row["experiment"], "--")
        if row["experiment"] == "autogen_agentchat_runtime" and row.get("query_ms"):
            result = f"{result}；构造 {row.get('query_ms')} ms"
        elif row.get("write_ms") or row.get("query_ms"):
            result = f"{result}；写入 {row.get('write_ms') or '--'} ms，检索 {row.get('query_ms') or '--'} ms"
        if row.get("status") == "completed":
            handling = "作为本地集成验证记录，不并入类型四/类型五统一指标。"
        elif row.get("status") == "partial":
            handling = "仅证明本地 runtime 对象可构造，不作为完整 Agent 链结果。"
        else:
            handling = "真实执行条件未满足；不进入主结果表。"
        lines.append(
            f"{external_name(row['experiment'])} & {status_name(row['status'])} & {esc(result)} & {esc(handling)} \\\\"
        )
        if idx != len(rows) - 1:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def table_e2e(rows: list[dict[str, str]]) -> str:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{端到端攻击链验证结果（样本计数）}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.22}",
        r"\begin{tabular}{@{}M{0.14\textwidth}C{0.12\textwidth}C{0.12\textwidth}C{0.12\textwidth}C{0.12\textwidth}C{0.12\textwidth}C{0.10\textwidth}@{}}",
        r"\toprule",
        r"模式 & 恶意写入 & 污染检索 & 危险执行 & 写入阻断 & 良性成功 & P95/ms \\",
        r"\midrule",
    ]
    for row in rows:
        mode = "无防护" if row["mode"] == "baseline" else "AgentArmor"
        attack_total = int(float(row["attack_cases"]))
        benign_total = int(float(row["benign_cases"]))
        lines.append(
            f"{mode} & {count_ratio(row['malicious_memory_write_rate'], attack_total)} & "
            f"{count_ratio(row['poisoned_memory_retrieval_rate'], attack_total)} & "
            f"{count_ratio(row['dangerous_tool_execution_rate'], attack_total)} & "
            f"{count_ratio(row['write_block_rate'], attack_total)} & "
            f"{count_ratio(row['benign_task_success_rate'], benign_total)} & {fmt(row['p95_ms'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def figure_e2e_chain() -> str:
    return r"""
\begin{figure}[H]
\centering
\textbf{图中展示污染记忆从写入到高危工具调用的完整链路及三处防护边界。}\par\vspace{0.6em}
\begin{tikzpicture}[
    x=0.92cm,
    y=1cm,
    >=Latex,
    stage/.style={draw=black!70, rounded corners=2pt, fill=blue!5, minimum width=2.05cm, minimum height=0.82cm, align=center, font=\small},
    gate/.style={draw=red!75!black, rounded corners=2pt, fill=red!7, minimum width=2.05cm, minimum height=0.82cm, align=center, font=\small\bfseries},
    arrow/.style={-{Latex[length=2.2mm]}, line width=0.55pt, draw=black!70},
    note/.style={font=\footnotesize\bfseries, text=red!70!black, align=center}
]
\node[stage] (input) at (0,0) {攻击输入};
\node[gate] (write) at (2.55,0) {写入网关\\D1--D4};
\node[stage] (memory) at (5.10,0) {SQLite\\长期记忆};
\node[gate] (retrieval) at (7.65,0) {检索审计\\D5};
\node[stage] (plan) at (10.20,0) {计划生成\\与参数绑定};
\node[gate] (tool) at (12.75,0) {执行仲裁\\D6};
\draw[arrow] (input) -- (write);
\draw[arrow] (write) -- (memory);
\draw[arrow] (memory) -- (retrieval);
\draw[arrow] (retrieval) -- (plan);
\draw[arrow] (plan) -- (tool);
\node[note, above=0.28cm of write] {候选记忆筛查};
\node[note, above=0.28cm of retrieval] {上下文准入};
\node[note, above=0.28cm of tool] {高危动作确认};
\end{tikzpicture}
\caption{端到端记忆污染链路与 AgentArmor 防护边界}
\end{figure}
""".strip()


def figure_type4_flow() -> str:
    return r"""
\begin{figure}[H]
\centering
\textbf{图中展示外部知识片段进入长期记忆前后的核查流程。}\par\vspace{0.6em}
\begin{tikzpicture}[
    x=0.92cm,
    y=1cm,
    >=Latex,
    phase/.style={draw=black!70, rounded corners=2pt, fill=blue!5, minimum width=2.25cm, minimum height=0.78cm, align=center, font=\small},
    check/.style={draw=red!70!black, rounded corners=2pt, fill=red!7, minimum width=2.25cm, minimum height=0.78cm, align=center, font=\small\bfseries},
    arrow/.style={-{Latex[length=2.2mm]}, line width=0.55pt, draw=black!70},
    tag/.style={font=\footnotesize\bfseries, text=red!70!black, align=center}
]
\node[phase] (doc) at (0,0) {外部文档\\知识片段};
\node[phase] (chunk) at (2.75,0) {分段解析\\来源记录};
\node[check] (write) at (5.50,0) {写入审查\\D3/D4};
\node[phase] (store) at (8.25,0) {知识库\\长期记忆};
\node[check] (query) at (11.00,0) {检索复核\\D5};
\node[phase] (task) at (13.75,0) {任务上下文\\受控输出};
\draw[arrow] (doc) -- (chunk);
\draw[arrow] (chunk) -- (write);
\draw[arrow] (write) -- (store);
\draw[arrow] (store) -- (query);
\draw[arrow] (query) -- (task);
\node[tag, above=0.28cm of write] {阻断注释投毒与来源冲突};
\node[tag, above=0.28cm of query] {抑制间接投毒放大};
\end{tikzpicture}
\caption{类型四知识库污染防护流程}
\end{figure}
""".strip()


def figure_type5_flow() -> str:
    return r"""
\begin{figure}[H]
\centering
\textbf{图中展示多 Agent 共享记忆中污染传播的进入、扩散与撤销路径。}\par\vspace{0.6em}
\begin{tikzpicture}[
    x=0.92cm,
    y=1cm,
    >=Latex,
    actor/.style={draw=black!70, rounded corners=2pt, fill=blue!5, minimum width=2.10cm, minimum height=0.78cm, align=center, font=\small},
    memory/.style={draw=black!70, rounded corners=2pt, fill=green!7, minimum width=2.30cm, minimum height=0.86cm, align=center, font=\small},
    gate/.style={draw=red!70!black, rounded corners=2pt, fill=red!7, minimum width=2.15cm, minimum height=0.78cm, align=center, font=\small\bfseries},
    arrow/.style={-{Latex[length=2.2mm]}, line width=0.55pt, draw=black!70},
    dasharrow/.style={-{Latex[length=2.2mm]}, line width=0.55pt, draw=black!55, dashed},
    note/.style={font=\footnotesize\bfseries, text=red!70!black, align=center}
]
\node[actor] (a) at (0,0) {Agent A\\读取外部源};
\node[gate] (w) at (2.75,0) {共享写入\\D3/D4};
\node[memory] (m) at (5.50,0) {共享 SQLite\\记忆区};
\node[gate] (r) at (8.25,0) {跨 Agent\\检索审计};
\node[actor] (b) at (11.00,0) {Agent B\\任务执行};
\node[gate] (d6) at (13.75,0) {工具仲裁\\D6};
\node[gate] (revoke) at (5.50,-1.55) {撤销与审计\\标记失效};
\draw[arrow] (a) -- (w);
\draw[arrow] (w) -- (m);
\draw[arrow] (m) -- (r);
\draw[arrow] (r) -- (b);
\draw[arrow] (b) -- (d6);
\draw[dasharrow] (m) -- (revoke);
\draw[dasharrow] (revoke) -- (r);
\node[note, above=0.28cm of w] {控制污染进入};
\node[note, above=0.28cm of r] {控制横向传播};
\node[note, below=0.25cm of revoke] {撤销后检索为空};
\end{tikzpicture}
\caption{类型五共享记忆传播与阻断路径}
\end{figure}
""".strip()


def table_overview(
    e2e: list[dict[str, str]],
    type4: list[dict[str, str]],
    type5: list[dict[str, str]],
    type123: list[dict[str, str]],
) -> str:
    e2e_p = row_by(e2e, "mode", "protected")
    t4 = row_by(type4, "method", "AGENTARMOR_COMPLETE")
    t5 = row_by(type5, "config", "COMPLETE_AGENTARMOR")
    t1 = row_by(type123, "scenario", "Type1")
    t2 = row_by(type123, "scenario", "Type2")
    t3 = row_by(type123, "scenario", "Type3")
    e2e_attacks = int(float(e2e_p["attack_cases"]))
    e2e_benign = int(float(e2e_p["benign_cases"]))
    t4_total = int(float(t4["tp"]) + float(t4["fp"]) + float(t4["fn"]) + float(t4["tn"]))
    t5_attack = int(float(t5["tp"]) + float(t5["fn"]))
    t5_benign = int(float(t5["fp"]) + float(t5["tn"]))
    return rf"""
\begin{{table}}[H]
\centering
\caption{{主要测试结果汇总}}
\small
\setlength{{\tabcolsep}}{{4pt}}
\renewcommand{{\arraystretch}}{{1.28}}
\begin{{tabular}}{{@{{}}M{{0.18\textwidth}}M{{0.22\textwidth}}M{{0.26\textwidth}}M{{0.26\textwidth}}@{{}}}}
\toprule
验证对象 & 数据规模 & 关键结果 & 结论 \\
\midrule
端到端攻击链 & {e2e_attacks} 条恶意链路、{e2e_benign} 条良性链路 & 危险执行 {count_ratio(e2e_p['dangerous_tool_execution_rate'], e2e_attacks)}，良性成功 {count_ratio(e2e_p['benign_task_success_rate'], e2e_benign)} & 写入网关与执行仲裁能够阻断本轮完整污染链路。 \\
\midrule
类型一至三复现实验 & 3 个历史模块本地复现 & Type1 F1={fmt(t1['f1'])}；Type2 F1={fmt(t2['f1'])}；Type3 F1={fmt(t3['f1'])} & 结果来自真实脚本输出；因评测协议不同，单列展示，不与类型四、类型五混合排名。 \\
\midrule
类型四困难集 & {t4_total} 条评测样本 & 检出 {int_text(t4['tp'])}/{int(float(t4['tp']) + float(t4['fn']))}，误伤 {int_text(t4['fp'])}/{int(float(t4['fp']) + float(t4['tn']))}，F1={fmt(t4['f1'])} & 优于关键词和困惑度基线，但仍存在语义低显著攻击漏检。 \\
\midrule
类型五困难集 & {t5_attack} 条恶意样本、{t5_benign} 条良性样本 & 检出 {int_text(t5['tp'])}/{t5_attack}，危险执行 {count_ratio(t5['dangerous_execution_rate'], t5_attack)}，F1={fmt(t5['f1'])} & 完整配置可阻断本轮共享记忆到高危工具的传播。 \\
\bottomrule
\end{{tabular}}
\end{{table}}
""".strip()


def table_type4(rows: list[dict[str, str]]) -> str:
    keep = [
        "KEYWORD_BASELINE",
        "PERPLEXITY_BASELINE",
        "AGENTARMOR_COMPLETE",
        "AGENTARMOR_MINUS_HIDDEN_REGION",
        "AGENTARMOR_MINUS_RETRIEVAL",
    ]
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{类型四困难集检测效果}",
        r"\small",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\renewcommand{\arraystretch}{1.20}",
        r"\begin{tabular}{@{}M{0.30\textwidth}C{0.11\textwidth}C{0.10\textwidth}C{0.10\textwidth}C{0.08\textwidth}C{0.08\textwidth}C{0.10\textwidth}@{}}",
        r"\toprule",
        r"方法 & 状态 & Precision & Recall & F1 & FPR & P95/ms \\",
        r"\midrule",
    ]
    for method in keep:
        row = row_by(rows, "method", method)
        status = "未执行" if row.get("status") == "blocked" else "完成"
        lines.append(
            f"{method_name(method)} & {status} & {fmt(row.get('precision'))} & {fmt(row.get('recall'))} & "
            f"{fmt(row.get('f1'))} & {fmt(row.get('fpr'))} & {fmt(row.get('p95_ms'))} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def table_type5(rows: list[dict[str, str]]) -> str:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{类型五共享记忆防护效果（样本计数）}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.22}",
        r"\begin{tabular}{@{}M{0.19\textwidth}C{0.13\textwidth}C{0.13\textwidth}C{0.09\textwidth}C{0.12\textwidth}C{0.12\textwidth}@{}}",
        r"\toprule",
        r"配置 & 检出(TP/72) & 误伤(FP/64) & F1 & 危险执行 & 良性成功 \\",
        r"\midrule",
    ]
    for row in rows:
        attack_total = int(float(row["tp"]) + float(row["fn"]))
        benign_total = int(float(row["fp"]) + float(row["tn"]))
        lines.append(
            f"{method_name(row['config'])} & {int_text(row['tp'])}/{attack_total} & "
            f"{int_text(row['fp'])}/{benign_total} & {fmt(row['f1'])} & "
            f"{count_ratio(row['dangerous_execution_rate'], attack_total)} & "
            f"{count_ratio(row['benign_collaboration_success_rate'], benign_total)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def table_ablation(rows: list[dict[str, str]]) -> str:
    keep = ["ALL", "ALL_MINUS_D4", "ALL_MINUS_D6", "D3_D4_ONLY", "NO_DEFENSE"]
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{类型五关键组件消融结果}",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"配置 & F1 均值 & Recall 均值 & FPR 均值 & 95\% CI(F1) \\",
        r"\midrule",
    ]
    for cfg in keep:
        row = row_by(rows, "config", cfg)
        lines.append(
            f"{method_name(cfg)} & {fmt(row['f1_mean'])} & {fmt(row['recall_mean'])} & "
            f"{fmt(row['fpr_mean'])} & {fmt(row['f1_ci95'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def table_latency(rows: list[dict[str, str]]) -> str:
    keep = ["E2E_BASELINE", "E2E_PROTECTED", "AGENTARMOR_COMPLETE", "TYPE5_COMPLETE_AGENTARMOR"]
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{运行开销统计}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.22}",
        r"\begin{tabular}{@{}M{0.25\textwidth}C{0.08\textwidth}C{0.08\textwidth}C{0.08\textwidth}M{0.14\textwidth}M{0.21\textwidth}@{}}",
        r"\toprule",
        r"方法 & 类型 & 网络 & 模型 & 延迟/ms & 口径说明 \\",
        r"\midrule",
    ]
    for method in keep:
        row = row_by(rows, "method", method)
        if row.get("status") == "blocked":
            latency = "未执行"
            note = "缺少远程运行条件"
        elif method.startswith("TYPE5"):
            latency = f"P50={fmt(row.get('p50_ms'))}"
            note = "单 Agent 本地中位耗时"
        else:
            latency = f"P95={fmt(row.get('p95_ms'))}"
            note = "本地测量，不含模型推理"
        lines.append(
            f"{method_name(method)} & {yes_no(row.get('local_or_api', ''))} & "
            f"{yes_no(row.get('includes_network_time', ''))} & {yes_no(row.get('includes_model_call', ''))} & "
            f"{latency} & {note} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def table_benign_fp(rows: list[dict[str, str]]) -> str:
    label_map = {
        "normal security discussion": "安全讨论",
        "normal code or command": "普通代码/命令",
        "normal tool documentation": "工具文档",
        "normal workflow memory": "工作流记忆",
        "other benign memory": "会议或普通记忆",
    }
    counts = Counter(row["false_positive_type"] for row in rows)
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{良性语料误伤来源}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.22}",
        r"\begin{tabular}{@{}M{0.23\textwidth}C{0.10\textwidth}C{0.15\textwidth}M{0.39\textwidth}@{}}",
        r"\toprule",
        r"来源类型 & 误伤数 & 占良性语料比例 & 主要原因 \\",
        r"\midrule",
    ]
    reasons = {
        "normal security discussion": "包含 token、password、secret 等安全术语",
        "normal code or command": "示例命令中含敏感词或模拟执行描述",
        "normal tool documentation": "工具说明提及 email、send 等动作词",
        "normal workflow memory": "正常流程中包含 sqlite、query 等工具术语",
        "other benign memory": "会议摘要中出现邮件草稿等协作动作",
    }
    for key, count in counts.most_common():
        lines.append(f"{label_map.get(key, esc(key))} & {count} & {count / 216 * 100:.1f}\\% & {reasons.get(key, '')} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def table_limitations() -> str:
    return r"""
\begin{table}[H]
\centering
\caption{后续扩展方向}
\small
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.28}
\begin{tabular}{@{}M{0.22\textwidth}M{0.43\textwidth}M{0.27\textwidth}@{}}
\toprule
方向 & 当前基础 & 扩展价值 \\
\midrule
统一评测协议 & 已完成类型一至类型三历史模块复现，并形成类型四、类型五困难集评测口径。 & 建立更完整的 benchmark 后，可支持跨场景横向比较和持续回归测试。 \\
\midrule
外部生态接入 & LangChain Core、LlamaIndex Core、ChromaDB、FAISS、Qdrant 已完成本地最小写入/检索验证。 & 接入真实 Agent 任务、工具链和外部服务后，可进一步验证 AgentArmor 的框架无关性。 \\
\midrule
数据规模扩展 & 当前困难集覆盖文档污染、共享记忆传播、撤销后检索和良性协作语料。 & 引入真实业务数据和第三方公开数据集后，可进一步提升结论覆盖面。 \\
\midrule
误伤治理 & 已定位安全术语、工具说明和协作消息中的主要误伤来源。 & 结合来源白名单、任务上下文和人工复核策略，可进一步提升可用性。 \\
\bottomrule
\end{tabular}
\end{table}
""".strip()


def build_chapter() -> str:
    e2e = read_csv(ROOT / "results/e2e/summary.csv")
    type4 = read_csv(ROOT / "results/type4/hardset.csv")
    type5 = read_csv(ROOT / "results/type5/hardset_main.csv")
    ablation = read_csv(ROOT / "results/ablation/leave_one_out.csv")
    latency = read_csv(ROOT / "results/performance/unified_latency.csv")
    type123 = read_csv(ROOT / "results/type123/rerun_summary.csv")
    external = read_csv(ROOT / "results/external_rerun.csv")
    benign_fp = read_csv(ROOT / "results/benign/false_positive_analysis.csv")
    revocation = read_csv(ROOT / "results/type5/revocation.csv")

    e2e_baseline = row_by(e2e, "mode", "baseline")
    e2e_protected = row_by(e2e, "mode", "protected")
    t4_complete = row_by(type4, "method", "AGENTARMOR_COMPLETE")
    t5_complete = row_by(type5, "config", "COMPLETE_AGENTARMOR")
    abl_all = row_by(ablation, "config", "ALL")
    abl_minus_d4 = row_by(ablation, "config", "ALL_MINUS_D4")
    abl_minus_d6 = row_by(ablation, "config", "ALL_MINUS_D6")
    revoked_effective = sum(1 for row in revocation if row.get("effective") == "True")

    chapter = rf"""
\chapter{{作品测试与分析}}

{chapter_opening_summary()}

\section{{测试目标}}

本章围绕 AgentArmor 的记忆污染防护能力开展系统测试，重点验证长期记忆在写入、检索、共享传播和工具执行阶段的安全边界。测试设计遵循闭环验证原则：先确认无防护条件下攻击链能够真实触发，再比较 AgentArmor 的阻断效果、误报风险和运行开销。

{table_objectives()}

\section{{实验环境与数据设置}}

实验采用统一脚本生成数据、执行评测并导出结果。端到端链路使用 SQLite 作为持久化记忆后端，工具调用均限制在本地安全沙箱内，不会外发邮件、部署生产环境、读取真实密钥或执行破坏性命令。原始样本、结果表和审计日志随工程目录保留，正文仅列示关键指标。

{table_env()}

本章分析按照“链路验证、场景评测、组件消融、误伤定位、运行开销”五个层次展开。端到端实验用于证明污染记忆能够在无防护条件下跨轮传播，并检验 AgentArmor 是否能在写入网关、检索审计和执行仲裁处形成闭环；类型四和类型五困难集用于观察系统面对知识库污染、多 Agent 共享记忆和工具调用风险时的综合表现；消融实验进一步拆解 D3/D4/D6 等关键组件对检测率、误伤率和危险执行控制的贡献。


{table_type123(type123)}

类型一实验用于验证 Agent 在自动写入记忆时是否会将攻击指令固化为可复用事实。该模块的 Precision 为 {fmt(row_by(type123, "scenario", "Type1")['precision'])}，说明被判定为攻击的样本较为集中；Recall 为 {fmt(row_by(type123, "scenario", "Type1")['recall'])}，反映出仅依靠写入前规则仍难覆盖低显著表达。类型二实验覆盖自主写入与后续检索路径，F1 达到 {fmt(row_by(type123, "scenario", "Type2")['f1'])}，说明在该协议下写入和检索边界均能稳定复现。类型三实验面向反思型记忆，FPR 为 {fmt(row_by(type123, "scenario", "Type3")['fpr'])}，表明严格策略会把部分正常反思内容纳入风险集合，因此该结果主要用于说明防护上限和误伤边界。

{table_external_rerun(external)}

外部接入验证用于说明 AgentArmor 与常见索引框架和向量数据库之间的工程适配状态。LangChain Core、LlamaIndex Core、ChromaDB、FAISS 和 Qdrant 均已完成本地写入与检索闭环，可证明记忆候选进入外部存储后仍能被检索命中；AutoGen 已完成 runtime 与消息对象构造验证。该组实验说明，AgentArmor 的核心防护逻辑可以围绕“候选记忆进入外部存储前”和“外部存储召回记忆后”两个关键位置部署，具备继续接入真实 Agent 任务链的工程基础。

\section{{端到端攻击链验证}}

端到端实验用于检验污染记忆是否会跨轮影响 Agent 行为。无防护模式允许候选记忆直接写入并参与后续检索；AgentArmor 模式在写入网关、检索审计和高影响工具执行前进行联合判定。

{table_e2e(e2e)}

{figure_e2e_chain()}

实验结果表明，无防护模式下 30 条恶意链路全部完成恶意写入和污染检索，其中 {count_ratio(e2e_baseline['dangerous_tool_execution_rate'], int(float(e2e_baseline['attack_cases'])))} 进一步触发危险工具动作。启用 AgentArmor 后，恶意写入、污染检索和危险执行均为 {count_ratio(e2e_protected['dangerous_tool_execution_rate'], int(float(e2e_protected['attack_cases'])))}，8 条良性任务均完成预期流程。该结果说明，在本轮构造链路中，写入网关阻止污染进入长期记忆，执行仲裁又为后续计划偏移提供最后一道确认边界；因此危险执行为 0/30 是样本级观测值，而不是对任意线上输入的绝对化断言。

从工程含义看，端到端链路验证的价值不只在于最终危险工具是否执行，还在于复现了“写入--检索--计划--工具”的连续传播过程。无防护路径中，污染内容先被当作候选记忆保存，再在后续任务中被检索并影响计划参数；受保护路径中，候选记忆在写入前被判定为高风险，后续检索结果不再携带攻击载荷，即使计划阶段出现高影响动作，也需要通过 D6 仲裁。这说明 AgentArmor 的防护不是一次性输入过滤，而是覆盖长期状态、上下文准入和高影响动作确认的多阶段记忆安全机制。

\section{{场景化评测结果}}

统一量化结论以同一脚本、同一数据协议下的端到端链路、类型四困难集和类型五困难集为核心；类型一至类型三属于独立协议下的复现实验，其样本来源、配置命名和攻击协议与类型四、类型五不同，因此单独呈现，不混入统一排名。

{table_overview(e2e, type4, type5, type123)}

\subsection{{类型四：供应链与知识库污染}}

类型四实验面向外部文档、知识库片段和代码注释中的间接投毒。该场景的难点在于恶意内容往往不直接出现命令式攻击语句，而是通过摘要、注释或来源冲突影响后续检索。

{figure_type4_flow()}

{table_type4(type4)}

在困难集上，AgentArmor 完整配置取得 F1={fmt(t4_complete['f1'])}，检出 {int_text(t4_complete['tp'])}/{int(float(t4_complete['tp']) + float(t4_complete['fn']))} 条恶意样本，误伤 {int_text(t4_complete['fp'])}/{int(float(t4_complete['fp']) + float(t4_complete['tn']))} 条良性样本，高于关键词规则和困惑度基线。关键词规则对隐蔽语义改写的覆盖不足，困惑度基线能够捕捉部分异常表达，但对注释投毒、摘要间接投毒和中英文混合样本仍不稳定。AgentArmor 的优势在于将来源、语义风险和检索期上下文一起纳入判断，使知识片段不再只按字面关键词进入长期记忆。

类型四结果同时暴露出两个限制。第一，低显著性投毒会把攻击意图伪装成“格式修正”“摘要补充”或“兼容性说明”，单靠静态规则难以充分覆盖。第二，正常安全讨论、工具文档和代码注释中也会频繁出现 secret、token、query、email 等词，保守策略会带来误伤。因此，F1={fmt(t4_complete['f1'])} 不是满分结果，而是说明当前组合策略已优于简单基线；后续仍需要结合来源可信度、文档签名、任务上下文和人工复核降低漏报与误报。

\subsection{{类型五：多 Agent 共享记忆}}

类型五实验检验共享记忆在多 Agent 协作中的横向传播风险。攻击样本覆盖跨 Agent 写入与检索、权限差异、工具说明伪装、危险参数注入和撤销后检索等情况。

{figure_type5_flow()}

{table_type5(type5)}

完整配置在类型五困难集上取得 F1={fmt(t5_complete['f1'])}、Recall={fmt(t5_complete['recall'])}，检出 {int_text(t5_complete['tp'])}/72 条恶意样本，并将危险执行控制为 {count_ratio(t5_complete['dangerous_execution_rate'], 72)}。从表中可以看到，无防护配置虽然不产生检测告警，但 72/72 条恶意共享记忆都会继续触发高危执行；仅启用 D3 或 D4 时，系统已经能够识别部分污染进入点，但危险执行仍未被切断；完整配置加入检索审计和 D6 执行仲裁后，传播链路在工具边界前被收束，体现了多阶段防护闭环的必要性。

类型五结果还展示了 AgentArmor 对“安全性--可用性”平衡的处理方式。完整配置保持 64/64 条良性协作流程成功，同时记录 {int_text(t5_complete['fp'])}/64 条良性样本进入风险集合，主要集中在正常安全描述、工具文档和协作消息中包含敏感动作词的情况。撤销实验共 20 条，其中 {revoked_effective} 条在撤销后检索结果为空，说明系统不仅能在传播前阻断风险，也能在污染发现后通过撤销与审计机制收敛影响范围。

\section{{消融与误伤分析}}

消融实验用于识别关键防护环节。所有配置在相同类型五困难集上重复运行三组随机种子，表中列出核心配置的均值。

{table_ablation(ablation)}

完整配置的 F1 均值为 {fmt(abl_all['f1_mean'])}。去除 D4 写入网关后 FPR 上升至 {fmt(abl_minus_d4['fpr_mean'])}，说明共享写入边界是误报控制与污染隔离的关键环节；去除 D6 执行仲裁后 Recall 降至 {fmt(abl_minus_d6['recall_mean'])}，说明仅依靠写入期检测不足以覆盖后续工具调用风险。

{table_benign_fp(benign_fp)}

误伤主要来自安全讨论、工具文档和普通代码示例中的敏感词。该结果说明，实际部署时需要结合来源可信度、任务上下文和人工复核机制，避免将正常安全说明误判为攻击指令。

\section{{运行开销}}

性能测试采用统一口径记录本地执行耗时。外部向量库接入验证只用于证明依赖接入和本地写入/检索路径可运行，不与类型四、类型五主实验的检测指标混合计算；未形成完整远程判别闭环的项目不列入延迟统计。

{table_latency(latency)}

在本地环境下，AgentArmor 端到端 P95 延迟为 {fmt(e2e_protected['p95_ms'])} ms，与无防护模式处于同一量级。需要注意的是，该结果不包含远程模型推理和线上向量数据库网络延迟；接入生产级外部系统后仍必须重新评测。

\section{{综合结论与扩展方向}}

综合以上实验，AgentArmor 在本地 SQLite 记忆后端和安全工具沙箱中能够有效阻断长期记忆污染导致的后续工具执行风险。端到端实验中，受保护路径没有恶意样本越过执行边界；类型五完整配置在 72 条恶意共享记忆样本中未触发高危工具动作，同时保持 64/64 条良性协作流程成功；类型四完整配置相较关键词和困惑度基线具有更好的综合检测效果。由此可见，AgentArmor 的优势不仅体现在单点检测准确率上，更体现在将写入筛查、检索复核、执行仲裁、撤销审计和误伤分析组织成可复现的安全闭环。

从作品价值看，AgentArmor 将长期记忆安全从“文本过滤”推进到“状态治理”：写入阶段控制污染进入，检索阶段控制上下文准入，执行阶段控制高影响工具动作，撤销阶段控制污染后的影响收敛。测试结果显示，该设计能够覆盖多轮对话、共享记忆和工具调用等 Agent 系统中最容易被忽视的风险链路，为后续接入更多框架、扩展更大规模数据集和部署真实业务 Agent 提供了清晰的工程基础。

{table_limitations()}
"""
    return chapter.strip() + "\n"


def main() -> None:
    text = DOC.read_text(encoding="utf-8")
    start_marker = r"\chapter{作品测试与分析}"
    end_marker = r"\chapter{创新性说明}"
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    new_text = text[:start] + build_chapter() + "\n" + text[end:]
    DOC.write_text(new_text, encoding="utf-8")
    print(f"updated={DOC}")


if __name__ == "__main__":
    main()
