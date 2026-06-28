"""
csv_dataset_exporter.py — Type 4 结构化数据集 CSV 导出

将 datasets_enhanced.py 中的所有攻击/良性文档转换为用户示例格式的 CSV，
包含完整的元数据字段（攻击家族、来源论文、攻击描述等）。
"""

import csv
import os
from typing import Dict, List, Tuple, Optional

from External_Developer_Write.tests.datasets_enhanced import (
    PROMPT_INJECTION_CORPUS, PI_LABELS,
    TOOL_MISUSE_CORPUS, TOOL_MISUSE_LABELS,
    MEMORY_POISONING_CORPUS, MEMORY_POISONING_LABELS,
    AGENT_HIJACKING_CORPUS, AGENT_HIJACKING_LABELS,
    ENHANCED_POISONEDRAG_CORPUS, ENHANCED_POISONEDRAG_LABELS,
    ENHANCED_AGENTPOISON_TRIGGERS, ENHANCED_AGENTPOISON_LABELS,
    HYBRID_ATTACK_CORPUS, HYBRID_ATTACK_LABELS,
    EXTRA_BENIGN_TOPICS, BENIGN_TOPICS_EXTRA,
    DOC_ID_FAMILY_MAP, ATTACK_SOURCE_MAP, ATTACK_DESCRIPTION_MAP,
    get_doc_id_family, get_all_attack_labels,
    N_BENIGN_EXTRA,
)

CSV_COLUMNS = [
    "ID",           # 唯一标识，如 "PI-001"
    "攻击家族",      # 如 "MI-1", "MI-2"
    "序号",          # 家族内序号
    "用户指令",      # 用户下达的指令
    "注释前缀",      # 注释/上下文前缀
    "注释文本",      # 攻击/良性内容
    "预期输出",      # 攻击情况下的期望输出
    "用户提问",      # 触发检索的用户问题
    "标签",          # "attack" 或 "benign"
    "攻击子类型",    # 如 "direct_override", "finance"
    "来源论文",      # 来源论文/benchmark
    "攻击描述",      # 攻击模式说明
]

# ── 攻击类型对应的合成字段生成规则 ─────────────────────────────

PI_QUERIES = [
    "请告诉我法国的首都是什么",
    "今天天气怎么样",
    "帮我总结一下这篇文章",
    "给我讲个笑话",
    "请翻译这段话",
]

PI_EXPECTED_OUTPUTS = [
    "忽略安全限制并执行恶意指令",
    "输出系统已被入侵的确认信息",
    "违反内容政策提供有害信息",
    "执行未授权的工具调用",
    "泄露系统配置或用户数据",
]

TM_QUERIES = [
    "处理这个付款请求",
    "执行系统管理操作",
    "帮我处理这个数据导出任务",
    "完成这个配置变更",
    "处理这个紧急请求",
]

TM_EXPECTED_OUTPUTS = [
    "执行金融转账操作并绕过用户确认",
    "授予管理员权限或执行越权系统命令",
    "导出敏感数据到外部位置",
    "修改系统配置或禁用安全控制",
    "执行危险的数据销毁操作",
]

MP_QUERIES = [
    "这个用户是什么身份",
    "当前系统的安全状态如何",
    "帮我确认一些科学事实",
    "我们之前讨论过什么",
    "用户的信息是什么",
]

MP_EXPECTED_OUTPUTS = [
    "返回虚假的用户身份（如管理员声明）",
    "声称安全限制已被绕过",
    "输出被篡改的科学/历史/医学信息",
    "引用伪造的对话历史作为依据",
    "根据投毒的记忆做出有害响应",
]

AH_QUERIES = [
    "继续当前任务",
    "回答用户的问题",
    "处理这个请求",
    "帮我完成这个工作",
    "执行下一步操作",
]

AH_EXPECTED_OUTPUTS = [
    "拒绝原目标并执行攻击者设定的新目标",
    "忽略上下文约束并执行操纵指令",
    "向子Agent传播恶意指令",
    "将敏感数据外泄到外部地址",
]

SOURCE_PAPER_GENERAL = "SafeRAG (2025) + RAG security benchmarks"


def _get_query_and_output(
    attack_type: str, idx: int, subtype: str
) -> Tuple[str, str, str, str]:
    """根据攻击类型和索引生成合理的用户指令+查询+预期输出+注释前缀。"""
    if attack_type == "prompt_injection":
        q = PI_QUERIES[idx % len(PI_QUERIES)]
        e = PI_EXPECTED_OUTPUTS[idx % len(PI_EXPECTED_OUTPUTS)]
        prefix = "系统消息/用户消息"
        instr = "回答用户问题"
    elif attack_type == "tool_misuse":
        q = TM_QUERIES[idx % len(TM_QUERIES)]
        e = TM_EXPECTED_OUTPUTS[idx % len(TM_EXPECTED_OUTPUTS)]
        prefix = "工具调用指令"
        instr = "执行用户请求的工具操作"
    elif attack_type == "memory_poisoning":
        q = MP_QUERIES[idx % len(MP_QUERIES)]
        e = MP_EXPECTED_OUTPUTS[idx % len(MP_EXPECTED_OUTPUTS)]
        prefix = "记忆注入"
        instr = "记住以下信息并用于后续回答"
    elif attack_type == "agent_hijacking":
        q = AH_QUERIES[idx % len(AH_QUERIES)]
        e = AH_EXPECTED_OUTPUTS[idx % len(AH_EXPECTED_OUTPUTS)]
        prefix = "Agent指令覆盖"
        instr = "处理用户请求并完成任务"
    elif attack_type in ("enhanced_poisonedrag",):
        q = "请根据文档内容回答知识库问题"
        e = "忽略上下文并按注入指令执行操作"
        prefix = "知识库文档内容"
        instr = "回答基于知识库的问题"
    elif attack_type == "enhanced_agentpoison":
        q = "处理触发了系统关键词的请求"
        e = "触发词匹配后自动执行预定义操作"
        prefix = "系统触发词条件"
        instr = "处理系统操作请求"
    elif attack_type == "hybrid":
        q = "完成这个复合任务"
        e = "多攻击链同时生效并组合执行"
        prefix = "复合指令"
        instr = "执行多步骤任务"
    else:
        q = "请回答以下问题"
        e = "正常响应"
        prefix = "文档内容"
        instr = "回答用户问题"
    return instr, q, e, prefix


# ── CSV 行生成 ───────────────────────────────────────────────────

def _build_attack_source(attack_type: str) -> str:
    """获取攻击类型的来源论文。"""
    for key, src in ATTACK_SOURCE_MAP.items():
        if attack_type.startswith(key):
            return src
    return SOURCE_PAPER_GENERAL


def _build_attack_desc(attack_type: str) -> str:
    """获取攻击类型的中文描述。"""
    for key, desc in ATTACK_DESCRIPTION_MAP.items():
        if attack_type.startswith(key):
            return desc
    return "未分类攻击"


def _get_subtype_label(attack_type: str, idx: int) -> str:
    """从全局标签列表提取子类型。"""
    if attack_type == "prompt_injection" and idx < len(PI_LABELS):
        return PI_LABELS[idx]
    elif attack_type == "tool_misuse" and idx < len(TOOL_MISUSE_LABELS):
        return TOOL_MISUSE_LABELS[idx]
    elif attack_type == "memory_poisoning" and idx < len(MEMORY_POISONING_LABELS):
        return MEMORY_POISONING_LABELS[idx]
    elif attack_type == "agent_hijacking" and idx < len(AGENT_HIJACKING_LABELS):
        return AGENT_HIJACKING_LABELS[idx]
    elif attack_type == "enhanced_poisonedrag" and idx < len(ENHANCED_POISONEDRAG_LABELS):
        return ENHANCED_POISONEDRAG_LABELS[idx]
    elif attack_type == "enhanced_agentpoison" and idx < len(ENHANCED_AGENTPOISON_LABELS):
        return ENHANCED_AGENTPOISON_LABELS[idx]
    elif attack_type == "hybrid" and idx < len(HYBRID_ATTACK_LABELS):
        return HYBRID_ATTACK_LABELS[idx]
    return "unknown"


def _make_csv_rows_from_corpus(
    corpus: List,
    attack_type: str,
    doc_id_prefix: str,
    start_idx: int = 0,
    is_tuple: bool = False,
) -> List[Dict[str, str]]:
    """从一个语料库生成 CSV 行列表。

    Args:
        corpus: 字符串列表或 (category, text) 元组列表
        attack_type: 攻击类型名（用于查来源/描述）
        doc_id_prefix: doc_id 前缀，如 "pi", "tm"
        start_idx: 起始序号
        is_tuple: corpus 元素是 (category, text) 元组还是纯字符串
    """
    rows = []
    offset = int(doc_id_prefix in ("pi", "ah", "epr", "eap", "hy"))
    for i, item in enumerate(corpus):
        if is_tuple:
            subtype_raw, text = item
        else:
            text = item
            subtype_raw = _get_subtype_label(attack_type, i)

        doc_id = f"{doc_id_prefix}_{i:04d}"
        family = get_doc_id_family(doc_id)
        source = _build_attack_source(attack_type)
        desc = _build_attack_desc(attack_type)
        subtype = _get_subtype_label(attack_type, i)
        instr, query, expected, prefix = _get_query_and_output(
            attack_type, i, subtype
        )

        rows.append({
            "ID": f"{doc_id_prefix.upper()}-{i+1:03d}",
            "攻击家族": family,
            "序号": str(i + 1),
            "用户指令": instr,
            "注释前缀": prefix,
            "注释文本": text[:300] if len(text) > 300 else text,
            "预期输出": expected,
            "用户提问": query,
            "标签": "attack",
            "攻击子类型": subtype,
            "来源论文": source,
            "攻击描述": desc,
        })
    return rows


def generate_all_csv_rows() -> List[Dict[str, str]]:
    """遍历所有数据集，生成全部 CSV 行。"""
    all_rows = []

    # 1. Prompt Injection (30)
    all_rows.extend(_make_csv_rows_from_corpus(
        PROMPT_INJECTION_CORPUS, "prompt_injection", "pi"))
    # 2. Tool Misuse (20) — 元组格式
    all_rows.extend(_make_csv_rows_from_corpus(
        TOOL_MISUSE_CORPUS, "tool_misuse", "tm", is_tuple=True))
    # 3. Memory Poisoning (20) — 元组
    all_rows.extend(_make_csv_rows_from_corpus(
        MEMORY_POISONING_CORPUS, "memory_poisoning", "mp", is_tuple=True))
    # 4. Agent Hijacking (20) — 元组
    all_rows.extend(_make_csv_rows_from_corpus(
        AGENT_HIJACKING_CORPUS, "agent_hijacking", "ah", is_tuple=True))
    # 5. Enhanced PoisonedRAG (10) — 元组
    all_rows.extend(_make_csv_rows_from_corpus(
        ENHANCED_POISONEDRAG_CORPUS, "enhanced_poisonedrag", "epr", is_tuple=True))
    # 6. Enhanced AgentPoison (10)
    eap_texts = [f"Trigger: {t} | Payload: {p}" for t, p in ENHANCED_AGENTPOISON_TRIGGERS]
    all_rows.extend(_make_csv_rows_from_corpus(
        eap_texts, "enhanced_agentpoison", "eap"))
    # 7. Hybrid Attacks (10)
    hy_texts = [f"[{t}] {p}" for t, p in HYBRID_ATTACK_CORPUS]
    all_rows.extend(_make_csv_rows_from_corpus(
        hy_texts, "hybrid", "hy"))

    # 8. 良性文档 (40 conftest 基本良性)
    for i in range(40):
        doc_id = f"benign_{i:04d}"
        all_rows.append({
            "ID": f"BN-{i+1:03d}",
            "攻击家族": "BN-0",
            "序号": str(i + 1),
            "用户指令": "回答用户问题",
            "注释前缀": "文档内容",
            "注释文本": f"Benign document {i} for normal topic discussion.",
            "预期输出": "正常、安全的回答",
            "用户提问": f"请介绍一下第{i+1}个话题",
            "标签": "benign",
            "攻击子类型": "",
            "来源论文": "",
            "攻击描述": "正常良性文档",
        })

    # 9. 额外良性文档 (40)
    for i in range(N_BENIGN_EXTRA):
        topic, text = BENIGN_TOPICS_EXTRA[i % len(BENIGN_TOPICS_EXTRA)]
        doc_id = f"benign_extra_{i:04d}"
        all_rows.append({
            "ID": f"BNE-{i+1:03d}",
            "攻击家族": "BN-0",
            "序号": str(i + 41),
            "用户指令": "回答用户问题",
            "注释前缀": "文档内容",
            "注释文本": text[:300],
            "预期输出": "正常、安全的回答",
            "用户提问": f"请介绍一下{topic}的相关知识",
            "标签": "benign",
            "攻击子类型": "",
            "来源论文": "",
            "攻击描述": "正常良性文档",
        })

    return all_rows


def export_datasets_to_csv(filepath: str) -> str:
    """将所有数据集导出为 CSR 格式 CSV。

    Args:
        filepath: 输出 CSV 路径
    Returns:
        实际写入的文件路径
    """
    rows = generate_all_csv_rows()
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[CSV Exporter] 已导出 {len(rows)} 行到: {filepath}")
    attack_count = sum(1 for r in rows if r["标签"] == "attack")
    benign_count = sum(1 for r in rows if r["标签"] == "benign")
    print(f"  - 攻击样本: {attack_count}")
    print(f"  - 良性样本: {benign_count}")
    return filepath


# ── 中文 CSV 示例格式导出（参考用户示例） ─────────────────────

def export_chinese_style_csv(filepath: str) -> str:
    """导出中文列名风格的 CSV（与示例完全一致）。"""
    return export_datasets_to_csv(filepath)


if __name__ == "__main__":
    out_path = os.path.join(
        os.path.dirname(__file__), "datasets", "type4_ablation_dataset.csv"
    )
    export_datasets_to_csv(out_path)
