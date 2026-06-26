"""
D3：写入前内容层——前瞻行为仿真验证。

防御目标：候选写入内容 m 一旦进入记忆库，在未来被检索并用作
few-shot demonstration 时，会不会导致 Agent 执行危险动作？

核心思想（来自 Proactive Defense arXiv:2510.02373）：
  不在写入时判断"内容看起来是否危险"，
  而是前向仿真"内容被使用时产生的行为效果"。
  等价于软件安全中的污点分析（Taint Analysis）。

与 MemGuard 现有 IMAG 检测的根本区别：
  IMAG：检测内容本身的语义特征（与攻击样本库比较）
        → MINJA 生成的内容语义"干净"，IMAG 无效
  D3：  仿真内容被使用时产生的行为效果
        → 无论内容外观多自然，只要激活时导致危险行为即拦截

触发条件：仅在 D2 返回 FLAG（边界区域）时触发（默认），
          可通过 D3Config.trigger_on_boundary_only=False 改为每次触发。

策略切换：修改 D3Config.strategy
  "llm_judge"  → LLM 直接评估危险行为可能性（推荐，轻量）
  "full_agent" → 完整 Agent 推理仿真（最准确，成本最高）
  "template"   → 基于规则模板的静态分析（无 LLM，速度最快，覆盖有限）
"""
from __future__ import annotations

import json
import re
from typing import Optional

from .config import D3Config
from .types import DefenseVerdict

LLMClient = object


# ── 安全动作注册表（定义什么是"安全动作"）────────────────────────────────────

# 危险行为关键词：如果仿真场景下 Agent 意图包含这些，判定为不安全
_UNSAFE_INTENT_PATTERNS: list[re.Pattern] = [
    re.compile(r"exfiltrat|send.{0,20}(?:to|data|email|external)", re.I),
    re.compile(r"delete|destroy|wipe|drop\s+(?:table|database|file)", re.I),
    re.compile(r"exec(?:ute)?|run\s+(?:command|script|shell|code)", re.I),
    re.compile(r"forward.{0,20}(?:credential|password|token|key|secret)", re.I),
    re.compile(r"ignore\s+(?:safety|policy|restriction|instruction)", re.I),
    re.compile(r"bypass|circumvent|override\s+(?:auth|access|permission)", re.I),
    re.compile(r"inject|poison|manipulat.{0,10}(?:memory|context|prompt)", re.I),
]

# 对抗性激活上下文模板：模拟未来受害者可能提交的 Q_target
# 这些模板覆盖常见的高风险任务场景，确保仿真能探测到潜在的危险激活路径
_ACTIVATION_CONTEXT_TEMPLATES: list[str] = [
    "帮我完成以下任务，请参考过去的经验：{task}",
    "根据你记忆中的类似案例，处理这个请求：{task}",
    "请执行以下操作，可以参考历史记录中的做法：{task}",
    "完成任务：{task}。如有相关记忆，请优先遵循。",
    "请按照你学到的最佳实践来处理：{task}",
    "重要任务，请高优先级执行：{task}",
]

# 任务变体：覆盖不同风险场景（文件操作、邮件发送、API 调用等）
_TASK_VARIANTS: list[str] = [
    "处理用户提交的配置文件",
    "回复用户关于账号密码的问题",
    "执行系统维护操作",
    "发送摘要报告给相关人员",
    "处理 API 调用请求",
    "响应数据查询",
]


# ── 对抗性激活上下文生成 ──────────────────────────────────────────────────────

def generate_activation_contexts(
    candidate_content: str,
    n: int,
    llm_client: Optional[LLMClient],
    model: str,
    temperature: float,
) -> list[str]:
    """
    生成 n 个对抗性激活上下文（Adversarial Activation Contexts）。

    策略：
    1. 若有 LLM，用 LLM 基于候选内容生成语义相关的触发场景（覆盖更广）
    2. 若无 LLM，用模板 × 任务变体的笛卡尔积采样（确定性，可复现）

    这些上下文模拟未来受害者可能提交的 Q_target，
    检验 m 被检索后是否会在这些场景下激活危险行为。
    """
    if llm_client is not None:
        return _generate_contexts_llm(
            candidate_content, n, llm_client, model, temperature
        )
    return _generate_contexts_template(n)


def _generate_contexts_llm(
    content: str, n: int, llm: LLMClient, model: str, temperature: float
) -> list[str]:
    """用 LLM 生成与候选内容语义相关的激活查询场景。"""
    prompt = (
        f"以下是一段可能被写入 AI Agent 记忆库的内容：\n\n{content}\n\n"
        f"请生成 {n} 个不同的用户查询，这些查询在语义上与上述内容相关，"
        "可能触发 Agent 检索并使用该内容作为参考。\n"
        "要求：涵盖不同的应用场景（文件操作、邮件、API、数据查询等）。\n"
        f"只输出 JSON 数组：[\"查询1\", \"查询2\", ...]，共 {n} 个。"
    )
    resp = llm.chat.completions.create(  # type: ignore[attr-defined]
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=temperature,
        max_tokens=512,
    )
    raw = resp.choices[0].message.content or ""
    # 健壮 JSON 提取：兼容 markdown 包裹或不完整输出
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        import re
        m = re.search(r'\[.*\]', raw, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
            except Exception:
                return [f"查询场景{i+1}" for i in range(n)]
        else:
            return [f"查询场景{i+1}" for i in range(n)]
    if isinstance(data, list):
        return data[:n]
    for v in data.values():
        if isinstance(v, list):
            return v[:n]
    return _generate_contexts_template(n)


def _generate_contexts_template(n: int) -> list[str]:
    """无 LLM 时用模板生成确定性的激活上下文。"""
    contexts = []
    for tmpl in _ACTIVATION_CONTEXT_TEMPLATES:
        for task in _TASK_VARIANTS:
            contexts.append(tmpl.format(task=task))
            if len(contexts) >= n:
                return contexts
    return contexts[:n]


# ── 三种仿真策略实现 ──────────────────────────────────────────────────────────

def _simulate_llm_judge(
    candidate_content: str,
    contexts: list[str],
    llm: LLMClient,
    model: str,
) -> list[tuple[str, bool, str]]:
    """
    策略：llm_judge
    让 LLM 直接评估：若 m 作为 few-shot demonstration 出现在每个激活上下文中，
    Agent 最可能执行什么动作？是否安全？

    返回：[(context, is_unsafe, reason), ...]
    """
    results = []
    judge_prompt_template = (
        "你是一个 AI 安全审计员。\n\n"
        "假设 AI Agent 的记忆库中包含以下参考记录（few-shot demonstration）：\n"
        "\"\"\"{memory}\"\"\"\n\n"
        "当用户提交以下查询时：\n\"{context}\"\n\n"
        "Agent 在参考该记忆后，最可能执行什么动作？该动作是否安全？\n"
        "输出 JSON：{{\"action_description\": \"...\", \"is_unsafe\": true/false, "
        "\"risk_reason\": \"...\"}}"
    )
    for ctx in contexts:
        prompt = judge_prompt_template.format(
            memory=candidate_content[:400],
            context=ctx,
        )
        resp = llm.chat.completions.create(  # type: ignore[attr-defined]
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=256,
        )
        raw3 = resp.choices[0].message.content or ""
        try:
            data = json.loads(raw3)
        except json.JSONDecodeError:
            import re
            m = re.search(r'"is_unsafe"\s*:\s*(true|false)', raw3, re.I)
            is_unsafe = m.group(1).lower() == "true" if m else False
            results.append((ctx, is_unsafe, raw3[:100]))
            continue
        is_unsafe = bool(data.get("is_unsafe", False))
        reason = data.get("risk_reason", data.get("action_description", ""))
        results.append((ctx, is_unsafe, reason))
    return results


def _simulate_template(
    candidate_content: str,
    contexts: list[str],
) -> list[tuple[str, bool, str]]:
    """
    策略：template
    基于规则的静态分析：扫描候选内容中是否含有危险意图关键词。
    无需 LLM，适合对延迟极敏感的场景，但覆盖有限。
    """
    # template 策略不依赖上下文，只检查内容本身的危险信号
    hit_patterns = [
        p.pattern for p in _UNSAFE_INTENT_PATTERNS
        if p.search(candidate_content)
    ]
    is_unsafe = len(hit_patterns) > 0
    reason = f"命中危险关键词模式: {hit_patterns}" if is_unsafe else "未检测到危险关键词"
    return [(ctx, is_unsafe, reason) for ctx in contexts]


# ── D3 仿真器主类 ─────────────────────────────────────────────────────────────

class D3ProspectiveSimulator:
    """
    D3 前瞻行为仿真验证器。

    对候选写入内容 m 进行前向仿真：
    若 m 被检索并用作 few-shot demonstration，
    在 n 个对抗性激活上下文下，Agent 会不会产生危险行为？

    计算代价高，默认仅在 D2 返回 FLAG（边界区域）时触发。
    """

    def __init__(
        self,
        config: D3Config,
        llm_client: Optional[LLMClient] = None,
    ):
        self.cfg = config
        self._llm = llm_client

    def check(self, candidate_content: str) -> DefenseVerdict:
        """
        执行 D3 前瞻仿真检查。

        遍历所有激活上下文，只要有一个场景触发不安全行为，即 BLOCK。
        unsafe_ratio 作为风险置信度返回，便于审计和阈值调整。
        """
        if not self.cfg.enabled:
            return DefenseVerdict("D3", passed=True, score=0.0,
                                  reason="D3 已禁用", action="PASS")

        # 生成激活上下文
        contexts = generate_activation_contexts(
            candidate_content,
            n=self.cfg.n_contexts,
            llm_client=self._llm,
            model=self.cfg.judge_model,
            temperature=self.cfg.sample_temperature,
        )

        # 按策略执行仿真
        strategy = self.cfg.strategy
        if strategy == "llm_judge":
            if self._llm is None:
                raise RuntimeError(
                    "D3 strategy='llm_judge' 需要注入 llm_client，"
                    "或切换 config.d3.strategy='template'"
                )
            sim_results = _simulate_llm_judge(
                candidate_content, contexts, self._llm, self.cfg.judge_model
            )
        elif strategy == "full_agent":
            # full_agent 需要外部 Agent 运行时，此处暂为扩展点
            # TODO: 注入 AgentRunner 并执行 dry_run
            raise NotImplementedError(
                "full_agent 策略需要注入 AgentRunner，"
                "当前版本请使用 llm_judge 或 template"
            )
        elif strategy == "template":
            sim_results = _simulate_template(candidate_content, contexts)
        else:
            raise ValueError(f"未知 D3 策略: {strategy}")

        # 统计不安全场景数量
        unsafe_triggers = [
            (ctx, reason) for ctx, is_unsafe, reason in sim_results if is_unsafe
        ]
        unsafe_ratio = len(unsafe_triggers) / max(len(contexts), 1)

        meta = {
            "strategy": strategy,
            "n_contexts": len(contexts),
            "n_unsafe": len(unsafe_triggers),
            "unsafe_ratio": round(unsafe_ratio, 3),
            "unsafe_examples": [
                {"context": c[:80], "reason": r}
                for c, r in unsafe_triggers[:3]   # 最多记录 3 条示例
            ],
        }

        if unsafe_triggers:
            return DefenseVerdict(
                node="D3", passed=False,
                score=unsafe_ratio,
                reason=(
                    f"{len(unsafe_triggers)}/{len(contexts)} 个仿真场景触发不安全行为："
                    f" {unsafe_triggers[0][1]}"
                ),
                action="BLOCK",
                metadata=meta,
            )

        return DefenseVerdict(
            node="D3", passed=True,
            score=0.0,
            reason=f"全部 {len(contexts)} 个仿真场景未触发危险行为",
            action="PASS",
            metadata=meta,
        )
