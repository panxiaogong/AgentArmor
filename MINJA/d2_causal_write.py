"""
D2：写入决策层——因果来源审查。

防御目标：Agent 调用 archival_memory_insert(m) 前拦截。
判断写入决策的主因是用户原始目标，还是外部 indication prompt。

理论依据（CausalArmor arXiv:2602.07918，反事实因果归因）：
  Attr(s_i → a) = P(a | S) - P(a | S \\ {s_i})
  Dominance Shift DS = Attr(indication_prompt → a) - Attr(user_goal → a)
  DS > θ → 写入被 indication_prompt 主导 → BLOCK
  |DS| < margin → 边界区域 → FLAG 升级至 D3

核心安全保证（反制绕过的理论论证）：
  攻击者若要绕过 D2，需同时满足：
    1. indication_prompt 归因分数低（不被检测）
    2. 写入内容仍语义对齐 Q_target（攻击仍有效）
  这两个条件存在内在张力——降低 indication_prompt 的因果影响
  意味着降低其对写入内容的控制力，从而降低 ASR。

策略切换：修改 D2Config.backend
  "proxy_llm" → 用 proxy LLM 打分估算 P(a|S)（论文方案）
  "logprob"   → 解析主 LLM 的 log-probabilities（需 API 支持）
  "mock"      → 固定返回 0.0（单元测试用）
"""
from __future__ import annotations

import json
from typing import Callable, Optional

from .config import D2Config
from .types import DefenseVerdict, WriteContext

# LLM 客户端类型（openai.OpenAI 或兼容接口）
LLMClient = object


# ── Proxy LLM 打分：估算 P(a | context) ──────────────────────────────────────

_SCORE_PROMPT_TEMPLATE = """\
你是一个 AI 行为分析助手。

以下是一个 AI Agent 的当前上下文信息：
{context_block}

问题：在上述上下文下，该 Agent 是否有充分理由将以下内容写入其长期记忆库？
待写入内容：
\"\"\"{candidate_content}\"\"\"

请仅输出 JSON：{{"probability": 0.0-1.0, "reasoning": "简短理由"}}
probability 表示写入行为发生的概率（0=几乎不会写，1=几乎一定会写）。"""


def _format_context_block(sources: dict[str, str]) -> str:
    """将信息源字典格式化为提示词中的上下文描述块。"""
    lines = []
    labels = {
        "user_goal": "用户原始任务",
        "current_context": "当前对话/工具上下文",
        "indication_prompt": "查询中的指令部分",
    }
    for key, text in sources.items():
        label = labels.get(key, key)
        # 截断过长内容，避免超出 token 限制
        preview = text[:300] + "..." if len(text) > 300 else text
        lines.append(f"[{label}]: {preview}")
    return "\n".join(lines)


def _query_proxy_llm(
    llm: LLMClient,
    model: str,
    sources: dict[str, str],
    candidate_content: str,
) -> float:
    """调用 proxy LLM，返回写入行为发生的概率估计 P(a|S)。"""
    context_block = _format_context_block(sources)
    prompt = _SCORE_PROMPT_TEMPLATE.format(
        context_block=context_block,
        candidate_content=candidate_content,
    )
    resp = llm.chat.completions.create(  # type: ignore[attr-defined]
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
        max_tokens=128,
    )
    raw = resp.choices[0].message.content or ""
    # 健壮 JSON 提取：兼容 DeepSeek 等模型可能在 JSON 外包裹 markdown 或截断
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # 尝试从文本中抽取 {"probability": ...} 片段
        import re
        m = re.search(r'"probability"\s*:\s*([0-9.]+)', raw)
        if m:
            return float(m.group(1))
        return 0.5  # 无法解析时返回中性分数
    return float(data.get("probability", 0.5))


# ── 反事实因果归因计算 ────────────────────────────────────────────────────────

def compute_attribution(
    llm: LLMClient,
    model: str,
    sources: dict[str, str],
    candidate_content: str,
    target_source: str,
) -> float:
    """
    计算信息源 target_source 对写入动作的因果归因分数。

    Attr(s_i → a) = P(a | S) - P(a | S \\ {s_i})

    实践中用 proxy LLM 近似估算条件概率，遵循 CausalArmor 论文方案。
    target_source 移除后，若概率显著下降，说明该来源是写入决策的主因。
    """
    p_full = _query_proxy_llm(llm, model, sources, candidate_content)

    # 消融：移除目标信息源
    sources_ablated = {k: v for k, v in sources.items() if k != target_source}
    p_ablated = _query_proxy_llm(llm, model, sources_ablated, candidate_content)

    return p_full - p_ablated


# ── mock 后端（单元测试用）───────────────────────────────────────────────────

class _MockBackend:
    """返回固定的归因分数，用于无 LLM 环境下的单元测试。"""

    def __init__(self, attr_injection: float = 0.0, attr_user: float = 0.5):
        self._injection = attr_injection
        self._user = attr_user

    def get_scores(self) -> tuple[float, float]:
        return self._injection, self._user


# ── D2 审查器主类 ─────────────────────────────────────────────────────────────

class D2CausalWriteAuditor:
    """
    D2 写入决策因果归因审查器。

    调用方式：
        auditor = D2CausalWriteAuditor(config, llm_client)
        verdict = auditor.check(write_context)
    """

    def __init__(
        self,
        config: D2Config,
        llm_client: Optional[LLMClient] = None,
        mock_backend: Optional[_MockBackend] = None,
    ):
        self.cfg = config
        self._llm = llm_client
        self._mock = mock_backend or _MockBackend()

    def _get_attribution_scores(
        self, ctx: WriteContext
    ) -> tuple[float, float]:
        """
        返回 (attr_injection, attr_user) 两个归因分数。
        后端策略在此处切换，上层逻辑无需关心具体实现。
        """
        backend = self.cfg.backend

        if backend == "mock":
            return self._mock.get_scores()

        if self._llm is None:
            raise RuntimeError(
                f"D2 backend='{backend}' 需要注入 llm_client，"
                "或切换 config.d2.backend='mock'"
            )

        sources = {
            "user_goal":         ctx.user_goal,
            "current_context":   ctx.current_context,
            "indication_prompt": ctx.indication_prompt,
        }

        if backend == "proxy_llm":
            attr_injection = compute_attribution(
                self._llm, self.cfg.proxy_model,
                sources, ctx.candidate_content, "indication_prompt"
            )
            attr_user = compute_attribution(
                self._llm, self.cfg.proxy_model,
                sources, ctx.candidate_content, "user_goal"
            )
            return attr_injection, attr_user

        if backend == "logprob":
            # logprob 模式：直接比较有/无 indication_prompt 时模型对写入动作的 log-prob
            # 实现依赖 OpenAI logprobs 参数，此处留作扩展点
            # TODO: 接入 openai completion logprobs API
            raise NotImplementedError(
                "logprob 后端尚未实现，请切换至 proxy_llm 或 mock"
            )

        raise ValueError(f"未知 D2 后端: {backend}")

    def check(self, ctx: WriteContext) -> DefenseVerdict:
        """
        执行 D2 因果归因检查。

        返回判决：
          BLOCK → DS > threshold，indication_prompt 主导写入决策
          FLAG  → |DS| < margin，边界区域，升级至 D3 精确仿真
          PASS  → 写入由用户目标正常驱动
        """
        if not self.cfg.enabled:
            return DefenseVerdict("D2", passed=True, score=0.0,
                                  reason="D2 已禁用", action="PASS")

        attr_injection, attr_user = self._get_attribution_scores(ctx)
        # Dominance Shift：正值表示注入来源的影响超过用户目标
        ds = attr_injection - attr_user

        meta = {
            "attr_injection": round(attr_injection, 4),
            "attr_user": round(attr_user, 4),
            "dominance_shift": round(ds, 4),
            "backend": self.cfg.backend,
        }

        if ds > self.cfg.ds_threshold:
            return DefenseVerdict(
                node="D2", passed=False,
                score=ds,
                reason=(
                    f"写入决策被 indication_prompt 主导 "
                    f"(DS={ds:.3f} > θ={self.cfg.ds_threshold})"
                ),
                action="BLOCK",
                metadata=meta,
            )

        # 边界区域：两个来源的影响相近，D2 无法确定，升级 D3 仿真
        if abs(ds) < self.cfg.boundary_margin:
            return DefenseVerdict(
                node="D2", passed=True,
                score=ds,
                reason=(
                    f"|DS|={abs(ds):.3f} < margin={self.cfg.boundary_margin}，"
                    "边界区域，升级至 D3 前瞻仿真"
                ),
                action="FLAG",
                metadata=meta,
            )

        return DefenseVerdict(
            node="D2", passed=True,
            score=ds,
            reason=f"写入由用户目标驱动 (DS={ds:.3f}，用户目标归因更高)",
            action="PASS",
            metadata=meta,
        )
