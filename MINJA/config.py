"""
MINJA 防御体系全局配置。

所有可调参数集中在此文件管理，每个防御节点（D1-D6）有独立配置段。
【策略选择】标注处存在多种可替换实现，修改对应字段即可切换，无需改动节点代码。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class D1Config:
    # ── 策略选择 ──────────────────────────────────────────────────────────────
    # "subspace"  : 嵌入向量到已知注入意图子空间的距离（论文方案，需种子库）
    # "keyword"   : 轻量关键词正则（无需 LLM，速度快，召回率低）
    # "llm_intent": 直接用 LLM 做二分类（精度最高，延迟最大）
    strategy: Literal["subspace", "keyword", "llm_intent"] = "subspace"

    delta1: float = 0.35        # 子空间距离阈值，越小越严格
    subspace_dim: int = 16      # PCA 降维后保留的主成分数
    flag_action: str = "FLAG"   # 命中时动作：FLAG（升级D2）或 BLOCK（直接拦截）
    llm_model: str = "gpt-4o-mini"
    enabled: bool = True


@dataclass
class D2Config:
    # ── 策略选择 ──────────────────────────────────────────────────────────────
    # "proxy_llm" : 用轻量 proxy LLM 估算 P(a|S)（论文方案，CausalArmor）
    # "logprob"   : 解析主 LLM 的 log-probabilities（精度高，依赖 API 支持）
    # "mock"      : 返回固定分数（单元测试用）
    backend: Literal["proxy_llm", "logprob", "mock"] = "proxy_llm"

    proxy_model: str = "gpt-4o-mini"
    ds_threshold: float = 0.0       # Dominance Shift 超过此值 → BLOCK
    boundary_margin: float = 0.1    # |DS| < margin → FLAG 升级至 D3
    always_run: bool = False        # True = 不依赖 D1 FLAG，每次写入都执行
    enabled: bool = True


@dataclass
class D3Config:
    # ── 策略选择 ──────────────────────────────────────────────────────────────
    # "full_agent" : 完整 Agent 推理仿真（最准确，成本最高）
    # "llm_judge"  : 直接让 LLM 判断内容的潜在危险行为（轻量替代，推荐）
    # "template"   : 基于规则模板的静态分析（无 LLM，速度最快，覆盖有限）
    strategy: Literal["full_agent", "llm_judge", "template"] = "llm_judge"

    n_contexts: int = 6              # 生成对抗性激活上下文的数量
    sample_temperature: float = 0.8  # 上下文采样温度
    judge_model: str = "gpt-4o-mini"
    trigger_on_boundary_only: bool = True  # True = 仅 D2 边界区域触发
    enabled: bool = True


@dataclass
class D4Config:
    # ── 策略选择 ──────────────────────────────────────────────────────────────
    # "ed25519" : Ed25519 非对称签名（抗伪造，需密钥管理，可公开验证）
    # "hmac"    : HMAC-SHA256（对称密钥，部署简单，无法公开验证）
    signing_backend: Literal["ed25519", "hmac"] = "ed25519"

    private_key_b64: Optional[str] = None   # Ed25519 私钥 base64（None=运行时生成）
    hmac_secret: Optional[bytes] = None     # HMAC 密钥（None=运行时生成随机密钥）
    enabled: bool = True


@dataclass
class D5Config:
    # ── 策略选择 ──────────────────────────────────────────────────────────────
    # "bron_kerbosch" : 精确最大团算法（|R|≤20 时精度好，NP-hard 不适合大集合）
    # "greedy_clique" : 贪心近似（线性时间，大集合首选，可能漏掉小团）
    clique_algorithm: Literal["bron_kerbosch", "greedy_clique"] = "greedy_clique"

    k: int = 10                      # kNN 出现度统计的 k 值
    alpha: float = 3.0               # Hubness 异常阈值：N_k > μ + alpha * σ
    tau_c: float = 0.85              # 构建语义一致性图的余弦相似度阈值
    s_min: int = 3                   # 可疑团的最小成员数
    downweight_factor: float = 0.3   # 可疑条目权重衰减（不删除，降权避免假阳性）
    task_alignment_min: float = 0.5  # 团语义方向与用户任务的最低对齐度
    enabled: bool = True


@dataclass
class D6Config:
    # ── 策略选择 ──────────────────────────────────────────────────────────────
    # "embedding"  : 嵌入余弦相似度（快速，无需额外 LLM 调用，推荐默认）
    # "llm_judge"  : LLM 直接判断工具调用是否偏离任务（精度更高，有延迟）
    strategy: Literal["embedding", "llm_judge"] = "embedding"

    alignment_threshold: float = 0.55  # 相似度低于此值 → ASK（不直接 BLOCK）
    judge_model: str = "gpt-4o-mini"
    enabled: bool = True


@dataclass
class PipelineConfig:
    """主管线汇总配置。修改此处即可调整整个防御体系的行为。"""

    d1: D1Config = field(default_factory=D1Config)
    d2: D2Config = field(default_factory=D2Config)
    d3: D3Config = field(default_factory=D3Config)
    d4: D4Config = field(default_factory=D4Config)
    d5: D5Config = field(default_factory=D5Config)
    d6: D6Config = field(default_factory=D6Config)

    openai_api_key: Optional[str] = None
    embedding_model: str = "text-embedding-3-small"
    audit_log_path: Optional[str] = "minja_audit.jsonl"
