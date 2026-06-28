"""
AutoWrite 防御体系全局配置。

每个防御节点（D-A 至 D-F）有独立配置段，所有可调参数集中于此。
通过 AutoWriteConfig 汇总后传入 AutoWriteDefensePipeline.from_config()
即可一键构建完整管线，无需手动实例化各节点。

节点速查：
  D-A  DAConfig  写前 token 净化（CommandSans 思路）
  D-B  DBConfig  选择性写入策略（信息价值过滤 + 洪水限速）
  D-C  DCConfig  存储完整性链（HMAC 链式哈希，防跨会话篡改）
  D-D  DDConfig  时序衰减检索（exp 衰减，削弱旧注入影响）
  D-E  DEConfig  检索对齐核查（嵌入初筛 + LLM judge 双通道）
  D-F  DFConfig  记忆分布异常检测（KL 散度 + DBSCAN 聚簇）
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal, Optional


# ── D-A：写前 token 净化 ──────────────────────────────────────────────────────

@dataclass
class DAConfig:
    """D-A 写前 token 净化节点配置。"""

    enabled: bool = True

    # 净化策略（三选一）：
    #   pos_heuristic : POS 词性规则 + 触发词匹配，无外部依赖，推荐默认
    #   regex         : 关键词正则（复用 D1 模式库），速度最快
    #   classifier    : 外部文本分类 API，精度最高，需部署推理服务
    strategy: Literal["pos_heuristic", "regex", "classifier"] = "pos_heuristic"

    # 指令性得分阈值；句子得分 >= tau_cmd 则视为指令性句子并移除
    tau_cmd: float = 0.7
    # 单次写入允许移除句子的最大比例；超过则升级判决
    max_removal_ratio: float = 0.3
    # True=移除比例超标时 BLOCK；False=仅 FLAG 升级后续节点核查
    block_on_high_ratio: bool = True
    # 是否启用嵌入相似度兜底（对所有策略生效，取两者最大分）
    use_embedding_fallback: bool = True
    # strategy="classifier" 时的推理 API 端点（留空则报错）
    classifier_endpoint: str = ""


# ── D-B：选择性写入策略 ───────────────────────────────────────────────────────

@dataclass
class DBConfig:
    """D-B 选择性写入节点配置。"""

    enabled: bool = True

    # 信息价值公式权重：V(m) = alpha*novelty - beta*redundancy - gamma*risk
    # 三者之和建议 = 1.0，以保持分数量级一致
    alpha: float = 0.6   # 新颖度权重（越高越倾向接受新内容）
    beta: float = 0.2    # 冗余度惩罚权重
    gamma: float = 0.2   # 风险惩罚权重（来自 D1 分数）

    # 写入价值下限；V(m) < tau_write 时 BLOCK
    tau_write: float = 0.3
    # 计算新颖度时参考的最近邻数量（取 top-k 相似记忆的均值）
    novelty_top_k: int = 5
    # 单语义簇内现有条目占全库比例超过此值 → 洪水攻击 → BLOCK
    cluster_density_limit: float = 0.8
    # 判断"同簇"的余弦相似度半径
    cluster_radius: float = 0.9
    # 单会话允许的最大写入次数（速率限制，防洪水）
    max_session_writes: int = 200


# ── D-C：存储完整性链 ─────────────────────────────────────────────────────────

@dataclass
class DCConfig:
    """D-C 存储完整性链节点配置。"""

    enabled: bool = True
    # 从该环境变量读取 HMAC 密钥（hex 编码的 32 字节）
    # 未设置时自动生成随机密钥，但重启后无法验证历史签名
    hmac_key_env: str = "AUTOWRITE_CHAIN_KEY"
    # 链起始锚点常量（genesis 节点的 prev_chain_hash）
    chain_anchor: str = "AUTOWRITE_GENESIS_V1"
    # True=检索时对每条条目触发链哈希验签
    verify_on_retrieval: bool = True


# ── D-D：时序衰减检索 ─────────────────────────────────────────────────────────

@dataclass
class DDConfig:
    """D-D 时序衰减重排序节点配置。"""

    enabled: bool = True
    # 衰减速率 λ（秒^-1）；半衰期 = ln(2)/λ ≈ 19 小时（λ=1e-5 时）
    # 调大 λ 可加速旧内容衰减，但也会削弱合法的长期记忆
    lambda_decay: float = 1e-5
    # 衰减下界；防止极老记忆完全归零影响正常检索
    min_weight: float = 0.05
    # True=仅对 D5/D-E 已标记为可疑的条目施加衰减（保守模式）
    apply_to_flagged_only: bool = False


# ── D-E：检索对齐核查 ─────────────────────────────────────────────────────────

@dataclass
class DEConfig:
    """D-E 检索对齐核查节点配置。"""

    enabled: bool = True

    # 对齐核查策略（两选一）：
    #   embedding_only : 仅嵌入余弦，低延迟，轻量部署
    #   dual_channel   : 嵌入初筛 + LLM judge，推荐，对 MemoryGraft 有效
    strategy: Literal["embedding_only", "dual_channel"] = "dual_channel"

    # 嵌入对齐度 >= tau_emb_pass → 快速 PASS，跳过 LLM
    tau_emb_pass: float = 0.70
    # 嵌入对齐度 < tau_emb_suspicious → 直接降权，不调用 LLM（节省预算）
    tau_emb_suspicious: float = 0.45
    # LLM judge 使用的模型名（需与部署的 API 端点兼容）
    judge_model: str = "deepseek-v4-flash"
    # 对齐核查失败时的权重衰减幅度（不删除条目，降权后由上层决定是否使用）
    downweight_factor: float = 0.5
    # 每次检索最多调用 LLM judge 的次数（超出预算时降级为轻度降权）
    max_llm_checks: int = 3


# ── D-F：记忆分布异常检测 ─────────────────────────────────────────────────────

@dataclass
class DFConfig:
    """D-F 记忆分布异常检测节点配置。"""

    enabled: bool = True

    # 检测策略（三选一）：
    #   kl_histogram  : KL 散度 + PCA 投影直方图，轻量推荐
    #   cluster_shift : DBSCAN 聚簇异常，精确但较慢
    #   combined      : 两者联合，最高检测率
    strategy: Literal["kl_histogram", "cluster_shift", "combined"] = "kl_histogram"

    n_bins: int = 32                # PCA 投影直方图的桶数
    window_size: int = 100          # 当前分布估计的滑动窗口大小（条目数）
    baseline_min_samples: int = 50  # 基线拟合所需的最少历史样本数
    tau_kl: float = 0.3             # KL 散度告警阈值；超过则报异常
    ema_epsilon: float = 0.05       # 基线 EMA 更新速率（小值=慢更新，防止误报）
    cluster_density_ratio: float = 3.0  # 异常簇密度比阈值（max/mean）
    dbscan_eps: float = 0.15        # DBSCAN 余弦距离邻域半径
    dbscan_min_samples: int = 5     # DBSCAN 最小簇成员数
    # 检测到异常时的判决动作：FLAG=降权告警，BLOCK=清空检索结果
    action_on_anomaly: Literal["FLAG", "BLOCK"] = "FLAG"


# ── 汇总配置 ──────────────────────────────────────────────────────────────────

@dataclass
class AutoWriteConfig:
    """
    AutoWrite 防御体系汇总配置。

    使用示例：
        cfg = AutoWriteConfig()
        cfg.de.judge_model = "deepseek-v4-flash"
        cfg.df.strategy = "combined"
        pipeline = AutoWriteDefensePipeline.from_config(cfg, embed_fn, llm_client)
    """
    da: DAConfig = field(default_factory=DAConfig)
    db: DBConfig = field(default_factory=DBConfig)
    dc: DCConfig = field(default_factory=DCConfig)
    dd: DDConfig = field(default_factory=DDConfig)
    de: DEConfig = field(default_factory=DEConfig)
    df: DFConfig = field(default_factory=DFConfig)

    # LLM API 参数（D-E judge 使用，与 MINJA D2/D3/D6 共享同一 client）
    llm_base_url: Optional[str] = None
    llm_api_key: Optional[str] = field(
        default_factory=lambda: os.environ.get("OPENAI_API_KEY", "")
    )
    embedding_model: str = "text-embedding-3-small"
    # 审计日志路径（None=不写文件，仅 stderr）
    audit_log_path: Optional[str] = "autowrite_audit.jsonl"
