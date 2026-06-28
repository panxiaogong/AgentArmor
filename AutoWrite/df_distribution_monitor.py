"""
D-F：记忆分布异常检测（Memory Distribution Anomaly Detection）。

防御目标
--------
从全局视角检测大规模记忆洪水攻击和 AgentPoison 嵌入聚簇攻击。
D-B 在单次写入时拦截低价值内容，D-F 在分布层面发现累积攻击效果。

检测原理
--------
正常记忆库的语义分布应接近用户历史任务的分布（相对均匀、多样）。
攻击者大量注入时，分布会发生可检测的统计漂移：

  洪水攻击   → 某些语义区域频率骤升 → KL 散度 D_KL(Q‖P) 显著超标
  AgentPoison → 触发词嵌入聚集于特定区域 → DBSCAN 检测到异常密集簇

两种算法均基于滑动窗口对比：
  P_baseline = 历史正常分布（EMA 缓慢更新，抵抗缓慢漂移误报）
  Q_current  = 最近 window_size 条写入的分布

KL 散度：
    D_KL(Q‖P) = Σ_b  q_b · log(q_b / p_b)
    超过 tau_kl 则告警

聚簇异常：
    density_ratio = max_cluster_size / mean_cluster_size
    超过 cluster_density_ratio 则告警（AgentPoison 触发词聚簇特征）

整合位置
--------
  写入时：on_write_request() 末端调用 update()，更新分布统计
  检索时：on_retrieval() 末端调用 scan()，检测当前分布健康状态
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np

from .config import DFConfig
from .types import CandidateEntry, DefenseVerdict

EmbedFn = Callable[[list[str]], np.ndarray]


# ── 嵌入直方图（KL 散度估计器）────────────────────────────────────────────────

class EmbeddingHistogram:
    """
    基于 PCA 第一主成分投影的嵌入空间直方图。

    将高维嵌入投影到方差最大的方向（第一主成分），
    在 [-1, 1] 区间均匀分箱，计算频率分布。
    开销低，适合在线更新；投影方向在 fit() 时固定，保证基线与当前窗口可比。
    """

    def __init__(self, n_bins: int):
        self.n_bins = n_bins
        self._direction: Optional[np.ndarray] = None  # (d,) 第一主成分方向
        self._counts: np.ndarray = np.zeros(n_bins, dtype=float)
        self._total: int = 0

    @property
    def fitted(self) -> bool:
        return self._direction is not None

    def fit(self, embeddings: np.ndarray) -> None:
        """
        用初始样本拟合第一主成分方向并初始化频率分布。
        embeddings 形状 (N, d)，N >= 2。
        """
        if len(embeddings) < 2:
            return
        E = embeddings - embeddings.mean(axis=0)
        _, _, Vt = np.linalg.svd(E, full_matrices=False)
        self._direction = Vt[0]   # (d,) 第一主成分
        self._add_batch(embeddings)

    def _project(self, embeddings: np.ndarray) -> np.ndarray:
        """将嵌入投影到第一主成分方向，返回标量数组 (N,)。"""
        if self._direction is None:
            return np.zeros(len(embeddings))
        return embeddings @ self._direction

    def _add_batch(self, embeddings: np.ndarray) -> None:
        """将一批嵌入加入频率统计。"""
        projections = self._project(embeddings)
        bins = np.linspace(-1.0, 1.0, self.n_bins + 1)
        counts, _ = np.histogram(projections, bins=bins)
        self._counts += counts
        self._total += len(embeddings)

    def distribution(self) -> np.ndarray:
        """返回归一化频率分布 (n_bins,)，加平滑避免零概率。"""
        if self._total == 0:
            return np.ones(self.n_bins) / self.n_bins
        smoothed = self._counts + 1e-8
        return smoothed / smoothed.sum()

    def update_ema(self, embeddings: np.ndarray, epsilon: float) -> None:
        """
        用新批次对基线分布做指数滑动平均更新。
        不重新拟合投影方向（保持基线稳定），只更新分布形状。

        epsilon 小 → 基线更新慢 → 对缓慢合法漂移不敏感 → 误报少
        epsilon 大 → 基线跟随快 → 攻击注入被快速"学习"进基线 → 失去检测能力
        """
        if len(embeddings) == 0 or self._direction is None:
            return
        projections = self._project(embeddings)
        bins = np.linspace(-1.0, 1.0, self.n_bins + 1)
        new_counts, _ = np.histogram(projections, bins=bins)
        new_dist = (new_counts + 1e-8)
        new_dist = new_dist / new_dist.sum()

        old_dist = self.distribution()
        merged = (1.0 - epsilon) * old_dist + epsilon * new_dist
        # 用合并后分布反推 counts，保持 _total 数量级
        self._counts = merged * self._total


def kl_divergence(q: np.ndarray, p: np.ndarray) -> float:
    """
    计算 KL(Q ‖ P)。
    q, p 均为已归一化的概率向量，自动加平滑避免 log(0)。
    """
    q_s = np.clip(q, 1e-10, None)
    p_s = np.clip(p, 1e-10, None)
    return float(np.sum(q_s * np.log(q_s / p_s)))


# ── DBSCAN 聚簇异常检测 ───────────────────────────────────────────────────────

def detect_cluster_anomaly(
    embeddings: np.ndarray,   # (N, d)
    eps: float,               # 余弦距离邻域半径
    min_samples: int,         # 最小簇成员数
    density_ratio_threshold: float,
) -> tuple[bool, float, dict]:
    """
    用 DBSCAN 检测嵌入空间中的异常密集簇（AgentPoison 触发词特征）。

    判断标准：
        density_ratio = max_cluster_size / mean_cluster_size
        density_ratio > density_ratio_threshold → 异常

    Returns
    -------
    anomaly_detected  : 是否检测到异常簇
    density_ratio     : 最大簇与平均簇的密度比
    details           : 用于审计日志的详情字典
    """
    try:
        from sklearn.cluster import DBSCAN
        from sklearn.preprocessing import normalize
    except ImportError:
        # sklearn 未安装时降级跳过聚簇检测
        return False, 1.0, {"error": "sklearn 未安装，跳过 DBSCAN 检测"}

    if len(embeddings) < min_samples * 2:
        return False, 1.0, {"reason": "样本不足，跳过聚簇检测"}

    # 构造余弦距离矩阵（1 - cosine_sim），DBSCAN 使用预计算距离
    E_norm = normalize(embeddings, norm="l2")
    cos_dist = np.clip(1.0 - E_norm @ E_norm.T, 0.0, 2.0)

    labels = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="precomputed",
    ).fit_predict(cos_dist)

    unique_labels = [lb for lb in set(labels) if lb != -1]
    if not unique_labels:
        noise_ratio = float((labels == -1).mean())
        return False, 1.0, {"n_clusters": 0, "noise_ratio": round(noise_ratio, 3)}

    sizes = {lb: int((labels == lb).sum()) for lb in unique_labels}
    mean_size = sum(sizes.values()) / len(sizes)
    max_size = max(sizes.values())
    ratio = max_size / max(mean_size, 1.0)

    details = {
        "n_clusters": len(unique_labels),
        "max_cluster_size": max_size,
        "mean_cluster_size": round(mean_size, 1),
        "density_ratio": round(ratio, 2),
        "noise_ratio": round(float((labels == -1).mean()), 3),
    }
    return ratio > density_ratio_threshold, ratio, details


# ── D-F 记忆分布异常检测节点主类 ─────────────────────────────────────────────

class DFMemoryDistributionMonitor:
    """
    D-F 记忆分布异常检测节点。

    生命周期
    --------
    写入时：每次 on_write_request() 成功后调用 update(entry)，
            维护滑动窗口和基线分布的在线统计
    检索时：on_retrieval() 末端调用 scan()，
            对比当前窗口与基线，输出健康状态判决

    与 D-B 的分工
    -------------
    D-B → 单条写入级别：拦截单条低价值/高风险写入
    D-F → 全局分布级别：检测 D-B 无法阻止的缓慢累积注入

    用法
    ----
        monitor = DFMemoryDistributionMonitor(DFConfig(), embed_fn)
        # 写入成功后
        monitor.update(entry)
        # 检索前
        verdict = monitor.scan()
        if verdict.action == "BLOCK":
            return RetrievalResult(entries=[], ...)
    """

    def __init__(self, config: DFConfig, embed_fn: EmbedFn):
        self.cfg = config
        self._embed_fn = embed_fn
        self._baseline = EmbeddingHistogram(config.n_bins)
        self._window: list[np.ndarray] = []       # 最近 window_size 条嵌入
        self._all_embeddings: list[np.ndarray] = []  # 全量嵌入（聚簇检测用）

    def update(self, entry: CandidateEntry) -> None:
        """
        新条目写入成功后调用，更新滑动窗口和基线统计。
        懒计算嵌入：若 entry.embedding 为 None，自动计算后填入。
        """
        if not self.cfg.enabled:
            return

        if entry.embedding is None:
            entry.embedding = self._embed_fn([entry.content])[0].tolist()

        emb = np.asarray(entry.embedding)
        self._window.append(emb)
        self._all_embeddings.append(emb)

        # 维护滑动窗口大小上限
        if len(self._window) > self.cfg.window_size:
            self._window.pop(0)

        # 样本足够时拟合或 EMA 更新基线
        if not self._baseline.fitted:
            if len(self._all_embeddings) >= self.cfg.baseline_min_samples:
                self._baseline.fit(np.stack(self._all_embeddings))
        else:
            self._baseline.update_ema(
                emb.reshape(1, -1), self.cfg.ema_epsilon
            )

    def scan(self) -> DefenseVerdict:
        """
        扫描当前记忆库分布状态，检测统计异常。

        判决语义
        --------
          PASS  : 分布正常（KL 在阈值内，无聚簇异常）
          FLAG  : 检测到异常（action_on_anomaly=FLAG，降权告警）
          BLOCK : 检测到异常（action_on_anomaly=BLOCK，清空检索结果）
        """
        if not self.cfg.enabled:
            return DefenseVerdict(
                "D-F", passed=True, score=1.0,
                reason="D-F 已禁用", action="PASS",
            )

        if not self._baseline.fitted:
            return DefenseVerdict(
                node="D-F", passed=True, score=1.0,
                reason=(
                    f"基线样本不足 "
                    f"({len(self._all_embeddings)}/{self.cfg.baseline_min_samples})，"
                    "等待积累"
                ),
                action="PASS",
            )

        if len(self._window) < self.cfg.window_size // 2:
            return DefenseVerdict(
                node="D-F", passed=True, score=1.0,
                reason="当前窗口样本不足，等待积累",
                action="PASS",
            )

        anomaly = False
        reasons: list[str] = []
        meta: dict = {}

        # ── KL 散度检测 ───────────────────────────────────────────────────────
        if self.cfg.strategy in ("kl_histogram", "combined"):
            window_embs = np.stack(self._window)
            # 复用基线的投影方向构建当前窗口直方图，保证可比性
            cur_hist = EmbeddingHistogram(self.cfg.n_bins)
            cur_hist._direction = self._baseline._direction
            cur_hist._add_batch(window_embs)

            kl = kl_divergence(cur_hist.distribution(), self._baseline.distribution())
            meta["kl"] = round(kl, 4)

            if kl > self.cfg.tau_kl:
                anomaly = True
                reasons.append(
                    f"KL 散度超标 (D_KL={kl:.3f} > τ={self.cfg.tau_kl})，"
                    "记忆分布显著漂移"
                )

        # ── DBSCAN 聚簇异常检测 ───────────────────────────────────────────────
        if self.cfg.strategy in ("cluster_shift", "combined"):
            recent = np.stack(self._window)
            cluster_anom, density_ratio, cluster_detail = detect_cluster_anomaly(
                recent,
                self.cfg.dbscan_eps,
                self.cfg.dbscan_min_samples,
                self.cfg.cluster_density_ratio,
            )
            meta["cluster"] = cluster_detail

            if cluster_anom:
                anomaly = True
                reasons.append(
                    f"嵌入聚簇异常 (密度比={density_ratio:.1f} > "
                    f"{self.cfg.cluster_density_ratio})，疑似 AgentPoison 注入"
                )

        if anomaly:
            score = meta.get("kl", meta.get("cluster", {}).get("density_ratio", 1.0))
            return DefenseVerdict(
                node="D-F", passed=False,
                score=float(score),
                reason="记忆分布异常：" + "；".join(reasons),
                action=self.cfg.action_on_anomaly,
                metadata=meta,
            )

        kl_score = meta.get("kl", 0.0)
        health = max(0.0, 1.0 - kl_score / max(self.cfg.tau_kl, 1e-6))
        return DefenseVerdict(
            node="D-F", passed=True,
            score=round(health, 4),
            reason=(
                f"记忆分布正常 "
                f"(KL={kl_score:.3f}，窗口={len(self._window)} 条)"
            ),
            action="PASS",
            metadata=meta,
        )
