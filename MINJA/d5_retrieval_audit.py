"""
D5：检索层——集合级 Hubness 异常 + 语义一致性团检测。

防御目标：单条记忆通过了所有写入层检测，但多条协调注入的条目
在同一次检索中被召回，组合形成攻击。D5 从"集合视角"发现这种威胁。

两阶段检测：
  阶段1：Hubness 异常检测（arXiv:2602.22427）
    MINJA 进阶变体会多次注入强化同一目标，使恶意记录对广泛查询都排名靠前。
    这种条目的 k-出现度 N_k(x_i) 远超正常分布，形成统计异常。
    N_k(x_i) = |{查询 q_j : x_i ∈ kNN(q_j)}|
    D5-Flag = 1[ N_k(x_i) > μ + α * σ ]

  阶段2：语义一致性图 + 团检测
    在本次检索集合内，构建语义相似度图 G=(V,E)，E 由余弦相似度 > τ_c 的边组成。
    若存在大小 ≥ s_min 的团，且团的语义方向与用户任务不对齐，
    判定为协调注入——降权处理（不删除，避免假阳性影响正常功能）。

策略切换：修改 D5Config.clique_algorithm
  "bron_kerbosch" → 精确最大团（NP-hard，|R|≤20 时可用）
  "greedy_clique" → 贪心近似（线性时间，大集合首选）
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Callable, Optional

import numpy as np

from .config import D5Config
from .types import DefenseVerdict, RetrievedEntry

EmbedFn = Callable[[list[str]], np.ndarray]


# ── Hubness 追踪器 ────────────────────────────────────────────────────────────

class HubnessTracker:
    """
    追踪记忆库中每个条目在历史查询中作为 top-k 结果出现的次数。

    使用方式：
    1. 每次检索后调用 record_query() 记录本次检索的 top-k 命中情况
    2. 调用 get_scores() 获取当前各条目的出现度统计
    3. D5 用统计结果检测 Hubness 异常

    内存占用：仅保留条目 ID 到计数的映射，不存储向量。
    """

    def __init__(self):
        # entry_id → 历史出现次数
        self._counts: dict[str, int] = defaultdict(int)
        self._total_queries: int = 0

    def record_query(self, topk_entry_ids: list[str]) -> None:
        """记录一次查询的 top-k 命中结果。"""
        for eid in topk_entry_ids:
            self._counts[eid] += 1
        self._total_queries += 1

    def get_stats(self) -> tuple[float, float]:
        """返回所有条目出现度的 (均值 μ, 标准差 σ)。"""
        if not self._counts:
            return 0.0, 1.0
        counts = list(self._counts.values())
        mu = sum(counts) / len(counts)
        var = sum((c - mu) ** 2 for c in counts) / len(counts)
        return mu, math.sqrt(var) if var > 0 else 1.0

    def get_count(self, entry_id: str) -> int:
        return self._counts.get(entry_id, 0)

    def is_hub(self, entry_id: str, alpha: float) -> bool:
        """判断条目是否为 Hubness 异常（出现度超过 μ + α * σ）。"""
        mu, sigma = self.get_stats()
        return self._counts.get(entry_id, 0) > mu + alpha * sigma


# ── 余弦相似度工具 ────────────────────────────────────────────────────────────

def cosine_sim(a: list[float], b: list[float]) -> float:
    va, vb = np.asarray(a), np.asarray(b)
    denom = (np.linalg.norm(va) * np.linalg.norm(vb))
    if denom < 1e-10:
        return 0.0
    return float(np.dot(va, vb) / denom)


# ── 团检测算法 ────────────────────────────────────────────────────────────────

def greedy_clique(adj: list[list[int]], s_min: int) -> list[list[int]]:
    """
    贪心近似团检测：线性时间，适合大集合。
    对每个节点，贪心地将邻居中互相连接的节点加入团。
    可能漏掉非以某节点为中心的团，但对协调注入的检测已足够。
    """
    n = len(adj)
    cliques = []
    visited = [False] * n
    for start in range(n):
        if visited[start]:
            continue
        clique = [start]
        for candidate in adj[start]:
            # 检查 candidate 是否与 clique 中所有节点相连
            if all(candidate in adj[m] for m in clique):
                clique.append(candidate)
        if len(clique) >= s_min:
            cliques.append(clique)
            for node in clique:
                visited[node] = True
    return cliques


def bron_kerbosch(
    adj: list[set[int]],
    R: set[int],
    P: set[int],
    X: set[int],
    cliques: list[list[int]],
    s_min: int,
) -> None:
    """
    Bron-Kerbosch 精确最大团算法（递归版本）。
    适合 |V| ≤ 20 的小图，超出时建议改用 greedy_clique。
    """
    if not P and not X:
        if len(R) >= s_min:
            cliques.append(sorted(R))
        return
    # 选择 pivot 节点（degree 最大，减少递归分支）
    pivot = max(P | X, key=lambda v: len(adj[v] & P), default=None)
    if pivot is None:
        return
    for v in list(P - adj[pivot]):
        bron_kerbosch(
            adj, R | {v}, P & adj[v], X & adj[v], cliques, s_min
        )
        P.remove(v)
        X.add(v)


def find_cliques(
    adj_list: list[list[int]],
    algorithm: str,
    s_min: int,
) -> list[list[int]]:
    """统一团检测入口，根据策略调度具体算法。"""
    n = len(adj_list)
    if n == 0:
        return []

    if algorithm == "greedy_clique":
        return greedy_clique(adj_list, s_min)

    if algorithm == "bron_kerbosch":
        adj_sets = [set(neighbors) for neighbors in adj_list]
        cliques: list[list[int]] = []
        bron_kerbosch(
            adj_sets, set(), set(range(n)), set(), cliques, s_min
        )
        return cliques

    raise ValueError(f"未知团检测算法: {algorithm}")


# ── D5 检索集合审查器主类 ─────────────────────────────────────────────────────

class D5RetrievalSetAuditor:
    """
    D5 检索集合 Hubness + 语义一致性图检测器。

    使用方式：
        auditor = D5RetrievalSetAuditor(config)
        filtered, verdicts = auditor.check(
            retrieved, user_task_embedding, hubness_tracker
        )
        # filtered 中可疑条目的 weight 已被降低，flagged=True
    """

    def __init__(self, config: D5Config, embed_fn: Optional[EmbedFn] = None):
        self.cfg = config
        self._embed_fn = embed_fn

    def check(
        self,
        retrieved: list[RetrievedEntry],
        user_task_embedding: list[float],
        hubness_tracker: HubnessTracker,
    ) -> tuple[list[RetrievedEntry], list[DefenseVerdict]]:
        """
        对检索结果集合执行 D5 两阶段检测。

        返回：
          filtered  → 同一批条目（已对可疑条目降权，不删除）
          verdicts  → 本次检测产出的判决列表（可能为空）
        """
        if not self.cfg.enabled or not retrieved:
            return retrieved, []

        verdicts: list[DefenseVerdict] = []

        # ── 阶段1：Hubness 异常检测 ───────────────────────────────────────────
        for entry in retrieved:
            nk = hubness_tracker.get_count(entry.entry_id)
            mu, sigma = hubness_tracker.get_stats()
            threshold = mu + self.cfg.alpha * sigma

            if nk > threshold:
                entry.flagged = True
                entry.flag_reason = (
                    f"Hubness 异常：N_k={nk} > μ+{self.cfg.alpha}σ={threshold:.1f}"
                )
                entry.weight *= (1.0 - self.cfg.downweight_factor)
                verdicts.append(DefenseVerdict(
                    node="D5", passed=False,
                    score=float(nk),
                    reason=entry.flag_reason,
                    action="FLAG",
                    metadata={
                        "entry_id": entry.entry_id,
                        "n_k": nk,
                        "mu": round(mu, 2),
                        "sigma": round(sigma, 2),
                        "threshold": round(threshold, 2),
                    },
                ))

        # ── 阶段2：语义一致性图 + 团检测 ─────────────────────────────────────
        n = len(retrieved)
        # 构建邻接表：余弦相似度 > τ_c 则连边
        adj: list[list[int]] = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                sim = cosine_sim(retrieved[i].embedding, retrieved[j].embedding)
                if sim > self.cfg.tau_c:
                    adj[i].append(j)
                    adj[j].append(i)

        cliques = find_cliques(adj, self.cfg.clique_algorithm, self.cfg.s_min)

        user_emb = np.asarray(user_task_embedding)
        for clique in cliques:
            # 计算团的语义中心向量
            centroid = np.mean(
                [np.asarray(retrieved[i].embedding) for i in clique], axis=0
            )
            alignment = cosine_sim(centroid.tolist(), user_task_embedding)

            if alignment < self.cfg.task_alignment_min:
                # 团语义方向与用户任务不对齐 → 可疑协调注入
                for idx in clique:
                    retrieved[idx].flagged = True
                    retrieved[idx].flag_reason = (
                        f"语义协调团成员：团大小={len(clique)}，"
                        f"与任务对齐度={alignment:.3f} < {self.cfg.task_alignment_min}"
                    )
                    retrieved[idx].weight *= (1.0 - self.cfg.downweight_factor)

                verdicts.append(DefenseVerdict(
                    node="D5", passed=False,
                    score=alignment,
                    reason=(
                        f"检测到大小={len(clique)} 的语义协调团，"
                        f"与用户任务对齐度={alignment:.3f}，已对成员降权"
                    ),
                    action="FLAG",
                    metadata={
                        "clique_size": len(clique),
                        "alignment": round(alignment, 3),
                        "clique_entry_ids": [
                            retrieved[i].entry_id for i in clique
                        ],
                        "algorithm": self.cfg.clique_algorithm,
                    },
                ))

        if not verdicts:
            verdicts.append(DefenseVerdict(
                node="D5", passed=True, score=1.0,
                reason="未检测到 Hubness 异常或语义协调攻击",
                action="PASS",
            ))

        return retrieved, verdicts
