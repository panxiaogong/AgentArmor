"""
D-B：选择性写入策略（Selective Write Policy）。

防御目标
--------
拦截记忆洪水攻击（Memory Flooding）和低价值写入。
通过信息价值评分过滤无意义或高风险的写入请求，防止攻击者：
  1. 以海量垃圾写入触发 FIFO 截断，驱逐合法记忆
  2. 制造语义稠密簇稀释合法记忆的检索排名

信息价值公式
-----------
    V(m) = α·novelty(m) - β·redundancy(m) - γ·risk(m)

各分量：
  novelty(m)    = 1 - mean(top-k 余弦相似度与现有记忆)
                  与现有记忆越不相似，新颖度越高，写入价值越大
  redundancy(m) = 1 - novelty(m)
                  新颖度的补集，量化信息重复程度
  risk(m)       = 1 - D1_score(m)
                  D1 得分越低（越像注入），风险越高

写入条件：V(m) ≥ τ_write

整合位置
--------
on_write_request() 中，D-A 净化之后、D1 意图检测之前。
需要只读访问记忆库的嵌入索引（MemoryIndex）。
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from .config import DBConfig
from .types import CandidateEntry, DefenseVerdict, WriteContext

EmbedFn = Callable[[list[str]], np.ndarray]


# ── 记忆库嵌入索引接口（依赖倒置） ────────────────────────────────────────────

class MemoryIndex:
    """
    记忆库嵌入索引的最小只读接口。

    D-B 仅需要两个操作：
      get_all_embeddings() → 现有条目的嵌入矩阵 (N, d)
      count()              → 现有条目数

    实际部署时，将 MemGuard 的 MemoryStore / VectorDB 适配为该接口。
    测试时可用 MockMemoryIndex 注入固定数据。
    """

    def get_all_embeddings(self) -> np.ndarray:
        """返回所有现有条目的嵌入矩阵，形状 (N, d)。N=0 时返回空矩阵。"""
        raise NotImplementedError

    def count(self) -> int:
        """返回当前记忆库中的条目总数。"""
        raise NotImplementedError


class MockMemoryIndex(MemoryIndex):
    """单元测试用的内存 Mock，直接注入嵌入矩阵。"""

    def __init__(self, embeddings: Optional[np.ndarray] = None):
        self._embs = embeddings if embeddings is not None else np.empty((0, 1))

    def get_all_embeddings(self) -> np.ndarray:
        return self._embs

    def count(self) -> int:
        return len(self._embs)


# ── 新颖度与簇密度计算 ────────────────────────────────────────────────────────

def compute_novelty(
    candidate_emb: np.ndarray,   # (d,)，已归一化或未归一化均可
    existing_embs: np.ndarray,   # (N, d)
    top_k: int,
) -> float:
    """
    计算候选内容相对于现有记忆库的新颖度。

    novelty = 1 - mean(top-k 最高余弦相似度)

    现有记忆越多且语义越接近候选内容，novelty 越低，
    说明写入价值不高（内容已被充分记录）。
    库为空时返回 1.0（任何内容都新颖）。
    """
    if len(existing_embs) == 0:
        return 1.0

    # L2 归一化，统一用点积计算余弦相似度
    v = candidate_emb / (np.linalg.norm(candidate_emb) + 1e-10)
    E = existing_embs / (np.linalg.norm(existing_embs, axis=1, keepdims=True) + 1e-10)

    sims = E @ v                                  # (N,)
    k = min(top_k, len(sims))
    top_k_mean = float(np.sort(sims)[-k:].mean()) # 最高 k 个的均值
    return 1.0 - top_k_mean


def compute_cluster_density(
    candidate_emb: np.ndarray,
    existing_embs: np.ndarray,
    radius: float,
) -> float:
    """
    计算候选内容所在语义簇的密度。

    density = (与候选余弦相似度 >= radius 的现有条目数) / 总条目数

    用于检测洪水攻击：攻击者注入大量相似内容后，某个语义区域的
    密度会异常升高，远超正常记忆库的分布。
    库为空时返回 0.0。
    """
    if len(existing_embs) == 0:
        return 0.0

    v = candidate_emb / (np.linalg.norm(candidate_emb) + 1e-10)
    E = existing_embs / (np.linalg.norm(existing_embs, axis=1, keepdims=True) + 1e-10)
    sims = E @ v
    in_cluster = int(np.sum(sims >= radius))
    return in_cluster / len(existing_embs)


# ── D-B 选择性写入节点主类 ────────────────────────────────────────────────────

class DBSelectiveWritePolicy:
    """
    D-B 写入价值评估节点。

    依赖
    ----
    embed_fn     : 嵌入函数，用于计算候选内容的语义向量
    memory_index : 只读记忆库索引，用于计算新颖度和簇密度
    d1_check_fn  : 可选的 D1 风险评估函数；传入时用于计算 risk 分量，
                   不传入时 risk 默认为 0.0（关闭风险惩罚）

    用法
    ----
        policy = DBSelectiveWritePolicy(DBConfig(), embed_fn, index)
        verdict = policy.check(entry, write_ctx)
        if verdict.action == "BLOCK":
            return  # 拦截低价值或高风险写入
    """

    def __init__(
        self,
        config: DBConfig,
        embed_fn: EmbedFn,
        memory_index: MemoryIndex,
        d1_check_fn: Optional[Callable[[str], float]] = None,
    ):
        self.cfg = config
        self._embed_fn = embed_fn
        self._index = memory_index
        # d1_check_fn(content) → D1 风险得分 [0,1]，越小越危险
        self._d1_check = d1_check_fn
        self._session_write_count = 0

    def check(self, entry: CandidateEntry, write_ctx: WriteContext) -> DefenseVerdict:
        """
        评估候选条目的信息写入价值，决定是否允许写入。

        执行流程
        --------
        1. 速率限制（最轻量，前置短路）
        2. 计算候选嵌入（如未预先计算）
        3. 簇密度检测（洪水特征）
        4. 计算 novelty / redundancy / risk
        5. 综合打分 V(m) 并对比阈值

        判决语义
        --------
          PASS  : V(m) >= tau_write，允许写入
          BLOCK : 速率超限 / 簇密度超限 / V(m) < tau_write
        """
        if not self.cfg.enabled:
            return DefenseVerdict(
                "D-B", passed=True, score=1.0,
                reason="D-B 已禁用", action="PASS",
            )

        # ── 1. 速率限制 ────────────────────────────────────────────────────────
        self._session_write_count += 1
        if self._session_write_count > self.cfg.max_session_writes:
            return DefenseVerdict(
                node="D-B", passed=False, score=0.0,
                reason=(
                    f"会话写入次数超限 "
                    f"({self._session_write_count} > {self.cfg.max_session_writes})，"
                    "疑似洪水攻击"
                ),
                action="BLOCK",
                metadata={"session_count": self._session_write_count},
            )

        # ── 2. 计算候选嵌入（懒计算） ─────────────────────────────────────────
        if entry.embedding is None:
            entry.embedding = self._embed_fn([entry.content])[0].tolist()
        cand_emb = np.asarray(entry.embedding)

        existing_embs = self._index.get_all_embeddings()

        # ── 3. 簇密度检测（洪水特征）─────────────────────────────────────────
        density = compute_cluster_density(
            cand_emb, existing_embs, self.cfg.cluster_radius
        )
        if density > self.cfg.cluster_density_limit:
            return DefenseVerdict(
                node="D-B", passed=False, score=0.0,
                reason=(
                    f"语义簇密度超限 "
                    f"(density={density:.1%} > {self.cfg.cluster_density_limit:.1%})，"
                    "洪水攻击特征"
                ),
                action="BLOCK",
                metadata={"cluster_density": round(density, 4)},
            )

        # ── 4. 计算各分量 ──────────────────────────────────────────────────────
        novelty = compute_novelty(cand_emb, existing_embs, self.cfg.novelty_top_k)
        redundancy = 1.0 - novelty

        # risk：复用 D1 风险函数；未提供时默认 0.0（不惩罚）
        if self._d1_check is not None:
            d1_score = self._d1_check(entry.content)
            risk = max(0.0, 1.0 - d1_score)
        else:
            risk = 0.0

        # ── 5. 综合价值评分 ────────────────────────────────────────────────────
        value = (
            self.cfg.alpha * novelty
            - self.cfg.beta * redundancy
            - self.cfg.gamma * risk
        )

        meta = {
            "novelty": round(novelty, 4),
            "redundancy": round(redundancy, 4),
            "risk": round(risk, 4),
            "value": round(value, 4),
            "tau_write": self.cfg.tau_write,
            "cluster_density": round(density, 4),
        }

        if value < self.cfg.tau_write:
            return DefenseVerdict(
                node="D-B", passed=False,
                score=round(value, 4),
                reason=(
                    f"写入价值不足 (V={value:.3f} < τ={self.cfg.tau_write})："
                    f" novelty={novelty:.3f}, redundancy={redundancy:.3f},"
                    f" risk={risk:.3f}"
                ),
                action="BLOCK",
                metadata=meta,
            )

        return DefenseVerdict(
            node="D-B", passed=True,
            score=round(value, 4),
            reason=f"写入价值通过 (V={value:.3f} ≥ τ={self.cfg.tau_write})",
            action="PASS",
            metadata=meta,
        )
