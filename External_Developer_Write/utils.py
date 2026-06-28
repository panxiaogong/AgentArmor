"""
utils.py — 防御节点间的共享工具函数

包含数学运算（距离度量、矩阵操作）、文本处理（分词、关键词提取）、
以及 LLM/NLP 交互的辅助封装。所有函数纯计算、无状态，方便单元测试。
"""

import math
import re
from typing import List, Set, Dict, Any, Tuple, Optional, Callable
from collections import Counter


# ============================================================
# 数学工具
# ============================================================

EPSILON = 1e-10


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """计算两个向量的余弦相似度。

    sim(a, b) = (a · b) / (||a||₂ · ||b||₂ + ε)

    Args:
        a: 向量 A
        b: 向量 B

    Returns:
        [-1, 1] 范围的相似度，1 表示完全同向，-1 表示完全反向
    """
    if len(a) != len(b):
        raise ValueError(f"维度不匹配: {len(a)} vs {len(b)}")

    dot_ab = sum(ai * bi for ai, bi in zip(a, b))
    norm_a = math.sqrt(sum(ai * ai for ai in a) + EPSILON)
    norm_b = math.sqrt(sum(bi * bi for bi in b) + EPSILON)

    return dot_ab / (norm_a * norm_b + EPSILON)


def euclidean_distance(a: List[float], b: List[float]) -> float:
    """计算欧氏距离。

    d(a, b) = ||a - b||₂

    Args:
        a: 向量 A
        b: 向量 B

    Returns:
        非负距离值
    """
    if len(a) != len(b):
        raise ValueError(f"维度不匹配: {len(a)} vs {len(b)}")

    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)) + EPSILON)


def mean_vector(vectors: List[List[float]]) -> List[float]:
    """计算一组向量的均值（逐元素平均）。

    Args:
        vectors: 向量列表

    Returns:
        均值向量
    """
    if not vectors:
        return []
    dim = len(vectors[0])
    result = [0.0] * dim
    for vec in vectors:
        for i in range(dim):
            result[i] += vec[i]
    return [x / len(vectors) for x in result]


def dot_product(a: List[float], b: List[float]) -> float:
    """计算点积。"""
    return sum(ai * bi for ai, bi in zip(a, b))


def matrix_vector_multiply(matrix: List[List[float]], vec: List[float]) -> List[float]:
    """矩阵 × 向量 乘法。

    用于计算 Σ · v 或 Σ⁻¹ · v（当 matrix 为 Σ⁻¹ 时）。
    """
    n = len(matrix)
    m = len(vec)
    if n != m:
        raise ValueError(f"矩阵和向量维度不匹配: {n}x? vs {m}")
    return [sum(row[j] * vec[j] for j in range(m)) for row in matrix]


def cholesky_decomposition(sigma: List[List[float]]) -> List[List[float]]:
    """Cholesky 分解：正定矩阵 Σ = L @ L^T。

    使用改进的 Cholesky 算法（添加微小正则化保证正定性）。
    若矩阵非正定，自动添加 λI 正则化。

    Args:
        sigma: n×n 对称正定矩阵

    Returns:
        下三角矩阵 L，满足 Σ ≈ L @ L^T
    """
    n = len(sigma)
    L = [[0.0] * n for _ in range(n)]

    # 添加微小正则化保证数值稳定性
    reg = 1e-6
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # 检查对称性
            pass

    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                val = sigma[i][i] - s + reg  # 正则化
                if val <= 0:
                    # 仍不正定，继续增加正则化
                    val = abs(val) + 1e-4
                L[i][j] = math.sqrt(val)
            else:
                L[i][j] = (sigma[i][j] - s) / (L[j][j] + EPSILON)

    return L


def forward_substitution(L: List[List[float]], b: List[float]) -> List[float]:
    """前代法求解 L · y = b（L 是下三角矩阵）。"""
    n = len(L)
    y = [0.0] * n
    for i in range(n):
        s = sum(L[i][j] * y[j] for j in range(i))
        y[i] = (b[i] - s) / (L[i][i] + EPSILON)
    return y


def backward_substitution(U: List[List[float]], y: List[float]) -> List[float]:
    """回代法求解 U · x = y（U 是上三角矩阵）。"""
    n = len(U)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = sum(U[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (y[i] - s) / (U[i][i] + EPSILON)
    return x


def solve_cholesky(sigma: List[List[float]], diff: List[float]) -> List[float]:
    """用 Cholesky 分解求解 Σ^{-1} · diff。

    原理: Σ · t = diff → (L·L^T) · t = diff
          → 前代: L · y = diff
          → 回代: L^T · t = y
          → t = Σ^{-1} · diff

    避免直接求逆矩阵，数值稳定性更好。

    Args:
        sigma: 协方差矩阵 Σ
        diff:  偏离向量 (x - μ)

    Returns:
        t = Σ^{-1} · (x - μ)
    """
    L = cholesky_decomposition(sigma)
    y = forward_substitution(L, diff)
    t = backward_substitution(
        [[L[j][i] for j in range(len(L))] for i in range(len(L))],
        y
    )
    return t


def mahalanobis_distance(x: List[float], mu: List[float],
                         sigma: List[List[float]]) -> float:
    """马氏距离：D_M(x) = sqrt((x-μ)^T Σ^{-1} (x-μ))。"""
    diff = [xi - mui for xi, mui in zip(x, mu)]
    t = solve_cholesky(sigma, diff)
    return math.sqrt(dot_product(diff, t) + EPSILON)


def log2(x: float) -> float:
    """以 2 为底的对数。"""
    return math.log2(x) if x > 0 else 0.0


def mean(values: List[float]) -> float:
    """算术平均。"""
    if not values:
        return 0.0
    return sum(values) / len(values)


def std(values: List[float]) -> float:
    """标准差（总体标准差）。"""
    if len(values) < 2:
        return 0.0
    m = mean(values)
    variance = sum((v - m) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


def percentile(values: List[float], p: float) -> float:
    """计算百分位数。

    Args:
        values: 数据列表
        p:      百分位 (0~1)，如 0.025 表示 2.5% 分位数

    Returns:
        对应的分位数值
    """
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = p * (len(sorted_vals) - 1)
    lower = int(math.floor(idx))
    upper = int(math.ceil(idx))
    if lower == upper:
        return sorted_vals[lower]
    return sorted_vals[lower] * (upper - idx) + sorted_vals[upper] * (idx - lower)


# ============================================================
# 文本处理工具
# ============================================================

STOPWORDS: Set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "just", "because", "but", "and", "or", "if", "while", "although",
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
    "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
    "没有", "看", "好", "自己", "这", "他", "她", "它", "们"
}


def simple_tokenize(text: str) -> List[str]:
    """简单的分词函数 —— 按空白字符和非字母数字符号分割。

    对于英文为主的内容效果较好。中文内容建议替换为 jieba 等专用分词器。
    这是最小可用实现，生产环境请换用更好的分词器。

    Args:
        text: 输入文本

    Returns:
        token 列表
    """
    # 转为小写
    text = text.lower()
    # 按非字母数字分割（保留英文单词、数字、中文字符）
    tokens = re.findall(r"[a-z]+|[0-9]+|[一-鿿]+|[^\w\s]", text)
    return [t for t in tokens if t.strip()]


def extract_keywords(text: str, top_n: int = 10) -> Set[str]:
    """从文本中提取关键词（简单版：TF 过滤停用词）。

    生产环境可替换为 TextRank / KeyBERT 等更精确的方法。
    对应 SP5 中的关键词提取步骤。

    Args:
        text:  输入文本
        top_n: 返回 Top-N 关键词

    Returns:
        关键词集合（去重）
    """
    tokens = simple_tokenize(text)
    # 过滤停用词和单字符
    filtered = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    # 词频统计
    freq = Counter(filtered)
    # 返回 Top-N
    return set(word for word, _ in freq.most_common(top_n))


def is_refusal_response(response: str) -> bool:
    """检测 LLM 输出是否为"拒绝回答"或"信息不足"类型的弃权响应。

    Args:
        response: LLM 生成的文本

    Returns:
        True 表示该响应应被视为弃权
    """
    refusal_patterns = [
        "i don't know", "cannot answer", "unable to",
        "insufficient information", "no information",
        "i cannot", "i'm not sure", "i am not sure",
        "not enough context", "not enough information",
        "does not provide", "没有足够的信息", "无法回答",
        "不知道", "我不确定", "无法确定", "无法提供",
        "i don't have", "no relevant", "cannot determine",
    ]
    response_lower = response.lower().strip()
    return any(pattern in response_lower for pattern in refusal_patterns)


def is_empty_response(response: str) -> bool:
    """检测 LLM 输出是否为空或仅有空白字符。"""
    return not response or not response.strip()


def format_context(docs: List["RetrievedDoc"]) -> str:
    """将检索文档列表格式化为 LLM 上下文文本。

    Args:
        docs: 检索文档列表

    Returns:
        格式化的上下文字符串
    """
    parts = []
    for i, doc in enumerate(docs):
        parts.append(f"[Document {i + 1}]\n{doc.content}\n")
    return "\n".join(parts)


def build_prompt(query: str, context: str) -> str:
    """构建 LLM 提示。

    将用户查询和检索上下文组合为标准输入格式。

    Args:
        query:   用户问题
        context: 检索到的上下文文档（已格式化）

    Returns:
        完整提示文本
    """
    return (
        f"请基于以下参考文档回答问题。\n\n"
        f"参考文档:\n{context}\n\n"
        f"问题: {query}\n\n"
        f"回答:"
    )
