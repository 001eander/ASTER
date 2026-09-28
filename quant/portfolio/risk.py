"""Ledoit-Wolf 收缩协方差（identity target）。

实现 Ledoit & Wolf (2004) *A well-conditioned estimator for large-dimensional
covariance matrices* 的 identity-target 版本：把样本协方差矩阵 ``S`` 收缩到
``μI``（``μ = tr(S) / N``），得到

    Σ̂ = δ · μI + (1 − δ) · S

收缩强度 ``δ ∈ [0, 1]`` 由数据估计：

- ``π̂ = Σ_ij (1/T) Σ_t (x_it x_jt − s_ij)²``：样本协方差各元素渐近方差之和。
  实现用等价向量化形式 ``π̂ = mean_t(‖x_t‖⁴) − ‖S‖²_F``。
- ``ρ̂ = (1/N) Σ_i π̂_ii``：identity target 与样本协方差的渐近协方差之和。
  target 的对角元 ``μ`` 是样本对角元的平均，故相对 ``π̂_ii`` 带 ``1/N`` 因子；
  ``π̂_ii = mean_t(x_it⁴) − s_ii²``。
- ``γ̂ = ‖S − μI‖²_F``：target 的误设程度。
- ``δ = clip( (π̂ − ρ̂) / (T · γ̂), 0, 1)``。

``γ̂ = 0``（``S`` 本身即 ``μI``）时无需收缩，取 ``δ = 0``；``N = 1`` 时
``π̂ = ρ̂``，同样退化到 ``δ = 0``。

identity target 的误设 ``γ`` 取决于真实协方差与 ``μI`` 的距离：真实协方差本就
接近 ``μI`` 时 ``δ`` 趋近 1（收缩到 target 最优），真实协方差结构明显（例如股票
间恒定正相关）时 ``γ`` 量级大，``δ`` 大致按 ``1/T`` 衰减。这是 identity target
的固有性质，不是实现偏差。

输入约定
--------
``returns`` 是收益矩阵，行 = 日期，列 = 证券，两种形式均可：

- ``polars.DataFrame``：每列一只证券，列必须全是数值类型；输出协方差的证券顺序
  与 DataFrame 列顺序一致。
- ``numpy.ndarray``：形状 ``(T, N)``，调用方保证行列顺序与证券清单对齐。

默认不假设零均值（按列去均值）；``assume_centered=True`` 时跳过去均值。输出为
``(N, N)`` 的 numpy 协方差矩阵，与输入列顺序一致。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

#: 至少需要两个观测才能估计协方差（去均值后不至全零）。
MIN_OBSERVATIONS: int = 2

#: 判定 ``γ̂`` 是否为零的相对阈值（相对 ``N · μ²``）。
GAMMA_REL_EPS: float = 1e-12


# ---------------------------------------------------------------------------
# 结果对象
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LedoitWolfResult:
    """收缩协方差估计结果。

    Attributes
    ----------
    covariance:
        ``(N, N)`` 收缩协方差矩阵。
    shrinkage:
        收缩强度 ``δ ∈ [0, 1]``，即 target ``μI`` 的权重。
    """

    covariance: np.ndarray
    shrinkage: float


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------


def _to_matrix(returns: pl.DataFrame | np.ndarray, *, assume_centered: bool) -> np.ndarray:
    """把输入统一成去均值后的 ``(T, N)`` float64 数组。"""
    if isinstance(returns, pl.DataFrame):
        if returns.width == 0:
            raise ValueError("收益矩阵没有任何证券列")
        non_numeric = [c for c, dt in returns.schema.items() if not dt.is_numeric()]
        if non_numeric:
            raise ValueError(f"收益矩阵含非数值列: {non_numeric}")
        matrix = returns.to_numpy().astype(np.float64, copy=False)
    elif isinstance(returns, np.ndarray):
        matrix = np.asarray(returns, dtype=np.float64)
    else:
        raise TypeError(f"不支持的收益矩阵类型: {type(returns)!r}")

    if matrix.ndim != 2:
        raise ValueError(f"收益矩阵必须是二维（T, N），实际 {matrix.ndim} 维")
    t, n = matrix.shape
    if t < MIN_OBSERVATIONS:
        raise ValueError(f"观测数 T={t} 少于 {MIN_OBSERVATIONS}，无法估计协方差")
    if n == 0:
        raise ValueError("收益矩阵没有任何证券列")
    if not np.isfinite(matrix).all():
        raise ValueError("收益矩阵含 NaN / Inf，请先清洗")

    if not assume_centered:
        matrix = matrix - matrix.mean(axis=0, keepdims=True)
    return matrix


def _shrinkage_intensity(x: np.ndarray) -> tuple[float, np.ndarray, float]:
    """由去均值收益计算 ``(δ, S, μ)``。"""
    t, n = x.shape
    sample = (x.T @ x) / t
    mu = float(np.trace(sample) / n)

    # π̂ = mean_t(‖x_t‖⁴) − ‖S‖²_F
    row_sq = np.einsum("ij,ij->i", x, x)
    pi = float(np.mean(row_sq**2) - np.sum(sample * sample))

    # ρ̂ = (1/N) Σ_i π̂_ii，π̂_ii = mean_t(x_it⁴) − s_ii²
    diag_sq = np.diag(sample)
    pi_diag = float(np.sum(np.mean(x**4, axis=0) - diag_sq * diag_sq))
    rho = pi_diag / n

    gamma = float(np.sum(sample * sample) - n * mu * mu)
    if gamma <= GAMMA_REL_EPS * abs(n * mu * mu):
        # S 已经是 μI（或数值退化），无需收缩。
        return 0.0, sample, mu

    kappa = (pi - rho) / gamma
    delta = float(np.clip(kappa / t, 0.0, 1.0))
    return delta, sample, mu


# ---------------------------------------------------------------------------
# 公开接口
# ---------------------------------------------------------------------------


def ledoit_wolf_shrinkage(
    returns: pl.DataFrame | np.ndarray, *, assume_centered: bool = False
) -> float:
    """返回 Ledoit-Wolf 收缩强度 ``δ ∈ [0, 1]``。"""
    x = _to_matrix(returns, assume_centered=assume_centered)
    delta, _, _ = _shrinkage_intensity(x)
    return delta


def ledoit_wolf_covariance(
    returns: pl.DataFrame | np.ndarray, *, assume_centered: bool = False
) -> np.ndarray:
    """返回 Ledoit-Wolf 收缩协方差矩阵，形状 ``(N, N)``。

    证券顺序与输入列顺序一致；矩阵对称，且当 ``μ > 0`` 时正定。
    """
    delta, sample, mu = _shrinkage_intensity(
        _to_matrix(returns, assume_centered=assume_centered)
    )
    n = sample.shape[0]
    covariance = (1.0 - delta) * sample + delta * mu * np.eye(n)
    return (covariance + covariance.T) / 2.0


def ledoit_wolf(
    returns: pl.DataFrame | np.ndarray, *, assume_centered: bool = False
) -> LedoitWolfResult:
    """同时返回收缩协方差与收缩强度。"""
    delta, sample, mu = _shrinkage_intensity(
        _to_matrix(returns, assume_centered=assume_centered)
    )
    n = sample.shape[0]
    covariance = (1.0 - delta) * sample + delta * mu * np.eye(n)
    covariance = (covariance + covariance.T) / 2.0
    return LedoitWolfResult(covariance=covariance, shrinkage=delta)
