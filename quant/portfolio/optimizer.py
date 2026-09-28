"""组合凸优化：截面 alpha 打分 → 目标权重。

架构 3.6 节的目标函数与约束：

    min  −αᵀw + λ·wᵀΣw + κ·|w − w_prev|₁
    s.t. Σw = 1,  0 ≤ w ≤ w_max,  |w − w_prev|₁ ≤ max_turnover

- ``Σ`` 用 :func:`quant.portfolio.risk.ledoit_wolf_covariance` 估计（identity
  target 收缩），输入收益矩阵列顺序必须与 ``instruments`` 一致，优化器内部校验。
- 权重上限留 buffer：``w_max`` 是**目标**上限（如 5%），约束内部按
  ``w_max × W_MAX_BUFFER``（默认 0.96）收紧，给后续整手取整留偏差余量。
- ``w_prev`` 是当前持仓权重，新仓传 ``None``（全零）。候选证券必须涵盖当前持仓，
  否则无法在换手约束里计入卖出。
- 无解时抛 :class:`InfeasibleError`，不放松约束；是否降级由调用方决定。

求解器用 cvxpy 默认（当前环境为 CLARABEL），问题固定、无随机性。
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import polars as pl

from quant.portfolio.risk import ledoit_wolf_covariance

# ---------------------------------------------------------------------------
# 配置（默认值集中在此）
# ---------------------------------------------------------------------------

#: 风险厌恶系数 λ。
LAMBDA_DEFAULT: float = 1.0

#: 换手惩罚系数 κ，内含印花税与佣金，抑制无效换手。
KAPPA_DEFAULT: float = 0.002

#: 单票目标权重上限（如 0.05 表示 5%）。
W_MAX_DEFAULT: float = 0.05

#: 双边换手上限，定义为 |w − w_prev|₁。
MAX_TURNOVER_DEFAULT: float = 0.30

#: 权重上限 buffer：约束内部用 w_max × buffer，吸收整手取整偏差。
W_MAX_BUFFER_DEFAULT: float = 0.96

#: 结果中权重绝对值小于该阈值的持仓视为约 0 并剔除。
ZERO_TOL_DEFAULT: float = 1e-6

_SOLVED_STATUSES = {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
_INFEASIBLE_STATUSES = {cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE}


# ---------------------------------------------------------------------------
# 异常与结果
# ---------------------------------------------------------------------------


class PortfolioError(Exception):
    """组合优化失败。"""


class InfeasibleError(PortfolioError):
    """约束不可行，且未做任何放松。调用方决定降级策略。"""


@dataclass(frozen=True)
class OptimizeResult:
    """优化结果。

    Attributes
    ----------
    weights:
        ``{instrument: weight}``，已剔除绝对值小于 ``zero_tol`` 的约零持仓。
    objective:
        最优目标值。
    turnover:
        实际换手 ``|w − w_prev|₁``（基于未剔除约零项的完整解）。
    status:
        cvxpy 求解状态字符串（``optimal`` / ``optimal_inaccurate``）。
    """

    weights: dict[str, float]
    objective: float
    turnover: float
    status: str


# ---------------------------------------------------------------------------
# 优化器
# ---------------------------------------------------------------------------


@dataclass
class PortfolioOptimizer:
    """凸组合优化器，参数默认值见模块顶部常量。"""

    lam: float = LAMBDA_DEFAULT
    kappa: float = KAPPA_DEFAULT
    w_max: float = W_MAX_DEFAULT
    max_turnover: float = MAX_TURNOVER_DEFAULT
    w_max_buffer: float = W_MAX_BUFFER_DEFAULT
    zero_tol: float = ZERO_TOL_DEFAULT
    solver: str | None = None

    def __post_init__(self) -> None:
        if self.lam < 0:
            raise ValueError(f"lam 不能为负: {self.lam}")
        if self.kappa < 0:
            raise ValueError(f"kappa 不能为负: {self.kappa}")
        if not 0 < self.w_max <= 1:
            raise ValueError(f"w_max 必须在 (0, 1]: {self.w_max}")
        if self.max_turnover < 0:
            raise ValueError(f"max_turnover 不能为负: {self.max_turnover}")
        if not 0 < self.w_max_buffer <= 1:
            raise ValueError(f"w_max_buffer 必须在 (0, 1]: {self.w_max_buffer}")
        if self.zero_tol < 0:
            raise ValueError(f"zero_tol 不能为负: {self.zero_tol}")

    def optimize(
        self,
        alpha: Mapping[str, float] | Sequence[float] | np.ndarray | pl.Series,
        instruments: Sequence[str],
        returns: pl.DataFrame | np.ndarray,
        w_prev: Mapping[str, float] | None = None,
        *,
        w_max: float | None = None,
    ) -> OptimizeResult:
        """求解目标权重。

        Parameters
        ----------
        alpha:
            截面打分，证券值越大越看多。支持 ``{instrument: score}`` 映射，或按
            ``instruments`` 顺序排列的序列 / numpy / polars Series。所有值必须有限。
        instruments:
            候选证券清单（顺序即收益矩阵列顺序）。当前持仓必须包含在内。
        returns:
            收益历史，用于估计协方差。``polars.DataFrame`` 的列顺序必须与
            ``instruments`` 完全一致；``numpy.ndarray`` 形状 ``(T, N)``。
        w_prev:
            当前持仓权重，``{instrument: weight}``；新仓传 ``None``。不得含候选集
            之外的证券。
        w_max:
            临时覆盖单票目标上限；仍会乘以 ``w_max_buffer`` 收紧。
        """
        instruments = list(instruments)
        if not instruments:
            raise ValueError("候选证券清单为空")
        if len(set(instruments)) != len(instruments):
            raise ValueError("候选证券清单存在重复")

        alpha_vec = _coerce_alpha(alpha, instruments)
        prev = _coerce_prev(w_prev, instruments)
        covariance = _coerce_covariance(returns, instruments)

        cap = self.w_max if w_max is None else float(w_max)
        if not 0 < cap <= 1:
            raise ValueError(f"w_max 必须在 (0, 1]: {cap}")
        upper = cap * self.w_max_buffer

        n = len(instruments)
        weights = cp.Variable(n)
        turnover_expr = cp.norm1(weights - prev)
        objective = (
            -alpha_vec @ weights
            + self.lam * cp.quad_form(weights, cp.psd_wrap(covariance))
            + self.kappa * turnover_expr
        )
        constraints = [
            cp.sum(weights) == 1,
            weights >= 0,
            weights <= upper,
            turnover_expr <= self.max_turnover,
        ]
        problem = cp.Problem(cp.Minimize(objective), constraints)
        problem.solve(solver=self.solver)

        status = str(problem.status)
        if status in _INFEASIBLE_STATUSES:
            raise InfeasibleError(
                f"组合约束不可行（w_max={cap}, buffer={self.w_max_buffer}, "
                f"max_turnover={self.max_turnover}, n={n}）"
            )
        if status not in _SOLVED_STATUSES:
            raise PortfolioError(f"求解失败，状态: {status}")

        full = np.clip(np.asarray(weights.value, dtype=np.float64).ravel(), 0.0, None)
        turnover = float(np.sum(np.abs(full - prev)))
        result_weights = {
            inst: float(w)
            for inst, w in zip(instruments, full)
            if abs(w) > self.zero_tol
        }
        return OptimizeResult(
            weights=result_weights,
            objective=float(problem.value),
            turnover=turnover,
            status=status,
        )


# ---------------------------------------------------------------------------
# 输入对齐
# ---------------------------------------------------------------------------


def _coerce_alpha(
    alpha: Mapping[str, float] | Sequence[float] | np.ndarray | pl.Series,
    instruments: list[str],
) -> np.ndarray:
    n = len(instruments)
    if isinstance(alpha, Mapping):
        missing = [inst for inst in instruments if inst not in alpha]
        if missing:
            raise ValueError(f"alpha 缺少候选证券: {missing}")
        vec = np.array([float(alpha[inst]) for inst in instruments], dtype=np.float64)
    elif isinstance(alpha, pl.Series):
        vec = alpha.to_numpy().astype(np.float64, copy=False)
    else:
        vec = np.asarray(alpha, dtype=np.float64).ravel()
    if vec.shape != (n,):
        raise ValueError(f"alpha 长度 {vec.shape} 与候选证券数 {n} 不符")
    if not np.isfinite(vec).all():
        raise ValueError("alpha 含 NaN / Inf")
    return vec


def _coerce_prev(w_prev: Mapping[str, float] | None, instruments: list[str]) -> np.ndarray:
    n = len(instruments)
    if w_prev is None:
        return np.zeros(n, dtype=np.float64)
    candidate_set = set(instruments)
    outside = [inst for inst in w_prev if inst not in candidate_set]
    if outside:
        raise ValueError(f"w_prev 含候选集之外的证券: {outside}")
    vec = np.array([float(w_prev.get(inst, 0.0)) for inst in instruments], dtype=np.float64)
    if not np.isfinite(vec).all():
        raise ValueError("w_prev 含 NaN / Inf")
    if (vec < -ZERO_TOL_DEFAULT).any():
        raise ValueError("w_prev 含负权重，第一版仅支持多头")
    return vec


def _coerce_covariance(
    returns: pl.DataFrame | np.ndarray, instruments: list[str]
) -> np.ndarray:
    """校验列对齐并返回 LW 收缩协方差。"""
    if isinstance(returns, pl.DataFrame):
        if list(returns.columns) != list(instruments):
            raise ValueError(
                "收益矩阵列与候选证券清单不一致（需按 instruments 顺序对齐）: "
                f"{list(returns.columns)} != {list(instruments)}"
            )
        covariance = ledoit_wolf_covariance(returns)
    else:
        matrix = np.asarray(returns, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(instruments):
            raise ValueError(
                f"收益矩阵形状 {matrix.shape} 与候选证券数 {len(instruments)} 不符"
            )
        covariance = ledoit_wolf_covariance(matrix)
    return (covariance + covariance.T) / 2.0
