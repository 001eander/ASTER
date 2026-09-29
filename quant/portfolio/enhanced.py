"""指数增强优化器：基准锚定约束族 + 放松阶梯 + 失败漂移（issue #68）。

范式参照 ``reference-portfolio-optimization.py``（pandas 原型，只作口径参照），
本模块用 polars / numpy / cvxpy 重写，约束数学形式与原型逐条对齐（见 PR 对拍）。

目标函数
--------
``min −αᵀw + λ·wᵀΣw``，``Σ`` 由 :func:`quant.portfolio.risk.ledoit_wolf_covariance`
估计。原型的目标函数是不含风险项的 ``min −sᵀw``，对拍时取 ``λ = 0`` 即可逐位比较。

约束族（全部相对基准，参数化）
------------------------------
- 满仓 ``Σw ∈ [1 − FULL_INVESTMENT_TOL, 1 + FULL_INVESTMENT_TOL]``、禁止卖空 ``w ≥ 0``。
- 个股带：成分股 ``w ∈ [bench_w − band, bench_w + band]``，非成分股
  ``w ∈ [non_member_min, non_member_max]``；``bench_w`` 为基准权重在候选集内归一后的值。
- 行业偏离：组合行业权重 ∈ 基准行业权重 ``×(1 ± ratio)``，行业用东财一级。
- 市值偏离：组合加权等效市值 ∈ 基准值 ``± std`` 倍；基准 std 取
  ``std(基准权重 × 成分等效市值, ddof=1)``（沿用原型的量，见 PR 说明）。
- 风格暴露：六因子各自 ∈ 基准暴露 ``± std`` 倍，基准 std 取成分内因子值的
  ``std(ddof=1)``（沿用原型）。
- 成分覆盖度：指数成分股权重之和 ≥ ``cover_rate_min``。
- 换手：``Σ|w − w_prev| ≤ turnover_max``，``w_prev`` 为漂移后实际权重。

工程机制
--------
1. 可行性预检 ``Minimize(0)`` 同约束求解；不可行进入放松阶梯。
2. 放松阶梯顺序：风格 → 行业/市值 → 换手，步长入配置，最多 ``max_relax_rounds`` 轮。
3. 放松用尽仍不可行时进入**终局台阶**（issue #71）：回到初始阈值、只把换手上限放到
   :data:`TURNOVER_FREE_MAX` 再解一次，打断「held → 漂移更远 → 更不可行」的自锁；
   台阶仍不可行才保持持仓，当日权重按漂移结果记录日志。
4. 调仓频率 ``D/W/M``，非调仓日权重按区间收益漂移（``drift_weights``）。
5. 优化日志逐日产出在 :class:`EnhancedResult.log`，落盘由调用方负责
   （本模块不发 IO，:func:`logs_to_frame` 汇总为 polars 表）。

无前视
------
所有输入只应包含 cutoff 当日及之前的数据；本模块不做时间过滤，由调用方保证。
``w_prev`` 是上一调仓日盘后权重经调仓区间收益漂移后的实际权重。
"""
from __future__ import annotations

import logging
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date as Date
from typing import Any

import cvxpy as cp
import numpy as np
import polars as pl

from quant.portfolio.risk import ledoit_wolf_covariance
from quant.portfolio.style import STYLE_FACTOR_NAMES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 配置（默认值集中在此）
# ---------------------------------------------------------------------------

#: 风险厌恶系数 λ。
LAMBDA_DEFAULT: float = 1.0

#: 成分股权重相对基准的带宽（单边）。
STOCK_BAND_DEFAULT: float = 0.005
#: 非成分股权重上限 / 下限。
NON_MEMBER_MAX_DEFAULT: float = 0.005
NON_MEMBER_MIN_DEFAULT: float = 0.0

#: 行业偏离比例阈值（相对基准行业权重）。
INDUSTRY_EXPOSURE_DEFAULT: float = 0.005
#: 市值偏离 std 倍数阈值。
MARKET_VALUE_EXPOSURE_DEFAULT: float = 0.005
#: 风格暴露 std 倍数阈值。
STYLE_EXPOSURE_DEFAULT: float = 0.005
#: 成分覆盖度下限。
COVER_RATE_MIN_DEFAULT: float = 0.5
#: 双边换手上限 ``Σ|w − w_prev|``。
TURNOVER_MAX_DEFAULT: float = 0.2

#: 放松阶梯步长。
INDUSTRY_RELAX_STEP_DEFAULT: float = 0.001
MARKET_VALUE_RELAX_STEP_DEFAULT: float = 0.001
STYLE_RELAX_STEP_DEFAULT: float = 0.001
TURNOVER_RELAX_STEP_DEFAULT: float = 0.01
#: 放松轮数上限（与原型一致）。
MAX_RELAX_ROUNDS_DEFAULT: int = 60

#: 满仓约束容差：``Σw ∈ [1 − tol, 1 + tol]``。
FULL_INVESTMENT_TOL: float = 1e-4
#: 求解结果中绝对值小于该阈值的权重视为约 0 并剔除。
ZERO_TOL_DEFAULT: float = 1e-4

#: 终局台阶的换手上限：长多组合 ``|w − w_prev|₁ ≤ Σw + Σw_prev ≤ 2``，取 2.0 等价于取消换手约束。
TURNOVER_FREE_MAX: float = 2.0
#: 终局台阶在阈值字典里留下的标记键。
FINAL_TIER_FLAG: str = "final_turnover_free"

#: 调仓频率 → 交易日步长（沿用原型的 ``[::1] / [::5] / [::20]``）。
REBALANCE_STRIDE: dict[str, int] = {"D": 1, "W": 5, "M": 20}

# 状态标记
STATUS_OPTIMAL: str = "optimal"
STATUS_RELAXED: str = "relaxed"
STATUS_HELD: str = "held"
STATUS_DRIFT: str = "drift"

# 行业缺失占位符：候选侧与基准侧分开，保证缺失行业互不匹配（沿用原型口径）。
CAND_UNKNOWN_INDUSTRY: str = "<cand_unknown>"
BENCH_UNKNOWN_INDUSTRY: str = "<bench_unknown>"

_SOLVED_STATUSES = {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}

#: 日志表 schema。
LOG_SCHEMA = pl.Schema(
    {
        "date": pl.Date,
        "status": pl.String,
        "relax_rounds": pl.Int64,
        "turnover": pl.Float64,
        "thresholds": pl.String,
        "exposures": pl.String,
    }
)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnhancedResult:
    """单日优化 / 漂移结果。

    Attributes
    ----------
    weights:
        当日盘后目标权重 ``{instrument: weight}``，已剔除约零项并归一。
    status:
        :data:`STATUS_OPTIMAL` / :data:`STATUS_RELAXED` / :data:`STATUS_HELD`
        / :data:`STATUS_DRIFT`。
    relax_rounds:
        成功求解前施加的放松次数；与原型日志的 ``relax_constraints_count``
        对齐（``0`` 表示首次即成功）。全程失败时为 ``max_relax_rounds − 1``。
    thresholds:
        求解时的约束阈值字典；失败时为最后一轮放松后的阈值，终局台阶命中时含
        ``turnover = 2.0`` 与 :data:`FINAL_TIER_FLAG` 标记。
    exposures:
        实际暴露明细，见 :func:`compute_exposures`。口径统一为**归一后的持仓**
        （``Σw = 1``）：求解分支即目标权重，held 分支为漂移后持仓的归一权重。
    turnover:
        当日双边换手；非调仓 / 失败日为 0，首次建仓日按原型取 1.0。
    objective:
        最优目标值，失败 / 漂移日为 ``None``。
    log:
        扁平日志行（``date/status/relax_rounds/turnover/thresholds/exposures``）。
    """

    weights: dict[str, float]
    status: str
    relax_rounds: int
    thresholds: dict[str, Any]
    exposures: dict[str, Any]
    turnover: float
    objective: float | None
    log: dict[str, Any]


@dataclass(frozen=True)
class EnhancedDayInput:
    """单日优化输入（由调用方按 cutoff 组装，保证无前视）。

    Attributes
    ----------
    date:
        当日日期。
    instruments:
        候选证券清单（顺序即协方差列顺序）。
    alpha:
        模型得分 ``{instrument: score}``，必须覆盖全部候选。
    bench_weights:
        基准成分权重 ``{instrument: weight}``（原始口径，和约 1），只含成分股。
    industry:
        ``{instrument: 东财一级行业}``，覆盖候选与成分；缺失自动用占位符。
    float_mv:
        ``{instrument: 等效市值}``，覆盖候选与成分。
    style:
        ``{instrument: {factor: value}}``，覆盖候选与成分。
    covariance:
        收益历史（``pl.DataFrame`` 列顺序须与 ``instruments`` 一致）或 ``(T, N)`` 数组。
    interval_returns:
        自上一调仓日以来各证券的区间收益 ``{instrument: return}``，用于权重漂移；
        缺失按 0 处理。
    """

    date: Date
    instruments: list[str]
    alpha: Mapping[str, float]
    bench_weights: Mapping[str, float]
    industry: Mapping[str, str]
    float_mv: Mapping[str, float]
    style: Mapping[str, Mapping[str, float]]
    covariance: pl.DataFrame | np.ndarray
    interval_returns: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class _ProblemData:
    """约束族固定部分（放松只改阈值，不改矩阵）。"""

    instruments: list[str]
    n: int
    alpha: np.ndarray
    covariance: np.ndarray
    lam: float
    full_investment_tol: float
    lower: np.ndarray
    upper: np.ndarray
    bench_norm: np.ndarray
    ind_a: np.ndarray
    ind_b: np.ndarray
    mv_a: np.ndarray
    mv_center: float
    mv_scale: float
    is_member: np.ndarray
    style_a: dict[str, np.ndarray]
    style_center: dict[str, float]
    style_scale: dict[str, float]
    prev: np.ndarray
    prev_valid: bool
    style_factor_names: tuple[str, ...]


# ---------------------------------------------------------------------------
# 优化器
# ---------------------------------------------------------------------------


@dataclass
class EnhancedOptimizer:
    """指数增强凸优化器，默认参数见模块顶部常量。"""

    lam: float = LAMBDA_DEFAULT
    stock_band: float = STOCK_BAND_DEFAULT
    non_member_max: float = NON_MEMBER_MAX_DEFAULT
    non_member_min: float = NON_MEMBER_MIN_DEFAULT
    industry_exposure: float = INDUSTRY_EXPOSURE_DEFAULT
    market_value_exposure: float = MARKET_VALUE_EXPOSURE_DEFAULT
    style_exposure: float = STYLE_EXPOSURE_DEFAULT
    cover_rate_min: float = COVER_RATE_MIN_DEFAULT
    turnover_max: float = TURNOVER_MAX_DEFAULT
    industry_relax_step: float = INDUSTRY_RELAX_STEP_DEFAULT
    market_value_relax_step: float = MARKET_VALUE_RELAX_STEP_DEFAULT
    style_relax_step: float = STYLE_RELAX_STEP_DEFAULT
    turnover_relax_step: float = TURNOVER_RELAX_STEP_DEFAULT
    max_relax_rounds: int = MAX_RELAX_ROUNDS_DEFAULT
    full_investment_tol: float = FULL_INVESTMENT_TOL
    zero_tol: float = ZERO_TOL_DEFAULT
    frequency: str = "D"
    style_factor_names: tuple[str, ...] = STYLE_FACTOR_NAMES
    solver: str | None = None

    def __post_init__(self) -> None:
        if self.lam < 0:
            raise ValueError(f"lam 不能为负: {self.lam}")
        for name in (
            "stock_band",
            "non_member_max",
            "non_member_min",
            "industry_exposure",
            "market_value_exposure",
            "style_exposure",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} 不能为负: {getattr(self, name)}")
        if not 0.0 <= self.cover_rate_min <= 1.0:
            raise ValueError(f"cover_rate_min 必须在 [0, 1]: {self.cover_rate_min}")
        if self.turnover_max < 0:
            raise ValueError(f"turnover_max 不能为负: {self.turnover_max}")
        if self.max_relax_rounds < 1:
            raise ValueError(f"max_relax_rounds 至少为 1: {self.max_relax_rounds}")
        if self.frequency not in REBALANCE_STRIDE:
            raise ValueError(
                f"frequency 必须是 {sorted(REBALANCE_STRIDE)} 之一: {self.frequency}"
            )

    # -- 单日优化 -----------------------------------------------------------

    def optimize_day(
        self,
        *,
        date: Date,
        instruments: Sequence[str],
        alpha: Mapping[str, float] | Sequence[float] | np.ndarray | pl.Series,
        bench_weights: Mapping[str, float],
        industry: Mapping[str, str],
        float_mv: Mapping[str, float],
        style: Mapping[str, Mapping[str, float]],
        covariance: pl.DataFrame | np.ndarray,
        w_prev: Mapping[str, float] | None = None,
    ) -> EnhancedResult:
        """求解单日目标权重，含可行性预检、放松阶梯与失败保持持仓。"""
        insts = list(instruments)
        if not insts:
            raise ValueError("候选证券清单为空")
        if len(set(insts)) != len(insts):
            raise ValueError("候选证券清单存在重复")

        data = _prepare(
            instruments=insts,
            alpha=_coerce_alpha(alpha, insts),
            bench_weights=bench_weights,
            industry=industry,
            float_mv=float_mv,
            style=style,
            covariance=_coerce_covariance(covariance, insts),
            w_prev=w_prev,
            lam=self.lam,
            full_investment_tol=self.full_investment_tol,
            stock_band=self.stock_band,
            non_member_max=self.non_member_max,
            non_member_min=self.non_member_min,
            style_factor_names=self.style_factor_names,
        )

        thresholds = self._initial_thresholds()
        solved, relax_rounds, used = self._solve_with_relaxation(data, thresholds)

        if solved is None:
            held_raw = {inst: float(w) for inst, w in zip(insts, data.prev) if w > 0}
            # 暴露口径统一到「归一后的持仓」：求解分支的权重和恒为 1，held 分支的漂移权重
            # 和可能小于 1（现金 / 未落地的取整残差），不归一会让覆盖度与带偏离无法横向比较。
            held = _normalize(held_raw)
            exposures = compute_exposures(
                held,
                bench_weights=bench_weights,
                industry=industry,
                float_mv=float_mv,
                style=style,
                style_factor_names=self.style_factor_names,
                turnover=0.0,
            )
            return _make_result(
                date=date,
                weights=held,
                status=STATUS_HELD,
                relax_rounds=relax_rounds,
                thresholds=used,
                exposures=exposures,
                turnover=0.0,
                objective=None,
            )

        raw, full, objective_value = solved
        turnover = (
            float(np.sum(np.abs(raw - data.prev))) if data.prev_valid else 1.0
        )
        weights = {inst: float(w) for inst, w in zip(insts, full) if w > 0.0}
        exposures = compute_exposures(
            weights,
            bench_weights=bench_weights,
            industry=industry,
            float_mv=float_mv,
            style=style,
            style_factor_names=self.style_factor_names,
            turnover=turnover,
        )
        status = STATUS_OPTIMAL if relax_rounds == 0 else STATUS_RELAXED
        return _make_result(
            date=date,
            weights=weights,
            status=status,
            relax_rounds=relax_rounds,
            thresholds=used,
            exposures=exposures,
            turnover=turnover,
            objective=objective_value,
        )

    # -- 多日驱动 -----------------------------------------------------------

    def run(
        self,
        days: Sequence[EnhancedDayInput],
        *,
        rebalance_dates: Sequence[Date] | None = None,
    ) -> list[EnhancedResult]:
        """按调仓计划遍历多日，非调仓日按区间收益漂移，返回逐日结果。

        ``rebalance_dates`` 显式给定调仓日；缺省按 ``frequency`` 的交易步长
        （``D = 1 / W = 5 / M = 20``）取 ``days`` 中每第 k 天调仓。
        """
        if rebalance_dates is not None:
            schedule = set(rebalance_dates)
        else:
            stride = REBALANCE_STRIDE[self.frequency]
            schedule = {day.date for pos, day in enumerate(days) if pos % stride == 0}

        results: list[EnhancedResult] = []
        prev: dict[str, float] = {}
        for day in days:
            drifted = drift_weights(prev, day.interval_returns) if prev else {}
            if day.date in schedule:
                result = self.optimize_day(
                    date=day.date,
                    instruments=day.instruments,
                    alpha=day.alpha,
                    bench_weights=day.bench_weights,
                    industry=day.industry,
                    float_mv=day.float_mv,
                    style=day.style,
                    covariance=day.covariance,
                    w_prev=drifted or None,
                )
            else:
                result = self._held_result(day, drifted)
            results.append(result)
            prev = result.weights
        return results

    # -- 内部 ---------------------------------------------------------------

    def _initial_thresholds(self) -> dict[str, float]:
        return {
            "industry": self.industry_exposure,
            "market_value": self.market_value_exposure,
            "cover_rate": self.cover_rate_min,
            "turnover": self.turnover_max,
            "style": self.style_exposure,
        }

    def _relax(self, thresholds: dict[str, float], count: int) -> dict[str, float]:
        """放松阶梯：风格 → 行业/市值 → 换手，每 3 轮循环一次（照原型）。"""
        out = dict(thresholds)
        if count % 3 == 0:
            out["style"] += self.style_relax_step
        elif count % 3 == 1:
            out["industry"] += self.industry_relax_step
            out["market_value"] += self.market_value_relax_step
        else:
            out["turnover"] += self.turnover_relax_step
        return out

    def _check_feasibility(self, constraints: list[cp.Constraint]) -> bool:
        try:
            problem = cp.Problem(cp.Minimize(0), constraints)
            problem.solve(solver=self._solver())
            return problem.status in _SOLVED_STATUSES
        except Exception:  # noqa: BLE001 - 求解器异常一律视为不可行
            return False

    def _solver(self) -> str:
        """求解器：显式配置优先，否则用 CLARABEL（与 issue 口径一致）。"""
        return self.solver or cp.CLARABEL

    def _solve_with_relaxation(
        self, data: _ProblemData, thresholds: dict[str, float]
    ) -> tuple[tuple[np.ndarray, np.ndarray, float] | None, int, dict[str, Any]]:
        """预检 + 放松循环 + 终局台阶。

        返回 ``((原始解, 归一解, 目标值), 放松轮数, 所用阈值)``；全程失败时首项为 ``None``。
        放松轮数与原型日志的 ``relax_constraints_count`` 对齐。

        终局台阶（issue #71）：``max_relax_rounds`` 轮放松用尽仍不可行时，回到**初始阈值**
        只把换手上限放到 :data:`TURNOVER_FREE_MAX`（等价取消换手约束）再解一次。这是为了
        打断「不可行 → held → 漂移更远 → 更不可行」的自锁：恢复持仓回到带内所需的换手
        可能超过换手约束允许的上限，此时唯一出路是先允许偏离再收敛。命中台阶时状态记
        :data:`STATUS_RELAXED`、``relax_rounds`` 记 ``max_relax_rounds``，阈值字典带
        :data:`FINAL_TIER_FLAG` 标记。台阶仍不可行才返回 held 语义（首项 ``None``）。
        """
        w = cp.Variable(data.n)
        if data.bench_norm.sum() > 0:
            w.value = data.bench_norm
        objective = _build_objective(w, data)
        initial = dict(thresholds)
        current = dict(thresholds)
        constraints = _build_constraints(w, data, current)

        feasible = False
        for count in range(self.max_relax_rounds):
            if not feasible:
                feasible = self._check_feasibility(constraints)
            if feasible:
                raw = self._solve_problem(w, objective, constraints)
                if raw is not None:
                    full = _finalize(raw, self.zero_tol)
                    return (raw, full, float(objective.value)), count, current
            current = self._relax(current, count)
            constraints = _build_constraints(w, data, current)

        final_tier = dict(initial)
        final_tier["turnover"] = TURNOVER_FREE_MAX
        final_tier[FINAL_TIER_FLAG] = True
        constraints = _build_constraints(w, data, final_tier)
        if self._check_feasibility(constraints):
            raw = self._solve_problem(w, objective, constraints)
            if raw is not None:
                full = _finalize(raw, self.zero_tol)
                return (
                    (raw, full, float(objective.value)),
                    self.max_relax_rounds,
                    final_tier,
                )
        return None, max(0, self.max_relax_rounds - 1), current

    def _solve_problem(
        self, w: cp.Variable, objective: cp.Minimize, constraints: list[cp.Constraint]
    ) -> np.ndarray | None:
        try:
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=self._solver())
        except Exception:  # noqa: BLE001
            return None
        if problem.status not in _SOLVED_STATUSES or w.value is None:
            return None
        value = np.asarray(w.value, dtype=np.float64).ravel()
        if not np.isfinite(value).all():
            return None
        return np.clip(value, 0.0, None)

    def _held_result(
        self, day: EnhancedDayInput, drifted: dict[str, float]
    ) -> EnhancedResult:
        thresholds = self._initial_thresholds()
        exposures = compute_exposures(
            drifted,
            bench_weights=day.bench_weights,
            industry=day.industry,
            float_mv=day.float_mv,
            style=day.style,
            style_factor_names=self.style_factor_names,
            turnover=0.0,
        )
        return _make_result(
            date=day.date,
            weights=dict(drifted),
            status=STATUS_DRIFT,
            relax_rounds=0,
            thresholds=thresholds,
            exposures=exposures,
            turnover=0.0,
            objective=None,
        )


# ---------------------------------------------------------------------------
# 漂移与暴露
# ---------------------------------------------------------------------------


def drift_weights(
    weights: Mapping[str, float], interval_returns: Mapping[str, float]
) -> dict[str, float]:
    """调仓区间权重漂移（原型的 ``close_weight`` 逻辑）。

    ``w_i ← w_i × (1 + r_i)`` 后归一；缺失区间收益按 0 处理（权重不变）。
    漂移后总和非正时返回空字典（视为清仓）。
    """
    if not weights:
        return {}
    raw = {
        inst: float(w) * (1.0 + float(interval_returns.get(inst, 0.0)))
        for inst, w in weights.items()
    }
    total = sum(raw.values())
    if not math.isfinite(total) or total <= 0.0:
        return {}
    return {inst: value / total for inst, value in raw.items() if value > 0.0}


def compute_exposures(
    weights: Mapping[str, float],
    *,
    bench_weights: Mapping[str, float],
    industry: Mapping[str, str],
    float_mv: Mapping[str, float],
    style: Mapping[str, Mapping[str, float]],
    style_factor_names: Sequence[str] = STYLE_FACTOR_NAMES,
    turnover: float = 0.0,
) -> dict[str, Any]:
    """计算实际暴露：覆盖度、换手、市值、行业与六风格。

    基准侧口径与约束族一致：市值基准 std 取 ``std(基准权重 × 成分等效市值, ddof=1)``，
    风格基准 std 取成分内因子值的 ``std(ddof=1)``。
    """
    weights = {inst: float(w) for inst, w in weights.items() if w > 0.0}
    constituents = [inst for inst, w in bench_weights.items() if w > 0.0]
    member_set = set(constituents)
    bench_values = np.array([float(bench_weights[i]) for i in constituents], dtype=float)

    cover_rate = float(sum(w for inst, w in weights.items() if inst in member_set))

    port_mv = sum(w * float(float_mv.get(inst, 0.0)) for inst, w in weights.items())
    if constituents:
        bench_mv = bench_values * np.array(
            [float(float_mv.get(i, 0.0)) for i in constituents], dtype=float
        )
        mv_center = float(bench_mv.sum())
        mv_scale = float(np.std(bench_mv, ddof=1)) if len(bench_mv) > 1 else 0.0
    else:
        mv_center, mv_scale = 0.0, 0.0
    market_value = {
        "ratio": (port_mv - mv_center) / mv_center if mv_center else 0.0,
        "std": (port_mv - mv_center) / mv_scale if mv_scale > 0 else 0.0,
    }

    industry_exp = _industry_exposure(weights, bench_weights, industry)

    style_exp: dict[str, dict[str, float]] = {}
    for factor in style_factor_names:
        port_f = sum(
            w * float(style.get(inst, {}).get(factor, 0.0)) for inst, w in weights.items()
        )
        if constituents:
            bench_f = np.array(
                [float(style.get(i, {}).get(factor, 0.0)) for i in constituents],
                dtype=float,
            )
            center = float((bench_values * bench_f).sum())
            scale = float(np.std(bench_f, ddof=1)) if len(bench_f) > 1 else 0.0
        else:
            center, scale = 0.0, 0.0
        style_exp[factor] = {
            "exposure": port_f,
            "ratio": (port_f - center) / center if center else 0.0,
            "std": (port_f - center) / scale if scale > 0 else 0.0,
        }

    return {
        "cover_rate": cover_rate,
        "turnover": float(turnover),
        "market_value": market_value,
        "industry": industry_exp,
        "style": style_exp,
    }


def _industry_exposure(
    weights: Mapping[str, float],
    bench_weights: Mapping[str, float],
    industry: Mapping[str, str],
) -> dict[str, dict[str, float]]:
    """组合与基准的行业权重（基准在候选行业内重新归一，与约束一致）。"""
    cand_codes = {
        industry.get(inst) or CAND_UNKNOWN_INDUSTRY for inst in weights
    }
    bench_raw: dict[str, float] = defaultdict(float)
    for inst, w in bench_weights.items():
        if w > 0.0:
            bench_raw[industry.get(inst) or BENCH_UNKNOWN_INDUSTRY] += float(w)
    universe = sorted(set(cand_codes) | set(bench_raw))
    total_bench = sum(bench_raw.get(code, 0.0) for code in universe)
    out: dict[str, dict[str, float]] = {}
    for code in universe:
        bench = bench_raw.get(code, 0.0) / total_bench if total_bench else 0.0
        port = float(
            sum(
                w
                for inst, w in weights.items()
                if (industry.get(inst) or CAND_UNKNOWN_INDUSTRY) == code
            )
        )
        out[code] = {
            "portfolio": port,
            "bench": bench,
            "ratio": (port - bench) / bench if bench else 0.0,
        }
    return out


def logs_to_frame(results: Sequence[EnhancedResult]) -> pl.DataFrame:
    """把逐日结果的日志行汇总为 polars 表，便于落盘。"""
    rows = [
        {
            "date": r.log["date"],
            "status": r.log["status"],
            "relax_rounds": r.log["relax_rounds"],
            "turnover": r.log["turnover"],
            "thresholds": str(r.log["thresholds"]),
            "exposures": str(r.log["exposures"]),
        }
        for r in results
    ]
    if not rows:
        return pl.DataFrame(schema=LOG_SCHEMA)
    return pl.DataFrame(rows, schema=LOG_SCHEMA)


# ---------------------------------------------------------------------------
# 输入对齐与约束构造
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


def _coerce_covariance(
    returns: pl.DataFrame | np.ndarray, instruments: list[str]
) -> np.ndarray:
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


def _map_to_vec(
    values: Mapping[str, float], instruments: Sequence[str], default: float = 0.0
) -> np.ndarray:
    return np.array(
        [float(values.get(inst, default)) for inst in instruments], dtype=np.float64
    )


def _prepare(
    *,
    instruments: list[str],
    alpha: np.ndarray,
    bench_weights: Mapping[str, float],
    industry: Mapping[str, str],
    float_mv: Mapping[str, float],
    style: Mapping[str, Mapping[str, float]],
    covariance: np.ndarray,
    w_prev: Mapping[str, float] | None,
    lam: float,
    full_investment_tol: float,
    stock_band: float,
    non_member_max: float,
    non_member_min: float,
    style_factor_names: tuple[str, ...],
) -> _ProblemData:
    n = len(instruments)
    raw_bench = _map_to_vec(bench_weights, instruments)
    total_bench = float(raw_bench.sum())
    bench_norm = raw_bench / total_bench if total_bench > 0 else np.zeros(n)
    band_member = bench_norm > 0.0
    is_member = (raw_bench > 0.0).astype(np.float64)

    upper = np.where(
        band_member, np.minimum(bench_norm + stock_band, 1.0), non_member_max
    )
    lower = np.where(
        band_member, np.maximum(bench_norm - stock_band, 0.0), non_member_min
    )

    # 行业：候选行业全集，基准行业权重在候选行业内重新归一。
    cand_codes = [industry.get(inst) or CAND_UNKNOWN_INDUSTRY for inst in instruments]
    members = [
        (inst, float(w)) for inst, w in bench_weights.items() if w > 0.0
    ]
    bench_ind_raw: dict[str, float] = defaultdict(float)
    for inst, weight in members:
        code = industry.get(inst) or BENCH_UNKNOWN_INDUSTRY
        bench_ind_raw[code] += weight
    cand_unique = sorted(set(cand_codes))
    col = {code: j for j, code in enumerate(cand_unique)}
    ind_b = np.array([bench_ind_raw.get(code, 0.0) for code in cand_unique], dtype=float)
    ind_b = ind_b / ind_b.sum() if ind_b.sum() > 0 else np.zeros_like(ind_b)
    ind_a = np.zeros((n, len(cand_unique)), dtype=float)
    for i, code in enumerate(cand_codes):
        ind_a[i, col[code]] = 1.0

    # 市值：基准 = Σ 基准权重 × 成分等效市值；std 取加权量 std(同原型)。
    constituents = [inst for inst, _ in members]
    bench_w_arr = np.array([weight for _, weight in members], dtype=float)
    bench_mv_arr = np.array(
        [float(float_mv.get(i, 0.0)) for i in constituents], dtype=float
    )
    mv_center = float((bench_w_arr * bench_mv_arr).sum())
    mv_scale = (
        float(np.std(bench_w_arr * bench_mv_arr, ddof=1))
        if len(constituents) > 1
        else 0.0
    )
    mv_a = _map_to_vec(float_mv, instruments)

    # 风格：基准 = Σ 基准权重 × 因子；std 取成分内因子值 std(同原型)。
    style_a: dict[str, np.ndarray] = {}
    style_center: dict[str, float] = {}
    style_scale: dict[str, float] = {}
    for factor in style_factor_names:
        style_a[factor] = np.array(
            [float(style.get(inst, {}).get(factor, 0.0)) for inst in instruments],
            dtype=float,
        )
        bench_f = np.array(
            [float(style.get(i, {}).get(factor, 0.0)) for i in constituents], dtype=float
        )
        style_center[factor] = float((bench_w_arr * bench_f).sum())
        style_scale[factor] = (
            float(np.std(bench_f, ddof=1)) if len(constituents) > 1 else 0.0
        )

    if w_prev is None:
        prev = np.zeros(n)
        prev_valid = False
    else:
        outside = [inst for inst in w_prev if inst not in set(instruments)]
        if outside:
            raise ValueError(f"w_prev 含候选集之外的证券: {outside}")
        prev = _map_to_vec(w_prev, instruments)
        if not np.isfinite(prev).all():
            raise ValueError("w_prev 含 NaN / Inf")
        if (prev < -ZERO_TOL_DEFAULT).any():
            raise ValueError("w_prev 含负权重，第一版仅支持多头")
        prev_valid = bool(not np.allclose(prev, 0.0, atol=ZERO_TOL_DEFAULT))

    return _ProblemData(
        instruments=instruments,
        n=n,
        alpha=alpha,
        covariance=covariance,
        lam=lam,
        full_investment_tol=full_investment_tol,
        lower=lower,
        upper=upper,
        bench_norm=bench_norm,
        ind_a=ind_a,
        ind_b=ind_b,
        mv_a=mv_a,
        mv_center=mv_center,
        mv_scale=mv_scale,
        is_member=is_member,
        style_a=style_a,
        style_center=style_center,
        style_scale=style_scale,
        prev=prev,
        prev_valid=prev_valid,
        style_factor_names=style_factor_names,
    )


def _build_objective(w: cp.Variable, data: _ProblemData) -> cp.Minimize:
    if data.lam > 0.0:
        return cp.Minimize(
            -data.alpha @ w
            + data.lam * cp.quad_form(w, cp.psd_wrap(data.covariance))
        )
    return cp.Minimize(-data.alpha @ w)


def _build_constraints(
    w: cp.Variable, data: _ProblemData, thresholds: Mapping[str, float]
) -> list[cp.Constraint]:
    tol = data.full_investment_tol
    constraints: list[cp.Constraint] = [
        cp.sum(w) >= 1.0 - tol,
        cp.sum(w) <= 1.0 + tol,
        w >= data.lower,
        w <= data.upper,
    ]

    ind_ratio = thresholds["industry"]
    constraints.append(data.ind_a.T @ w <= data.ind_b * (1.0 + ind_ratio))
    constraints.append(data.ind_a.T @ w >= data.ind_b * (1.0 - ind_ratio))

    mv_ratio = thresholds["market_value"]
    constraints.append(data.mv_a @ w <= data.mv_center + mv_ratio * data.mv_scale)
    constraints.append(data.mv_a @ w >= data.mv_center - mv_ratio * data.mv_scale)

    constraints.append(data.is_member @ w >= thresholds["cover_rate"])
    if data.prev_valid:
        constraints.append(cp.norm1(w - data.prev) <= thresholds["turnover"])

    style_ratio = thresholds["style"]
    for factor in data.style_factor_names:
        center = data.style_center[factor]
        scale = data.style_scale[factor]
        constraints.append(data.style_a[factor] @ w <= center + style_ratio * scale)
        constraints.append(data.style_a[factor] @ w >= center - style_ratio * scale)

    return constraints


def _finalize(raw: np.ndarray, zero_tol: float) -> np.ndarray:
    """约零剔除 + 归一（照原型：先置零再按和归一，和为 0 时全零）。"""
    full = np.where(raw < zero_tol, 0.0, raw)
    total = float(full.sum())
    if total <= 0.0:
        return np.zeros_like(full)
    return full / total


def _normalize(weights: Mapping[str, float]) -> dict[str, float]:
    """把权重归一到和为 1；总和非正时返回空字典。"""
    total = float(sum(weights.values()))
    if not math.isfinite(total) or total <= 0.0:
        return {}
    return {inst: float(w) / total for inst, w in weights.items() if w > 0.0}


def _thresholds_dict(thresholds: Mapping[str, Any]) -> dict[str, Any]:
    """阈值字典落日志：数值取 float，标记位（如 :data:`FINAL_TIER_FLAG`）原样保留。"""
    out: dict[str, Any] = {}
    for key, value in thresholds.items():
        out[key] = value if isinstance(value, bool) else float(value)
    return out


def _make_result(
    *,
    date: Date,
    weights: dict[str, float],
    status: str,
    relax_rounds: int,
    thresholds: Mapping[str, Any],
    exposures: dict[str, Any],
    turnover: float,
    objective: float | None,
) -> EnhancedResult:
    thresholds_dict = _thresholds_dict(thresholds)
    log = {
        "date": date,
        "status": status,
        "relax_rounds": int(relax_rounds),
        "turnover": float(turnover),
        "thresholds": thresholds_dict,
        "exposures": exposures,
    }
    return EnhancedResult(
        weights=weights,
        status=status,
        relax_rounds=int(relax_rounds),
        thresholds=thresholds_dict,
        exposures=exposures,
        turnover=float(turnover),
        objective=objective,
        log=log,
    )


__all__ = [
    "BENCH_UNKNOWN_INDUSTRY",
    "CAND_UNKNOWN_INDUSTRY",
    "COVER_RATE_MIN_DEFAULT",
    "EnhancedDayInput",
    "EnhancedOptimizer",
    "EnhancedResult",
    "FINAL_TIER_FLAG",
    "FULL_INVESTMENT_TOL",
    "INDUSTRY_EXPOSURE_DEFAULT",
    "LAMBDA_DEFAULT",
    "LOG_SCHEMA",
    "MARKET_VALUE_EXPOSURE_DEFAULT",
    "MAX_RELAX_ROUNDS_DEFAULT",
    "NON_MEMBER_MAX_DEFAULT",
    "NON_MEMBER_MIN_DEFAULT",
    "REBALANCE_STRIDE",
    "STATUS_DRIFT",
    "STATUS_HELD",
    "STATUS_OPTIMAL",
    "STATUS_RELAXED",
    "STOCK_BAND_DEFAULT",
    "STYLE_EXPOSURE_DEFAULT",
    "TURNOVER_FREE_MAX",
    "TURNOVER_MAX_DEFAULT",
    "compute_exposures",
    "drift_weights",
    "logs_to_frame",
]
