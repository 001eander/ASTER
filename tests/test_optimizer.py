"""``quant.portfolio.optimizer`` 单元测试。

覆盖：两资产解析可核对的小例子、约束满足、换手惩罚、不可行抛错、
输入对齐校验、约零权重剔除。
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from quant.portfolio.optimizer import (
    InfeasibleError,
    OptimizeResult,
    PortfolioOptimizer,
)


def _returns(n: int, t: int = 250, *, seed: int = 0, corr: float = 0.0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((t, n))
    if corr:
        common = rng.standard_normal((t, 1))
        z = np.sqrt(corr) * common + np.sqrt(1 - corr) * z
    z = z * 0.01
    return pl.DataFrame(z, schema=[f"{600000 + i:06d}.SH" for i in range(n)])


INSTR2 = ["600000.SH", "600001.SH"]
INSTR5 = [f"{600000 + i:06d}.SH" for i in range(5)]


# ---------------------------------------------------------------------------
# 解析可核对
# ---------------------------------------------------------------------------


def test_two_asset_analytic_at_cap() -> None:
    """λ = κ = 0 时是线性目标，最优解把上限打满：w = (0.9×0.96, 1−…) 。"""
    optimizer = PortfolioOptimizer(lam=0.0, kappa=0.0, w_max=0.9, max_turnover=1.0)
    result = optimizer.optimize({"600000.SH": 1.0, "600001.SH": 0.0}, INSTR2, _returns(2))
    upper = 0.9 * 0.96
    assert result.weights["600000.SH"] == pytest.approx(upper, abs=1e-6)
    assert result.weights["600001.SH"] == pytest.approx(1.0 - upper, abs=1e-6)
    assert result.objective == pytest.approx(-upper, abs=1e-7)
    assert result.turnover == pytest.approx(1.0, abs=1e-6)
    assert result.status == "optimal"


def test_higher_alpha_gets_more_weight() -> None:
    optimizer = PortfolioOptimizer(lam=0.0, kappa=0.0, w_max=0.4, max_turnover=1.0)
    alpha = {inst: score for inst, score in zip(INSTR5, [5.0, 4.0, 3.0, 2.0, 1.0])}
    result = optimizer.optimize(alpha, INSTR5, _returns(5, seed=1))
    ordered = [result.weights.get(inst, 0.0) for inst in INSTR5]
    # 头两只都打满上限，允许求解器的浮点抖动
    for higher, lower in zip(ordered, ordered[1:]):
        assert higher >= lower - 1e-6


def test_zero_tol_prunes_tiny_positions() -> None:
    optimizer = PortfolioOptimizer(
        lam=0.0, kappa=0.0, w_max=0.4, max_turnover=1.0, zero_tol=1e-4
    )
    alpha = {inst: score for inst, score in zip(INSTR5, [5.0, 4.0, 3.0, 2.0, 1.0])}
    result = optimizer.optimize(alpha, INSTR5, _returns(5, seed=2))
    assert "600003.SH" not in result.weights
    assert "600004.SH" not in result.weights
    assert set(result.weights) == {"600000.SH", "600001.SH", "600002.SH"}


# ---------------------------------------------------------------------------
# 约束满足
# ---------------------------------------------------------------------------


def test_constraints_satisfied() -> None:
    optimizer = PortfolioOptimizer(w_max=0.4, max_turnover=0.5)
    alpha = {inst: float(i) for i, inst in enumerate(INSTR5)}
    prev = {inst: 0.2 for inst in INSTR5}
    result = optimizer.optimize(alpha, INSTR5, _returns(5, seed=3, corr=0.3), prev)

    assert sum(result.weights.values()) == pytest.approx(1.0, abs=1e-6)
    for weight in result.weights.values():
        assert weight >= 0.0
        assert weight <= optimizer.w_max + 1e-9
        assert weight <= optimizer.w_max * optimizer.w_max_buffer + 1e-6
    assert result.turnover <= optimizer.max_turnover + 1e-6
    # 结果换手与权重清单一致
    manual = sum(
        abs(result.weights.get(inst, 0.0) - prev.get(inst, 0.0)) for inst in INSTR5
    )
    assert result.turnover == pytest.approx(manual, abs=1e-6)


def test_w_max_override() -> None:
    optimizer = PortfolioOptimizer(lam=0.0, kappa=0.0, w_max=0.9, max_turnover=1.0)
    result = optimizer.optimize(
        {"600000.SH": 1.0, "600001.SH": 0.0}, INSTR2, _returns(2), w_max=0.6
    )
    assert result.weights["600000.SH"] <= 0.6 + 1e-9


def test_zero_alpha_holds_previous_weights() -> None:
    """α 全零、换手惩罚较大时，最优解停在 w_prev，换手为 0。"""
    optimizer = PortfolioOptimizer(lam=1.0, kappa=1.0, w_max=0.5, max_turnover=1.0)
    prev = {inst: 0.2 for inst in INSTR5}
    result = optimizer.optimize(
        {inst: 0.0 for inst in INSTR5}, INSTR5, _returns(5, seed=4, corr=0.3), prev
    )
    assert result.turnover < 1e-6
    for inst in INSTR5:
        assert result.weights[inst] == pytest.approx(prev[inst], abs=1e-5)


def test_turnover_penalty_reduces_turnover() -> None:
    """同样的 alpha，κ 越大换手越低。"""
    alpha = {inst: float(i) for i, inst in enumerate(INSTR5)}
    prev = {inst: 0.2 for inst in INSTR5}
    returns = _returns(5, seed=5, corr=0.3)
    low = PortfolioOptimizer(lam=1.0, kappa=0.0, w_max=0.5, max_turnover=1.0).optimize(
        alpha, INSTR5, returns, prev
    )
    high = PortfolioOptimizer(lam=1.0, kappa=0.5, w_max=0.5, max_turnover=1.0).optimize(
        alpha, INSTR5, returns, prev
    )
    assert high.turnover < low.turnover


# ---------------------------------------------------------------------------
# 不可行
# ---------------------------------------------------------------------------


def test_infeasible_capacity_raises() -> None:
    """票数 × 上限不足 1（2 × 0.3 × 0.96 < 1），不可行。"""
    optimizer = PortfolioOptimizer(w_max=0.3, max_turnover=1.0)
    with pytest.raises(InfeasibleError):
        optimizer.optimize({"600000.SH": 1.0, "600001.SH": 1.0}, INSTR2, _returns(2))


def test_infeasible_turnover_budget_raises() -> None:
    """满仓单票必须降到上限以下，所需换手超过极小的换手预算，不可行。"""
    optimizer = PortfolioOptimizer(w_max=0.9, max_turnover=0.01)
    prev = {"600000.SH": 1.0, "600001.SH": 0.0}
    with pytest.raises(InfeasibleError):
        optimizer.optimize(
            {"600000.SH": 1.0, "600001.SH": 0.0}, INSTR2, _returns(2), prev
        )


# ---------------------------------------------------------------------------
# 输入校验
# ---------------------------------------------------------------------------


def test_returns_column_mismatch_raises() -> None:
    optimizer = PortfolioOptimizer()
    with pytest.raises(ValueError, match="列与候选证券清单不一致"):
        optimizer.optimize({"600000.SH": 1.0, "600001.SH": 1.0}, INSTR2, _returns(3))


def test_alpha_missing_instrument_raises() -> None:
    optimizer = PortfolioOptimizer()
    with pytest.raises(ValueError, match="缺少候选证券"):
        optimizer.optimize({"600000.SH": 1.0}, INSTR2, _returns(2))


def test_alpha_wrong_length_raises() -> None:
    optimizer = PortfolioOptimizer()
    with pytest.raises(ValueError, match="长度"):
        optimizer.optimize([1.0, 2.0, 3.0], INSTR2, _returns(2))


def test_prev_outside_candidates_raises() -> None:
    optimizer = PortfolioOptimizer()
    with pytest.raises(ValueError, match="候选集之外"):
        optimizer.optimize(
            {"600000.SH": 1.0, "600001.SH": 0.0},
            INSTR2,
            _returns(2),
            {"999999.SH": 0.1},
        )


def test_duplicate_instruments_raises() -> None:
    optimizer = PortfolioOptimizer()
    with pytest.raises(ValueError, match="重复"):
        optimizer.optimize(
            {"600000.SH": 1.0}, ["600000.SH", "600000.SH"], _returns(2)
        )


def test_invalid_params_raise() -> None:
    with pytest.raises(ValueError):
        PortfolioOptimizer(lam=-1.0)
    with pytest.raises(ValueError):
        PortfolioOptimizer(kappa=-1.0)
    with pytest.raises(ValueError):
        PortfolioOptimizer(w_max=0.0)
    with pytest.raises(ValueError):
        PortfolioOptimizer(max_turnover=-0.1)
    with pytest.raises(ValueError):
        PortfolioOptimizer(w_max_buffer=0.0)


def test_result_type() -> None:
    optimizer = PortfolioOptimizer(lam=0.0, kappa=0.0, w_max=0.9, max_turnover=1.0)
    result = optimizer.optimize({"600000.SH": 1.0, "600001.SH": 0.0}, INSTR2, _returns(2))
    assert isinstance(result, OptimizeResult)
