"""``quant.portfolio.enhanced`` 单元测试。

覆盖：每条约束的生效 / 不生效、放松阶梯顺序、换手驱动的放松路径、失败漂移、
调仓频率与区间漂移、风险项、输入校验，以及与 pandas 原型的对拍（读
``tests/data/enhanced_opt_fixture``，测试路径不导入 pandas / 原型）。
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from quant.portfolio.enhanced import (
    FINAL_TIER_FLAG,
    MAX_RELAX_ROUNDS_DEFAULT,
    STATUS_DRIFT,
    STATUS_HELD,
    STATUS_OPTIMAL,
    STATUS_RELAXED,
    TURNOVER_FREE_MAX,
    EnhancedDayInput,
    EnhancedOptimizer,
    drift_weights,
    logs_to_frame,
)
from quant.portfolio.risk import ledoit_wolf_covariance
from quant.portfolio.style import STYLE_FACTOR_NAMES

FIXTURE_DIR = Path(__file__).parent / "data" / "enhanced_opt_fixture"
FACTORS = STYLE_FACTOR_NAMES


# ---------------------------------------------------------------------------
# 测试样本
# ---------------------------------------------------------------------------


def _universe(
    *,
    seed: int = 0,
    n: int = 20,
    n_members: int = 14,
    n_industries: int = 3,
    n_returns: int = 80,
    alpha: dict[str, float] | None = None,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    instruments = [f"{600000 + i:06d}.SH" for i in range(n)]
    bench_raw = np.zeros(n)
    bench_raw[:n_members] = rng.dirichlet(np.ones(n_members))
    bench = {
        inst: float(bench_raw[i])
        for i, inst in enumerate(instruments)
        if bench_raw[i] > 0
    }
    industry = {
        inst: f"I{rng.integers(0, n_industries)}" for inst in instruments
    }
    scores = (
        alpha
        if alpha is not None
        else {inst: float(rng.standard_normal()) for inst in instruments}
    )
    float_mv = {inst: float(np.exp(rng.normal(24.0, 1.0))) for inst in instruments}
    style = {
        inst: {f: float(rng.standard_normal()) for f in FACTORS}
        for inst in instruments
    }
    block = rng.standard_normal((n, n))
    covariance = (block @ block.T) / n * 1.0e-4
    returns = rng.standard_normal((n_returns, n)) * 0.01
    return {
        "instruments": instruments,
        "bench": bench,
        "industry": industry,
        "alpha": scores,
        "float_mv": float_mv,
        "style": style,
        "covariance": returns,
        "risk_covariance": covariance,
        "bench_raw": bench_raw,
    }


def _optimize(
    u: dict[str, Any],
    *,
    opt: EnhancedOptimizer | None = None,
    w_prev: dict[str, float] | None = None,
    covariance: np.ndarray | None = None,
    day: date | None = None,
):
    optimizer = opt or EnhancedOptimizer()
    return optimizer.optimize_day(
        date=day or date(2025, 1, 2),
        instruments=u["instruments"],
        alpha=u["alpha"],
        bench_weights=u["bench"],
        industry=u["industry"],
        float_mv=u["float_mv"],
        style=u["style"],
        covariance=u["covariance"] if covariance is None else covariance,
        w_prev=w_prev,
    )


def _bench_norm(u: dict[str, Any]) -> np.ndarray:
    raw = np.array(
        [u["bench"].get(inst, 0.0) for inst in u["instruments"]], dtype=float
    )
    return raw / raw.sum()


def _far_prev(u: dict[str, Any]) -> dict[str, float]:
    """与基准有一定距离的上期权重，用于触发换手约束。"""
    norm = _bench_norm(u)
    count = len(u["instruments"])
    return {
        inst: 0.7 * float(norm[i]) + 0.3 / count
        for i, inst in enumerate(u["instruments"])
    }


# ---------------------------------------------------------------------------
# 基础约束
# ---------------------------------------------------------------------------


def test_full_investment_no_short_and_stock_band() -> None:
    u = _universe(seed=1)
    result = _optimize(u)
    assert result.status == STATUS_OPTIMAL
    assert sum(result.weights.values()) == pytest.approx(1.0, abs=1e-4)

    norm = _bench_norm(u)
    max_non_member = 0.0
    for i, inst in enumerate(u["instruments"]):
        weight = result.weights.get(inst, 0.0)
        assert weight >= 0.0
        if norm[i] > 0:
            assert weight >= norm[i] - 0.005 - 1e-4
            assert weight <= norm[i] + 0.005 + 1e-4
        else:
            assert weight <= 0.005 + 1e-4
            max_non_member = max(max_non_member, weight)
    assert max_non_member <= 0.005 + 1e-4


def test_industry_constraint_enforced() -> None:
    u = _universe(seed=2)
    tight = _optimize(u, opt=EnhancedOptimizer(industry_exposure=0.001))
    for code, exp in tight.exposures["industry"].items():
        if exp["bench"] > 0:
            assert abs(exp["portfolio"] - exp["bench"]) / exp["bench"] < 2e-3

    loose = _optimize(u, opt=EnhancedOptimizer(industry_exposure=1.0))
    max_dev = max(
        abs(exp["portfolio"] - exp["bench"])
        for exp in loose.exposures["industry"].values()
    )
    assert max_dev > 1e-3


def test_market_value_constraint_enforced() -> None:
    u = _universe(seed=3)
    tight = _optimize(u, opt=EnhancedOptimizer(market_value_exposure=0.001))
    assert abs(tight.exposures["market_value"]["std"]) < 3e-3

    loose = _optimize(u, opt=EnhancedOptimizer(market_value_exposure=100.0))
    assert abs(loose.exposures["market_value"]["std"]) > 2e-3


def test_style_constraint_enforced() -> None:
    u = _universe(seed=4)
    result = _optimize(u, opt=EnhancedOptimizer(style_exposure=0.001))
    for factor, exp in result.exposures["style"].items():
        assert abs(exp["std"]) < 2e-3, factor

    loose = _optimize(u, opt=EnhancedOptimizer(style_exposure=100.0))
    max_std = max(abs(exp["std"]) for exp in loose.exposures["style"].values())
    assert max_std > 1e-3


def test_cover_rate_constraint_binds() -> None:
    u = _universe(seed=5)
    tight = _optimize(u, opt=EnhancedOptimizer(cover_rate_min=1.0))
    assert tight.exposures["cover_rate"] == pytest.approx(1.0, abs=1e-3)

    # 给非成分股极高的 alpha，宽松覆盖度下它会被买满非成分上限。
    high = u["instruments"][-1]
    alpha = dict(u["alpha"])
    alpha[high] = 1e6
    loose_inputs = dict(u, alpha=alpha)
    loose = _optimize(loose_inputs, opt=EnhancedOptimizer(cover_rate_min=0.0))
    assert loose.weights.get(high, 0.0) > 0.004
    assert loose.exposures["cover_rate"] < 0.99


def test_turnover_constraint_binds_and_relaxes() -> None:
    u = _universe(seed=6)
    far = _far_prev(u)

    loose = _optimize(
        u, opt=EnhancedOptimizer(turnover_max=2.0), w_prev=far
    )
    assert loose.status == STATUS_OPTIMAL
    assert loose.turnover > 0.1

    bound = _optimize(u, opt=EnhancedOptimizer(turnover_max=0.05), w_prev=far)
    assert bound.status == STATUS_RELAXED
    assert bound.relax_rounds > 0
    assert bound.thresholds["turnover"] > 0.05
    assert bound.turnover <= bound.thresholds["turnover"] + 1e-3


# ---------------------------------------------------------------------------
# 放松阶梯
# ---------------------------------------------------------------------------


def test_relaxation_ladder_order() -> None:
    opt = EnhancedOptimizer(
        industry_relax_step=0.001,
        market_value_relax_step=0.002,
        style_relax_step=0.003,
        turnover_relax_step=0.01,
    )
    t0 = opt._initial_thresholds()
    t1 = opt._relax(t0, 0)
    assert t1["style"] == pytest.approx(t0["style"] + 0.003)
    assert t1["industry"] == t0["industry"]
    assert t1["market_value"] == t0["market_value"]
    assert t1["turnover"] == t0["turnover"]

    t2 = opt._relax(t1, 1)
    assert t2["industry"] == pytest.approx(t1["industry"] + 0.001)
    assert t2["market_value"] == pytest.approx(t1["market_value"] + 0.002)
    assert t2["style"] == t1["style"]
    assert t2["turnover"] == t1["turnover"]

    t3 = opt._relax(t2, 2)
    assert t3["turnover"] == pytest.approx(t2["turnover"] + 0.01)
    assert t3["style"] == t2["style"]

    t4 = opt._relax(t3, 3)
    assert t4["style"] == pytest.approx(t3["style"] + 0.003)
    assert t4["turnover"] == t3["turnover"]


def test_relaxation_relaxes_style_before_turnover() -> None:
    u = _universe(seed=7)
    far = _far_prev(u)
    result = _optimize(
        u, opt=EnhancedOptimizer(turnover_max=0.05, style_exposure=0.005), w_prev=far
    )
    assert result.status == STATUS_RELAXED
    assert result.thresholds["style"] > 0.005
    assert result.thresholds["turnover"] > 0.05


# ---------------------------------------------------------------------------
# 失败与漂移
# ---------------------------------------------------------------------------


def test_failure_holds_previous_weights() -> None:
    """求解器不可用（终局台阶也无法求解）时保持上期权重，relax_rounds 记 59。"""
    u = _universe(seed=8)
    far = {inst: 1.0 / len(u["instruments"]) for inst in u["instruments"]}
    result = _optimize(
        u,
        opt=EnhancedOptimizer(turnover_max=0.0, solver="__no_such_solver__"),
        w_prev=far,
    )
    assert result.status == STATUS_HELD
    assert result.relax_rounds == 59
    assert result.weights == pytest.approx(far, abs=1e-12)


# ---------------------------------------------------------------------------
# 终局台阶（issue #71）
# ---------------------------------------------------------------------------


def test_final_tier_recovers_when_turnover_only_blocks() -> None:
    """唯一阻碍是换手约束时，终局台阶取消换手后产出解（status=relaxed）。"""
    u = _universe(seed=8)
    far = {inst: 1.0 / len(u["instruments"]) for inst in u["instruments"]}
    result = _optimize(u, opt=EnhancedOptimizer(turnover_max=0.0), w_prev=far)

    assert result.status == STATUS_RELAXED
    assert result.relax_rounds == MAX_RELAX_ROUNDS_DEFAULT
    assert result.thresholds[FINAL_TIER_FLAG] is True
    assert result.thresholds["turnover"] == pytest.approx(TURNOVER_FREE_MAX)
    # 终局台阶回到初始阈值：个股带仍守住（否则等于放弃了约束）。
    norm = _bench_norm(u)
    for i, inst in enumerate(u["instruments"]):
        weight = result.weights.get(inst, 0.0)
        if norm[i] > 0:
            assert weight <= norm[i] + 0.005 + 1e-4
            assert weight >= norm[i] - 0.005 - 1e-4
        else:
            assert weight <= 0.005 + 1e-4


def test_final_tier_not_used_when_ladder_solves() -> None:
    """常规放松能解出时不得触发终局台阶，阈值里没有标记位。"""
    u = _universe(seed=1)
    result = _optimize(u)
    assert result.status == STATUS_OPTIMAL
    assert FINAL_TIER_FLAG not in result.thresholds
    assert result.thresholds["turnover"] == pytest.approx(0.2)


def test_final_tier_failure_keeps_held() -> None:
    """终局台阶也解不出（求解器不可用）时仍是 held。"""
    u = _universe(seed=9)
    far = {inst: 1.0 / len(u["instruments"]) for inst in u["instruments"]}
    result = _optimize(
        u,
        opt=EnhancedOptimizer(solver="__no_such_solver__"),
        w_prev=far,
    )
    assert result.status == STATUS_HELD
    assert result.relax_rounds == 59
    assert FINAL_TIER_FLAG not in result.thresholds


def test_held_exposures_are_normalized() -> None:
    """held 分支暴露按归一权重算：权重和小于 1 时覆盖度不随现金比例缩水。"""
    u = _universe(seed=15)
    bench_norm = _bench_norm(u)
    # 只有一半权重落在候选集里的「漂移后持仓」（模拟现金 / 取整残差）。
    half = {inst: 0.5 * float(bench_norm[i]) for i, inst in enumerate(u["instruments"])}
    result = _optimize(
        u,
        opt=EnhancedOptimizer(turnover_max=0.0, solver="__no_such_solver__"),
        w_prev=half,
    )
    assert result.status == STATUS_HELD
    assert sum(result.weights.values()) == pytest.approx(1.0, abs=1e-12)
    assert result.exposures["cover_rate"] == pytest.approx(1.0, abs=1e-6)


def test_drift_weights_normalizes_and_handles_missing() -> None:
    base = {"a": 0.5, "b": 0.5}
    drifted = drift_weights(base, {"a": 0.10})
    total = drifted["a"] + drifted["b"]
    assert total == pytest.approx(1.0)
    assert drifted["a"] == pytest.approx(0.55 / 1.05)
    assert drifted["b"] == pytest.approx(0.5 / 1.05)
    # 缺失收益按 0 处理；全为 0 收益时权重不变。
    assert drift_weights(base, {}) == pytest.approx(base)
    assert drift_weights({}, {"a": 1.0}) == {}


def test_run_drifts_on_failure() -> None:
    u = _universe(seed=9)
    far = {inst: 1.0 / len(u["instruments"]) for inst in u["instruments"]}
    returns = {inst: (0.02 if i % 2 == 0 else -0.01) for i, inst in enumerate(u["instruments"])}
    day = EnhancedDayInput(
        date=date(2025, 1, 2),
        instruments=u["instruments"],
        alpha=u["alpha"],
        bench_weights=u["bench"],
        industry=u["industry"],
        float_mv=u["float_mv"],
        style=u["style"],
        covariance=u["covariance"],
        interval_returns=returns,
    )
    # 用上一个日的结果当作已持仓，再以零换手预算触发失败（求解器不可用模拟终局台阶也失败）。
    optimizer = EnhancedOptimizer(turnover_max=0.0, solver="__no_such_solver__")
    first = optimizer.run(
        [
            EnhancedDayInput(**{**day.__dict__, "interval_returns": {}}),
        ]
    )
    assert first[0].status == STATUS_HELD
    held = drift_weights(far, returns)
    result = _optimize(u, opt=optimizer, w_prev=held)
    assert result.status == STATUS_HELD
    assert result.weights == pytest.approx(held, abs=1e-12)


# ---------------------------------------------------------------------------
# 调仓频率
# ---------------------------------------------------------------------------


def _days(u: dict[str, Any], count: int, intervals: list[dict[str, float]] | None = None):
    start = date(2025, 1, 6)
    out = []
    for i in range(count):
        out.append(
            EnhancedDayInput(
                date=start + timedelta(days=i),
                instruments=u["instruments"],
                alpha=u["alpha"],
                bench_weights=u["bench"],
                industry=u["industry"],
                float_mv=u["float_mv"],
                style=u["style"],
                covariance=u["covariance"],
                interval_returns=(intervals[i] if intervals else {inst: 0.01 for inst in u["instruments"]}),
            )
        )
    return out


def test_frequency_daily_all_rebalance() -> None:
    u = _universe(seed=10)
    results = EnhancedOptimizer(frequency="D", turnover_max=2.0).run(_days(u, 4))
    assert all(r.status != STATUS_DRIFT for r in results)


def test_frequency_weekly_drifts_between() -> None:
    u = _universe(seed=11)
    results = EnhancedOptimizer(frequency="W", turnover_max=2.0).run(_days(u, 6))
    statuses = [r.status for r in results]
    assert statuses[0] != STATUS_DRIFT
    assert statuses[1:5] == [STATUS_DRIFT] * 4
    assert statuses[5] != STATUS_DRIFT

    # 漂移日权重等于上日权重按区间收益漂移的结果。
    expected = drift_weights(results[0].weights, {inst: 0.01 for inst in u["instruments"]})
    assert results[1].weights == pytest.approx(expected, abs=1e-12)


def test_frequency_monthly_only_first() -> None:
    u = _universe(seed=12)
    results = EnhancedOptimizer(frequency="M", turnover_max=2.0).run(_days(u, 6))
    assert results[0].status != STATUS_DRIFT
    assert all(r.status == STATUS_DRIFT for r in results[1:])


# ---------------------------------------------------------------------------
# 风险项
# ---------------------------------------------------------------------------


def test_lambda_risk_term_reduces_variance() -> None:
    u = _universe(seed=13)
    n = len(u["instruments"])
    rng = np.random.default_rng(99)
    # 结构化收益（异质 beta），保证 LW 协方差非各向同性，风险项可观测。
    factor = rng.standard_normal((300, 1)) * 0.02
    returns = np.linspace(0.2, 3.0, n) * factor + rng.standard_normal((300, n)) * 0.01
    cov = ledoit_wolf_covariance(returns)
    alpha = {inst: 0.0 for inst in u["instruments"]}

    def variance(weights: dict[str, float]) -> float:
        vec = np.array([weights.get(i, 0.0) for i in u["instruments"]])
        return float(vec @ cov @ vec)

    low = _optimize(dict(u, alpha=alpha), opt=EnhancedOptimizer(lam=0.0), covariance=returns)
    high = _optimize(dict(u, alpha=alpha), opt=EnhancedOptimizer(lam=10.0), covariance=returns)
    assert variance(high.weights) < variance(low.weights) - 1e-6


# ---------------------------------------------------------------------------
# 日志与输入校验
# ---------------------------------------------------------------------------


def test_logs_to_frame() -> None:
    u = _universe(seed=14)
    results = EnhancedOptimizer(turnover_max=2.0).run(_days(u, 3))
    frame = logs_to_frame(results)
    assert frame.height == 3
    assert frame.columns == [
        "date",
        "status",
        "relax_rounds",
        "turnover",
        "thresholds",
        "exposures",
    ]
    assert logs_to_frame([]).height == 0


def test_exposure_keys() -> None:
    u = _universe(seed=15)
    result = _optimize(u)
    exp = result.exposures
    assert set(exp) == {"cover_rate", "turnover", "market_value", "industry", "style"}
    assert set(exp["style"]) == set(FACTORS)
    assert set(exp["market_value"]) == {"ratio", "std"}


def test_invalid_params_raise() -> None:
    with pytest.raises(ValueError):
        EnhancedOptimizer(lam=-1.0)
    with pytest.raises(ValueError):
        EnhancedOptimizer(stock_band=-0.001)
    with pytest.raises(ValueError):
        EnhancedOptimizer(cover_rate_min=1.5)
    with pytest.raises(ValueError):
        EnhancedOptimizer(turnover_max=-0.1)
    with pytest.raises(ValueError):
        EnhancedOptimizer(max_relax_rounds=0)
    with pytest.raises(ValueError):
        EnhancedOptimizer(frequency="Q")


def test_input_validation() -> None:
    u = _universe(seed=16)
    with pytest.raises(ValueError, match="缺少候选证券"):
        _optimize(dict(u, alpha={"600000.SH": 1.0}))
    with pytest.raises(ValueError, match="形状"):
        _optimize(u, covariance=np.zeros((10, 3)))
    with pytest.raises(ValueError, match="候选集之外"):
        _optimize(u, w_prev={"999999.SH": 0.1})
    with pytest.raises(ValueError, match="为空"):
        EnhancedOptimizer().optimize_day(
            date=date(2025, 1, 2),
            instruments=[],
            alpha={},
            bench_weights={},
            industry={},
            float_mv={},
            style={},
            covariance=np.zeros((5, 0)),
        )
    with pytest.raises(ValueError, match="重复"):
        EnhancedOptimizer().optimize_day(
            date=date(2025, 1, 2),
            instruments=["600000.SH", "600000.SH"],
            alpha=[1.0, 2.0],
            bench_weights={"600000.SH": 1.0},
            industry={"600000.SH": "I0"},
            float_mv={"600000.SH": 1.0},
            style={"600000.SH": {f: 0.0 for f in FACTORS}},
            covariance=np.zeros((5, 2)),
        )


# ---------------------------------------------------------------------------
# 与原型对拍
# ---------------------------------------------------------------------------


def _load_fixture() -> tuple[dict[str, Any], dict[str, Any]]:
    inputs = json.loads((FIXTURE_DIR / "inputs.json").read_text(encoding="utf-8"))
    expected = json.loads((FIXTURE_DIR / "expected.json").read_text(encoding="utf-8"))
    return inputs, expected


def _fixture_kwargs(inputs: dict[str, Any], sample: dict[str, Any]):
    instruments = inputs["instruments"]
    n = len(instruments)
    bench = {
        instruments[i]: inputs["bench_weights"][i]
        for i in range(n)
        if inputs["members"][i]
    }
    return {
        "instruments": instruments,
        "alpha": {instruments[i]: inputs["alpha"][i] for i in range(n)},
        "bench_weights": bench,
        "industry": {instruments[i]: inputs["industry"][i] for i in range(n)},
        "float_mv": {instruments[i]: inputs["float_mv"][i] for i in range(n)},
        "style": {
            instruments[i]: {f: inputs["style"][f][i] for f in FACTORS}
            for i in range(n)
        },
        "covariance": np.array(inputs["covariance"], dtype=float),
        "w_prev": {
            instruments[i]: inputs[sample.get("prev", "w_prev")][i] for i in range(n)
        },
    }


@pytest.mark.parametrize(
    "sample_name", ["optimal", "relaxed", "tight_cover", "infeasible"]
)
def test_parity_with_prototype(sample_name: str) -> None:
    """同参数下与原型逐日解偏差 < 1e-4，放松轮次与阈值一致。

    ``infeasible`` 样本的**唯一**不可行源是换手约束（原型在 59 轮放松后 held）。issue #71
    加了终局台阶：取消换手约束再解一次，该样本因此变为 ``relaxed``（``relax_rounds`` 记
    ``max_relax_rounds``）。60 轮内的放松语义与阈值推进不受影响，其余三个样本逐位一致。
    """
    if not (FIXTURE_DIR / "expected.json").exists():
        pytest.skip("缺少对拍 fixture")
    inputs, expected = _load_fixture()
    sample = expected[sample_name]
    params = sample["params"]
    optimizer = EnhancedOptimizer(
        lam=0.0,
        industry_exposure=params.get("initial_industry_exposure", 0.005),
        market_value_exposure=params.get("initial_market_value_exposure", 0.005),
        style_exposure=params.get("initial_style_factors_exposure", 0.005),
        cover_rate_min=params.get("initial_cover_rate_lower_bound", 0.5),
        turnover_max=params.get("initial_turn_over_rate_upper_bound", 0.2),
    )
    result = optimizer.optimize_day(
        date=date(2025, 1, 2), **_fixture_kwargs(inputs, sample)
    )

    if sample_name == "infeasible":
        assert sample["relax_rounds"] == 59
        assert result.relax_rounds == MAX_RELAX_ROUNDS_DEFAULT
        assert result.status == STATUS_RELAXED
        assert result.thresholds[FINAL_TIER_FLAG] is True
        return

    assert result.relax_rounds == sample["relax_rounds"]
    assert FINAL_TIER_FLAG not in result.thresholds
    if sample["status"] == 1:
        assert result.status in {STATUS_OPTIMAL, STATUS_RELAXED}
        ours = np.array(
            [result.weights.get(inst, 0.0) for inst in inputs["instruments"]]
        )
        expected_weights = np.array(sample["weights"])
        assert np.max(np.abs(ours - expected_weights)) < 1e-4
    else:
        assert result.status == STATUS_HELD

    for key in ("industry_exposure", "market_value_exposure", "style_factors_exposure",
                "turn_over_rate_upper_bound"):
        mapped = {
            "industry_exposure": "industry",
            "market_value_exposure": "market_value",
            "style_factors_exposure": "style",
            "turn_over_rate_upper_bound": "turnover",
        }[key]
        assert result.thresholds[mapped] == pytest.approx(
            sample["thresholds"][key], abs=1e-9
        )
