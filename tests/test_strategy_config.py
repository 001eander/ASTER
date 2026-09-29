"""``quant.daily.strategy`` 单元测试（issue #70）。

覆盖策略配置的合法 / 非法样例、优化器构造分派、调仓日判定与文件读写。全部离线，
不碰真实 ``data/``。
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from quant.daily.strategy import (
    STRATEGY_INDEX_ENHANCED,
    STRATEGY_STOCK_SELECTION,
    StrategyConfig,
    StrategyConfigError,
    build_enhanced_optimizer,
    build_optimizer,
    build_stock_optimizer,
    is_rebalance_day,
    load_strategy_config,
    parse_strategy_config,
)
from quant.portfolio.enhanced import EnhancedOptimizer
from quant.portfolio.optimizer import PortfolioOptimizer


# ---------------------------------------------------------------------------
# 合法样例
# ---------------------------------------------------------------------------


def test_minimal_stock_selection_uses_defaults() -> None:
    config = parse_strategy_config({"strategy": STRATEGY_STOCK_SELECTION})
    assert config.strategy == STRATEGY_STOCK_SELECTION
    assert config.universe is None
    assert config.benchmark is None
    assert config.top_k == 50
    assert config.rebalance_freq == "D"
    assert config.optimize == {}
    assert not config.is_index_enhanced


def test_full_index_enhanced_config() -> None:
    raw = {
        "strategy": STRATEGY_INDEX_ENHANCED,
        "universe": "zz1000",
        "benchmark": "000852",
        "top_k": 80,
        "rebalance_freq": "M",
        "optimize": {"stock_band": 0.005, "cover_rate_min": 0.5, "turnover_max": 0.2},
    }
    config = parse_strategy_config(raw)
    assert config.is_index_enhanced
    assert config.universe == "zz1000"
    assert config.benchmark == "000852"
    assert config.top_k == 80
    assert config.rebalance_freq == "M"
    assert config.to_dict() == raw


def test_from_dict_matches_parse() -> None:
    raw = {"strategy": STRATEGY_STOCK_SELECTION, "optimize": {"w_max": 0.03}}
    assert StrategyConfig.from_dict(raw) == parse_strategy_config(raw)


# ---------------------------------------------------------------------------
# 非法样例
# ---------------------------------------------------------------------------


def test_unknown_top_level_field_raises() -> None:
    with pytest.raises(StrategyConfigError, match="未知字段"):
        parse_strategy_config({"strategy": STRATEGY_STOCK_SELECTION, "foo": 1})


def test_missing_strategy_raises() -> None:
    with pytest.raises(StrategyConfigError, match="缺少必填"):
        parse_strategy_config({"universe": "zz1000"})


def test_invalid_strategy_enum_raises() -> None:
    with pytest.raises(StrategyConfigError, match="strategy"):
        parse_strategy_config({"strategy": "market_neutral"})


def test_invalid_rebalance_freq_raises() -> None:
    with pytest.raises(StrategyConfigError, match="rebalance_freq"):
        parse_strategy_config(
            {"strategy": STRATEGY_STOCK_SELECTION, "rebalance_freq": "Q"}
        )


@pytest.mark.parametrize("bad", [0, -3, True, 1.5, "5"])
def test_invalid_top_k_raises(bad: object) -> None:
    with pytest.raises(StrategyConfigError, match="top_k"):
        parse_strategy_config({"strategy": STRATEGY_STOCK_SELECTION, "top_k": bad})


def test_unknown_optimize_param_for_strategy_raises() -> None:
    with pytest.raises(StrategyConfigError, match="optimize"):
        parse_strategy_config(
            {
                "strategy": STRATEGY_STOCK_SELECTION,
                "optimize": {"stock_band": 0.005},  # 这是指增参数
            }
        )


def test_index_enhanced_requires_benchmark() -> None:
    with pytest.raises(StrategyConfigError, match="benchmark"):
        parse_strategy_config({"strategy": STRATEGY_INDEX_ENHANCED})


def test_index_enhanced_rejects_frequency_in_optimize() -> None:
    with pytest.raises(StrategyConfigError, match="frequency"):
        parse_strategy_config(
            {
                "strategy": STRATEGY_INDEX_ENHANCED,
                "benchmark": "000852",
                "optimize": {"frequency": "W"},
            }
        )


def test_invalid_param_value_triggers_optimizer_validation() -> None:
    with pytest.raises(ValueError, match="cover_rate_min"):
        parse_strategy_config(
            {
                "strategy": STRATEGY_INDEX_ENHANCED,
                "benchmark": "000852",
                "optimize": {"cover_rate_min": 2.0},
            }
        )


# ---------------------------------------------------------------------------
# 优化器构造
# ---------------------------------------------------------------------------


def test_build_stock_optimizer_maps_params() -> None:
    config = parse_strategy_config(
        {"strategy": STRATEGY_STOCK_SELECTION, "optimize": {"w_max": 0.03, "lam": 2.0}}
    )
    optimizer = build_stock_optimizer(config)
    assert isinstance(optimizer, PortfolioOptimizer)
    assert optimizer.w_max == pytest.approx(0.03)
    assert optimizer.lam == pytest.approx(2.0)


def test_build_enhanced_optimizer_sets_frequency() -> None:
    config = parse_strategy_config(
        {
            "strategy": STRATEGY_INDEX_ENHANCED,
            "benchmark": "000852",
            "rebalance_freq": "W",
            "optimize": {"stock_band": 0.004, "turnover_max": 0.15},
        }
    )
    optimizer = build_enhanced_optimizer(config)
    assert isinstance(optimizer, EnhancedOptimizer)
    assert optimizer.stock_band == pytest.approx(0.004)
    assert optimizer.turnover_max == pytest.approx(0.15)
    assert optimizer.frequency == "W"


def test_build_optimizer_dispatches_by_strategy() -> None:
    stock = build_optimizer(parse_strategy_config({"strategy": STRATEGY_STOCK_SELECTION}))
    assert isinstance(stock, PortfolioOptimizer)
    enhanced = build_optimizer(
        parse_strategy_config(
            {"strategy": STRATEGY_INDEX_ENHANCED, "benchmark": "000852"}
        )
    )
    assert isinstance(enhanced, EnhancedOptimizer)


# ---------------------------------------------------------------------------
# 文件读写
# ---------------------------------------------------------------------------


def test_load_strategy_config_roundtrip(tmp_path: Path) -> None:
    raw = {
        "strategy": STRATEGY_INDEX_ENHANCED,
        "universe": "zz1000",
        "benchmark": "000852",
        "top_k": 60,
        "rebalance_freq": "W",
        "optimize": {"cover_rate_min": 0.6},
    }
    path = tmp_path / "strategy.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    config = load_strategy_config(path)
    assert config.to_dict() == raw


def test_load_strategy_config_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(StrategyConfigError, match="不存在"):
        load_strategy_config(tmp_path / "nope.json")


# ---------------------------------------------------------------------------
# 调仓日判定
# ---------------------------------------------------------------------------


def test_is_rebalance_day_daily_always_true() -> None:
    days = [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 12)]
    assert all(is_rebalance_day(days, day, "D") for day in days)


def test_is_rebalance_day_weekly_first_open_of_week() -> None:
    days = [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7), date(2026, 1, 12)]
    assert is_rebalance_day(days, date(2026, 1, 5), "W")
    assert not is_rebalance_day(days, date(2026, 1, 6), "W")
    assert is_rebalance_day(days, date(2026, 1, 12), "W")


def test_is_rebalance_day_monthly_first_open_of_month() -> None:
    days = [date(2026, 1, 5), date(2026, 1, 30), date(2026, 2, 2)]
    assert is_rebalance_day(days, date(2026, 1, 5), "M")
    assert not is_rebalance_day(days, date(2026, 1, 30), "M")
    assert is_rebalance_day(days, date(2026, 2, 2), "M")
