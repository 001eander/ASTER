"""``quant.backtest.fee`` 单元测试：佣金 / 印花税 / 滑点口径。"""
from __future__ import annotations

import pytest

from quant.backtest.fee import (
    DEFAULT_COMMISSION_RATE,
    DEFAULT_MIN_COMMISSION,
    DEFAULT_SLIPPAGE_BP,
    DEFAULT_STAMP_TAX_RATE,
    Fee,
    FeeModel,
)

MODEL = FeeModel()


# ---------------------------------------------------------------------------
# 默认取值
# ---------------------------------------------------------------------------


def test_defaults_are_a_share_current() -> None:
    assert DEFAULT_COMMISSION_RATE == 0.00025
    assert DEFAULT_MIN_COMMISSION == 5.0
    assert DEFAULT_STAMP_TAX_RATE == 0.0005
    assert DEFAULT_SLIPPAGE_BP == 10.0
    assert MODEL.commission_rate == DEFAULT_COMMISSION_RATE
    assert MODEL.slippage_rate == pytest.approx(0.001)


# ---------------------------------------------------------------------------
# 佣金
# ---------------------------------------------------------------------------


def test_commission_above_minimum() -> None:
    # 10 万 × 万 2.5 = 25 元，未触发最低佣金。
    fee = MODEL.buy_cost(100_000.0)
    assert fee.commission == pytest.approx(25.0)


def test_commission_hits_minimum() -> None:
    # 1 万 × 万 2.5 = 2.5 元 < 5 元，按最低 5 元收。
    fee = MODEL.buy_cost(10_000.0)
    assert fee.commission == pytest.approx(5.0)


def test_commission_exactly_at_minimum_boundary() -> None:
    # 2 万 × 万 2.5 = 5 元，正好触底，仍为 5。
    assert MODEL.buy_cost(20_000.0).commission == pytest.approx(5.0)


def test_commission_on_sell_also_has_minimum() -> None:
    assert MODEL.sell_cost(10_000.0).commission == pytest.approx(5.0)


def test_zero_notional_has_no_commission() -> None:
    assert MODEL.buy_cost(0.0).commission == 0.0
    assert MODEL.sell_cost(0.0).commission == 0.0
    assert MODEL.buy_cost(0.0).total == 0.0


# ---------------------------------------------------------------------------
# 印花税
# ---------------------------------------------------------------------------


def test_stamp_tax_only_on_sell() -> None:
    buy = MODEL.buy_cost(100_000.0)
    sell = MODEL.sell_cost(100_000.0)
    assert buy.stamp_tax == 0.0
    assert sell.stamp_tax == pytest.approx(100_000.0 * DEFAULT_STAMP_TAX_RATE)
    assert sell.stamp_tax == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# 滑点与成交价
# ---------------------------------------------------------------------------


def test_execution_price_buy_adds_slippage() -> None:
    assert MODEL.execution_price("buy", 10.0) == pytest.approx(10.01)


def test_execution_price_sell_subtracts_slippage() -> None:
    assert MODEL.execution_price("sell", 10.0) == pytest.approx(9.99)


def test_slippage_cost_component() -> None:
    assert MODEL.buy_cost(100_000.0).slippage == pytest.approx(100.0)
    assert MODEL.sell_cost(100_000.0).slippage == pytest.approx(100.0)


def test_unknown_side_raises() -> None:
    with pytest.raises(ValueError):
        MODEL.execution_price("hold", 10.0)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 结构化费用与自定义参数
# ---------------------------------------------------------------------------


def test_fee_total_and_cash_cost_split() -> None:
    fee = MODEL.sell_cost(100_000.0)
    # 佣金 25 + 印花税 50 = 现金费用 75；滑点 100 已在成交价里。
    assert fee.cash_cost == pytest.approx(75.0)
    assert fee.total == pytest.approx(175.0)


def test_custom_fee_model_is_configurable() -> None:
    model = FeeModel(
        commission_rate=0.0003,
        min_commission=1.0,
        stamp_tax_rate=0.001,
        slippage_bp=20.0,
    )
    buy = model.buy_cost(10_000.0)
    assert buy.commission == pytest.approx(3.0)
    assert buy.slippage == pytest.approx(20.0)  # 10000 × 0.002
    assert model.execution_price("buy", 10.0) == pytest.approx(10.02)
    assert model.execution_price("sell", 10.0) == pytest.approx(9.98)


def test_fee_defaults_are_zero() -> None:
    fee = Fee()
    assert fee.commission == 0.0
    assert fee.stamp_tax == 0.0
    assert fee.slippage == 0.0
    assert fee.cash_cost == 0.0
    assert fee.total == 0.0
