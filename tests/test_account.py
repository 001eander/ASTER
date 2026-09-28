"""``quant.backtest.account`` 单元测试：现金、T+1、成本、公司行为、序列化。"""
from __future__ import annotations

import pytest

from quant.backtest.account import (
    Account,
    AccountError,
    InsufficientCashError,
    InsufficientPositionError,
    Position,
)
from quant.backtest.fee import Fee

ZERO_FEE = Fee()


# ---------------------------------------------------------------------------
# 基本买入
# ---------------------------------------------------------------------------


def test_buy_deducts_cash_and_freezes_sellable() -> None:
    account = Account(cash=100_000.0)
    account.buy("600000.SH", 1000, 10.0, Fee(commission=5.0))

    assert account.cash == pytest.approx(100_000.0 - 10_000.0 - 5.0)
    position = account.position("600000.SH")
    assert position is not None
    assert position.volume == 1000
    assert position.sellable == 0  # 当日买入 T+1 冻结
    assert position.cost_basis == pytest.approx(10_005.0)
    assert position.avg_cost == pytest.approx(10.005)


def test_buy_accumulates_position() -> None:
    account = Account(cash=100_000.0)
    account.buy("600000.SH", 1000, 10.0, ZERO_FEE)
    account.buy("600000.SH", 500, 12.0, ZERO_FEE)

    position = account.position("600000.SH")
    assert position is not None
    assert position.volume == 1500
    assert position.sellable == 0
    assert position.cost_basis == pytest.approx(10_000.0 + 6_000.0)


# ---------------------------------------------------------------------------
# T+1
# ---------------------------------------------------------------------------


def test_cannot_sell_same_day() -> None:
    account = Account(cash=100_000.0)
    account.buy("600000.SH", 1000, 10.0, ZERO_FEE)
    with pytest.raises(InsufficientPositionError):
        account.sell("600000.SH", 1000, 11.0, ZERO_FEE)


def test_settle_new_day_unlocks_previous_buy() -> None:
    account = Account(cash=100_000.0)
    account.buy("600000.SH", 1000, 10.0, ZERO_FEE)
    account.settle_new_day()

    position = account.position("600000.SH")
    assert position is not None
    assert position.sellable == 1000

    account.sell("600000.SH", 1000, 11.0, ZERO_FEE)
    assert account.position("600000.SH") is None


def test_settle_does_not_unlock_same_day_buy_after_settlement() -> None:
    account = Account(cash=100_000.0)
    account.buy("600000.SH", 1000, 10.0, ZERO_FEE)
    account.settle_new_day()
    account.buy("600000.SH", 500, 10.0, ZERO_FEE)  # 新一轮当日买入
    position = account.position("600000.SH")
    assert position is not None
    assert position.sellable == 1000  # 只有旧仓可卖
    assert position.volume == 1500


# ---------------------------------------------------------------------------
# 卖出与成本结转
# ---------------------------------------------------------------------------


def test_sell_adds_cash_and_reduces_cost_basis() -> None:
    account = Account(cash=100_000.0)
    account.buy("600000.SH", 1000, 10.0, ZERO_FEE)
    account.settle_new_day()

    proceeds = account.sell("600000.SH", 400, 12.0, Fee(commission=5.0, stamp_tax=2.4))
    assert proceeds == pytest.approx(400 * 12.0 - 7.4)
    assert account.cash == pytest.approx(100_000.0 - 10_000.0 + 4800.0 - 7.4)

    position = account.position("600000.SH")
    assert position is not None
    assert position.volume == 600
    assert position.sellable == 600
    assert position.cost_basis == pytest.approx(6_000.0)


def test_sell_more_than_sellable_raises() -> None:
    account = Account(cash=100_000.0)
    account.buy("600000.SH", 1000, 10.0, ZERO_FEE)
    account.settle_new_day()
    with pytest.raises(InsufficientPositionError):
        account.sell("600000.SH", 1001, 11.0, ZERO_FEE)


def test_sell_without_position_raises() -> None:
    account = Account(cash=100_000.0)
    with pytest.raises(InsufficientPositionError):
        account.sell("600000.SH", 100, 11.0, ZERO_FEE)


# ---------------------------------------------------------------------------
# 超支与参数校验
# ---------------------------------------------------------------------------


def test_overspend_raises() -> None:
    account = Account(cash=100.0)
    with pytest.raises(InsufficientCashError):
        account.buy("600000.SH", 100, 10.0, ZERO_FEE)


def test_overspend_including_fee_raises() -> None:
    account = Account(cash=10_000.0)
    with pytest.raises(InsufficientCashError):
        # 名义金额刚好等于现金，但佣金 5 元导致超支。
        account.buy("600000.SH", 1000, 10.0, Fee(commission=5.0))


def test_buy_tolerates_float_rounding() -> None:
    account = Account(cash=1.0)
    account.buy("600000.SH", 10, 0.1, ZERO_FEE)  # 10 × 0.1 = 1.0
    assert account.cash == pytest.approx(0.0, abs=1e-9)


def test_non_positive_volume_raises() -> None:
    account = Account(cash=100_000.0)
    with pytest.raises(AccountError):
        account.buy("600000.SH", 0, 10.0, ZERO_FEE)
    with pytest.raises(AccountError):
        account.sell("600000.SH", 0, 10.0, ZERO_FEE)


# ---------------------------------------------------------------------------
# 估值与净值
# ---------------------------------------------------------------------------


def test_market_value_nav() -> None:
    account = Account(cash=50_000.0)
    account.buy("600000.SH", 1000, 10.0, ZERO_FEE)
    account.buy("000001.SZ", 500, 20.0, ZERO_FEE)

    prices = {"600000.SH": 12.0, "000001.SZ": 25.0}
    assert account.market_value(prices) == pytest.approx(1000 * 12.0 + 500 * 25.0)
    assert account.nav(prices) == pytest.approx(50_000.0 - 20_000.0 + 24_500.0)


def test_market_value_missing_price_raises() -> None:
    account = Account(cash=50_000.0)
    account.buy("600000.SH", 100, 10.0, ZERO_FEE)
    with pytest.raises(KeyError):
        account.market_value({})


# ---------------------------------------------------------------------------
# 公司行为
# ---------------------------------------------------------------------------


def test_cash_dividend_increases_cash() -> None:
    account = Account(cash=100_000.0)
    account.buy("600000.SH", 1000, 10.0, ZERO_FEE)
    account.apply_corporate_action("600000.SH", cash_per_share=0.5)

    assert account.cash == pytest.approx(100_000.0 - 10_000.0 + 500.0)
    position = account.position("600000.SH")
    assert position is not None
    assert position.volume == 1000  # 分红不改股数


def test_bonus_shares_increase_volume_and_dilute_cost() -> None:
    account = Account(cash=100_000.0)
    account.buy("600000.SH", 1000, 10.0, ZERO_FEE)
    account.settle_new_day()
    account.apply_corporate_action("600000.SH", share_per_share=0.3)

    position = account.position("600000.SH")
    assert position is not None
    assert position.volume == 1300
    assert position.sellable == 1300  # 送转股直接可卖
    assert position.cost_basis == pytest.approx(10_000.0)  # 总成本不变
    assert position.avg_cost == pytest.approx(10_000.0 / 1300)


def test_dividend_and_bonus_together() -> None:
    account = Account(cash=100_000.0)
    account.buy("600000.SH", 1000, 10.0, ZERO_FEE)
    account.apply_corporate_action(
        "600000.SH", cash_per_share=0.2, share_per_share=0.5
    )
    assert account.cash == pytest.approx(100_000.0 - 10_000.0 + 200.0)
    position = account.position("600000.SH")
    assert position is not None
    assert position.volume == 1500


def test_corporate_action_ignores_unheld_instrument() -> None:
    account = Account(cash=0.0)
    account.apply_corporate_action("600000.SH", cash_per_share=0.5)
    assert account.cash == 0.0
    assert account.position("600000.SH") is None


# ---------------------------------------------------------------------------
# 序列化
# ---------------------------------------------------------------------------


def test_serialization_round_trip() -> None:
    account = Account(cash=123_456.78)
    account.buy("600000.SH", 1000, 10.0, Fee(commission=5.0))
    account.settle_new_day()
    account.buy("000001.SZ", 300, 20.0, ZERO_FEE)
    account.sell("600000.SH", 200, 11.0, Fee(commission=5.0, stamp_tax=1.1))

    restored = Account.from_dict(account.to_dict())

    assert restored.cash == pytest.approx(account.cash)
    assert set(restored.positions) == set(account.positions)
    for instrument, position in account.positions.items():
        other = restored.position(instrument)
        assert other is not None
        assert other.volume == position.volume
        assert other.sellable == position.sellable
        assert other.cost_basis == pytest.approx(position.cost_basis)


def test_position_round_trip() -> None:
    position = Position("600000.SH", volume=1000, sellable=800, cost_basis=10_005.0)
    restored = Position.from_dict(position.to_dict())
    assert restored == position
