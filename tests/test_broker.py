"""``quant.backtest.broker`` 单元测试：停牌 / 涨跌停 / 整手 / T+1 / 缩单 / 费用。"""
from __future__ import annotations

import polars as pl
import pytest

from quant.backtest.account import Account, Position
from quant.backtest.broker import (
    Broker,
    ExecutionReport,
    Fill,
    KCB_MIN_BUY_VOLUME,
    MAIN_LOT_SIZE,
    Order,
    Reject,
    RejectReason,
    round_buy_volume,
)
from quant.backtest.fee import FeeModel

MODEL = FeeModel()
BROKER = Broker(MODEL)

MAIN = "600000.SH"
KCB = "688001.SH"
SZ_MAIN = "000001.SZ"


# ---------------------------------------------------------------------------
# 构造辅助
# ---------------------------------------------------------------------------


def bars_of(entries: list[tuple[str, float, float | None, float | None]]) -> pl.DataFrame:
    """(instrument, open, limit_up, limit_down) -> 当日行情表。"""
    return pl.DataFrame(
        {
            "instrument": [entry[0] for entry in entries],
            "open": [entry[1] for entry in entries],
            "limit_up": [entry[2] for entry in entries],
            "limit_down": [entry[3] for entry in entries],
        },
        schema={
            "instrument": pl.String,
            "open": pl.Float64,
            "limit_up": pl.Float64,
            "limit_down": pl.Float64,
        },
    )


def normal_bars() -> pl.DataFrame:
    """主板 open=10、涨停 11、跌停 9。"""
    return bars_of([(MAIN, 10.0, 11.0, 9.0)])


def single_fill(report: ExecutionReport) -> Fill:
    assert report.rejects == []
    assert len(report.fills) == 1
    return report.fills[0]


def single_reject(report: ExecutionReport) -> Reject:
    assert report.fills == []
    assert len(report.rejects) == 1
    return report.rejects[0]


# ---------------------------------------------------------------------------
# 订单类型校验
# ---------------------------------------------------------------------------


def test_order_rejects_invalid_side() -> None:
    with pytest.raises(ValueError):
        Order(MAIN, "hold", 100)  # type: ignore[arg-type]


def test_order_rejects_non_positive_volume() -> None:
    with pytest.raises(ValueError):
        Order(MAIN, "buy", 0)
    with pytest.raises(ValueError):
        Order(MAIN, "sell", -1)


# ---------------------------------------------------------------------------
# 正常买入 / 卖出：滑点与费用到账
# ---------------------------------------------------------------------------


def test_buy_fills_at_open_plus_slippage() -> None:
    account = Account(cash=100_000.0)

    report = BROKER.execute([Order(MAIN, "buy", 500)], normal_bars(), account)

    fill = single_fill(report)
    assert fill.instrument == MAIN
    assert fill.side == "buy"
    assert fill.volume == 500
    assert fill.price == pytest.approx(10.01)  # 10 × (1 + 10bp)
    # 名义 500 × 10.01 = 5005；佣金 max(1.25125, 5) = 5；滑点 5.005。
    assert fill.notional == pytest.approx(5005.0)
    assert fill.fee.commission == pytest.approx(5.0)
    assert fill.fee.slippage == pytest.approx(5.005)
    assert fill.fee.cash_cost == pytest.approx(5.0)

    # 现金支出 = 名义 + 现金费用（滑点不重复扣）。
    assert account.cash == pytest.approx(100_000.0 - 5005.0 - 5.0)
    position = account.position(MAIN)
    assert position is not None
    assert position.volume == 500
    assert position.sellable == 0  # 当日买入 T+1 冻结
    assert position.cost_basis == pytest.approx(5010.0)


def test_sell_after_settle_prices_with_slippage_and_stamp_tax() -> None:
    account = Account(cash=100_000.0)
    BROKER.execute([Order(MAIN, "buy", 500)], normal_bars(), account)
    account.settle_new_day()
    cash_after_buy = account.cash

    report = BROKER.execute([Order(MAIN, "sell", 300)], normal_bars(), account)

    fill = single_fill(report)
    assert fill.side == "sell"
    assert fill.volume == 300
    assert fill.price == pytest.approx(9.99)  # 10 × (1 − 10bp)
    # 名义 300 × 9.99 = 2997；佣金 5（0.74925 触最低）；印花税 2997 × 万5 = 1.4985。
    assert fill.notional == pytest.approx(2997.0)
    assert fill.fee.commission == pytest.approx(5.0)
    assert fill.fee.stamp_tax == pytest.approx(1.4985)
    assert fill.fee.slippage == pytest.approx(2.997)
    assert fill.fee.cash_cost == pytest.approx(6.4985)

    assert account.cash == pytest.approx(cash_after_buy + 2997.0 - 6.4985)
    position = account.position(MAIN)
    assert position is not None
    assert position.volume == 200
    assert position.sellable == 200


def test_fill_price_ignores_close_column() -> None:
    # 无前视：即使行情里带了离谱的 close，成交价也只由 open 决定。
    bars = pl.DataFrame(
        {
            "instrument": [MAIN],
            "open": [10.0],
            "close": [999.0],
            "limit_up": [11.0],
            "limit_down": [9.0],
        }
    )
    account = Account(cash=100_000.0)

    fill = single_fill(BROKER.execute([Order(MAIN, "buy", 100)], bars, account))

    assert fill.price == pytest.approx(10.01)


# ---------------------------------------------------------------------------
# 停牌
# ---------------------------------------------------------------------------


def test_buy_rejected_when_instrument_has_no_bar() -> None:
    account = Account(cash=100_000.0)
    bars = bars_of([(SZ_MAIN, 10.0, 11.0, 9.0)])

    reject = single_reject(BROKER.execute([Order(MAIN, "buy", 100)], bars, account))

    assert reject.reason is RejectReason.SUSPENDED
    assert reject.requested == 100
    assert account.cash == 100_000.0
    assert account.position(MAIN) is None


def test_sell_rejected_when_instrument_has_no_bar() -> None:
    account = Account(cash=0.0)
    account.positions[MAIN] = Position(MAIN, volume=500, sellable=500, cost_basis=5000.0)
    bars = bars_of([(SZ_MAIN, 10.0, 11.0, 9.0)])

    reject = single_reject(BROKER.execute([Order(MAIN, "sell", 500)], bars, account))

    assert reject.reason is RejectReason.SUSPENDED
    assert account.position(MAIN) is not None


def test_empty_bars_rejects_every_order() -> None:
    account = Account(cash=100_000.0)

    report = BROKER.execute(
        [Order(MAIN, "buy", 100), Order(SZ_MAIN, "buy", 100)],
        bars_of([]),
        account,
    )

    assert report.fills == []
    assert {reject.reason for reject in report.rejects} == {RejectReason.SUSPENDED}


def test_suspended_takes_priority_over_lot_size() -> None:
    # 停牌拒单先于整手校验：不足一手的单也是 SUSPENDED。
    account = Account(cash=100_000.0)
    reject = single_reject(
        BROKER.execute([Order(MAIN, "buy", 50)], bars_of([]), account)
    )
    assert reject.reason is RejectReason.SUSPENDED


# ---------------------------------------------------------------------------
# 涨跌停（严格版）
# ---------------------------------------------------------------------------


def test_buy_rejected_when_open_at_limit_up() -> None:
    account = Account(cash=100_000.0)
    bars = bars_of([(MAIN, 11.0, 11.0, 9.9)])

    reject = single_reject(BROKER.execute([Order(MAIN, "buy", 100)], bars, account))

    assert reject.reason is RejectReason.LIMIT_UP
    assert account.cash == 100_000.0


def test_buy_rejected_within_price_epsilon() -> None:
    account = Account(cash=100_000.0)
    bars = bars_of([(MAIN, 11.0 - 5e-10, 11.0, 9.9)])

    reject = single_reject(BROKER.execute([Order(MAIN, "buy", 100)], bars, account))

    assert reject.reason is RejectReason.LIMIT_UP


def test_buy_allowed_clearly_below_limit_up() -> None:
    account = Account(cash=100_000.0)
    bars = bars_of([(MAIN, 10.999_999, 11.0, 9.9)])

    fill = single_fill(BROKER.execute([Order(MAIN, "buy", 100)], bars, account))

    assert fill.volume == 100


def test_buy_allowed_when_limit_up_is_null() -> None:
    # 老股首日 / 新股首日无涨跌停价：跳过该项检查。
    account = Account(cash=100_000.0)
    bars = bars_of([(MAIN, 10.0, None, None)])

    fill = single_fill(BROKER.execute([Order(MAIN, "buy", 100)], bars, account))

    assert fill.price == pytest.approx(10.01)


def test_sell_rejected_when_open_at_limit_down() -> None:
    account = Account(cash=0.0)
    account.positions[MAIN] = Position(MAIN, volume=500, sellable=500, cost_basis=5000.0)
    bars = bars_of([(MAIN, 9.0, 11.0, 9.0)])

    reject = single_reject(BROKER.execute([Order(MAIN, "sell", 500)], bars, account))

    assert reject.reason is RejectReason.LIMIT_DOWN
    position = account.position(MAIN)
    assert position is not None
    assert position.volume == 500


def test_sell_rejected_within_price_epsilon() -> None:
    account = Account(cash=0.0)
    account.positions[MAIN] = Position(MAIN, volume=500, sellable=500, cost_basis=5000.0)
    bars = bars_of([(MAIN, 9.0 + 5e-10, 11.0, 9.0)])

    reject = single_reject(BROKER.execute([Order(MAIN, "sell", 500)], bars, account))

    assert reject.reason is RejectReason.LIMIT_DOWN


def test_sell_allowed_when_limit_down_is_null() -> None:
    account = Account(cash=0.0)
    account.positions[MAIN] = Position(MAIN, volume=500, sellable=500, cost_basis=5000.0)
    bars = bars_of([(MAIN, 9.0, None, None)])

    fill = single_fill(BROKER.execute([Order(MAIN, "sell", 500)], bars, account))

    assert fill.volume == 500
    assert fill.price == pytest.approx(8.991)


# ---------------------------------------------------------------------------
# T+1 冻结
# ---------------------------------------------------------------------------


def test_sell_same_day_is_frozen() -> None:
    account = Account(cash=100_000.0)
    BROKER.execute([Order(MAIN, "buy", 500)], normal_bars(), account)

    reject = single_reject(
        BROKER.execute([Order(MAIN, "sell", 500)], normal_bars(), account)
    )

    assert reject.reason is RejectReason.T1_FROZEN
    position = account.position(MAIN)
    assert position is not None
    assert position.volume == 500


def test_sell_truncated_to_sellable() -> None:
    # 部分可卖：超出部分按部分成交，不做整单拒。
    account = Account(cash=0.0)
    account.positions[MAIN] = Position(MAIN, volume=500, sellable=300, cost_basis=5000.0)

    fill = single_fill(BROKER.execute([Order(MAIN, "sell", 500)], normal_bars(), account))

    assert fill.volume == 300
    position = account.position(MAIN)
    assert position is not None
    assert position.volume == 200
    assert position.sellable == 0


def test_sell_without_position_is_frozen() -> None:
    account = Account(cash=0.0)

    reject = single_reject(
        BROKER.execute([Order(MAIN, "sell", 100)], normal_bars(), account)
    )

    assert reject.reason is RejectReason.T1_FROZEN


# ---------------------------------------------------------------------------
# 整手规则
# ---------------------------------------------------------------------------


def test_round_buy_volume_main_board() -> None:
    assert round_buy_volume(MAIN, 250) == 200
    assert round_buy_volume(MAIN, 100) == 100
    assert round_buy_volume(MAIN, 99) == 0
    assert round_buy_volume("300750.SZ", 350) == 300  # 创业板同主板口径


def test_round_buy_volume_kcb() -> None:
    assert round_buy_volume(KCB, 199) == 0
    assert round_buy_volume(KCB, KCB_MIN_BUY_VOLUME) == 200
    assert round_buy_volume(KCB, 250) == 250  # 超过 200 按 1 股递增
    assert round_buy_volume("689009.SH", 201) == 201


def test_buy_rounds_down_to_lot_main_board() -> None:
    account = Account(cash=100_000.0)

    fill = single_fill(BROKER.execute([Order(MAIN, "buy", 250)], normal_bars(), account))

    assert fill.volume == 200
    position = account.position(MAIN)
    assert position is not None
    assert position.volume == 200


def test_buy_below_one_lot_rejected() -> None:
    account = Account(cash=100_000.0)

    reject = single_reject(BROKER.execute([Order(MAIN, "buy", 99)], normal_bars(), account))

    assert reject.reason is RejectReason.LOT_SIZE
    assert account.cash == 100_000.0


def test_kcb_buy_accepts_odd_volume_above_minimum() -> None:
    account = Account(cash=100_000.0)
    bars = bars_of([(KCB, 10.0, 11.0, 9.0)])

    fill = single_fill(BROKER.execute([Order(KCB, "buy", 250)], bars, account))

    assert fill.volume == 250


def test_kcb_buy_below_minimum_rejected() -> None:
    account = Account(cash=100_000.0)
    bars = bars_of([(KCB, 10.0, 11.0, 9.0)])

    reject = single_reject(BROKER.execute([Order(KCB, "buy", 199)], bars, account))

    assert reject.reason is RejectReason.LOT_SIZE


def test_sell_odd_lot_allowed() -> None:
    account = Account(cash=0.0)
    account.positions[MAIN] = Position(MAIN, volume=150, sellable=150, cost_basis=1500.0)

    fill = single_fill(BROKER.execute([Order(MAIN, "sell", 150)], normal_bars(), account))

    assert fill.volume == 150
    assert account.position(MAIN) is None


# ---------------------------------------------------------------------------
# 现金不足缩单
# ---------------------------------------------------------------------------


def test_buy_shrinks_to_affordable_lot() -> None:
    # 现金 2008：300 股需 3003 + 5 = 3008 不够；200 股需 2002 + 5 = 2007，够。
    account = Account(cash=2008.0)

    fill = single_fill(BROKER.execute([Order(MAIN, "buy", 300)], normal_bars(), account))

    assert fill.volume == 200
    assert account.cash == pytest.approx(1.0)


def test_buy_rejected_when_one_lot_unaffordable() -> None:
    # 现金 1000：100 股需 1001 + 5 = 1006，缩到 0 手。
    account = Account(cash=1000.0)

    reject = single_reject(BROKER.execute([Order(MAIN, "buy", 100)], normal_bars(), account))

    assert reject.reason is RejectReason.CASH_SHORT
    assert account.cash == 1000.0


def test_shrink_never_goes_below_kcb_minimum() -> None:
    # 现金 2000 连 200 股（2002 + 5）都买不起，缩单不得落到 199 股。
    account = Account(cash=2000.0)
    bars = bars_of([(KCB, 10.0, 11.0, 9.0)])

    reject = single_reject(BROKER.execute([Order(KCB, "buy", 250)], bars, account))

    assert reject.reason is RejectReason.CASH_SHORT


def test_kcb_shrink_steps_by_one_share() -> None:
    # 现金 2507：250 股需 2502.5 + 5 = 2507.5 不够；249 股需 2492.49 + 5 = 2497.49，够。
    account = Account(cash=2507.0)
    bars = bars_of([(KCB, 10.0, 11.0, 9.0)])

    fill = single_fill(BROKER.execute([Order(KCB, "buy", 250)], bars, account))

    assert fill.volume == 249


def test_orders_share_cash_sequentially() -> None:
    account = Account(cash=10_000.0)
    bars = bars_of([(MAIN, 10.0, 11.0, 9.0), (SZ_MAIN, 20.0, 22.0, 18.0)])

    report = BROKER.execute(
        [Order(MAIN, "buy", 500), Order(SZ_MAIN, "buy", 500)], bars, account
    )

    assert len(report.fills) == 2
    first, second = report.fills
    assert first.volume == 500  # 5005 + 5 = 5010
    assert second.volume == 200  # 剩余 4990：200 × 20.02 = 4004 + 5 = 4009
    assert account.cash == pytest.approx(4990.0 - 4009.0)


# ---------------------------------------------------------------------------
# 报告汇总
# ---------------------------------------------------------------------------


def test_report_totals_for_buys_only() -> None:
    account = Account(cash=1_000_000.0)
    bars = bars_of([(MAIN, 10.0, 11.0, 9.0), (SZ_MAIN, 20.0, 22.0, 18.0)])

    report = BROKER.execute(
        [Order(MAIN, "buy", 1000), Order(SZ_MAIN, "buy", 500)], bars, account
    )

    assert report.buy_amount == pytest.approx(1000 * 10.01 + 500 * 20.02)
    assert report.sell_amount == 0.0
    assert report.cash_cost == pytest.approx(10.0)  # 两笔佣金各 5
    assert report.total_cost == pytest.approx(10.0 + 20020.0 * 0.001)
    assert report.net_cash_flow == pytest.approx(-(20020.0 + 10.0))
    assert report.filled_volume(MAIN) == 1000


def test_report_totals_with_buy_and_sell() -> None:
    account = Account(cash=100_000.0)
    account.positions[MAIN] = Position(MAIN, volume=1000, sellable=1000, cost_basis=10_000.0)
    bars = bars_of([(MAIN, 10.0, 11.0, 9.0), (SZ_MAIN, 20.0, 22.0, 18.0)])

    report = BROKER.execute(
        [Order(SZ_MAIN, "buy", 100), Order(MAIN, "sell", 1000)], bars, account
    )

    assert report.buy_amount == pytest.approx(100 * 20.02)
    assert report.sell_amount == pytest.approx(1000 * 9.99)
    # 买佣金 5 + 卖佣金 5 + 印花税 9990 × 0.0005 = 4.995
    assert report.cash_cost == pytest.approx(14.995)
    assert report.net_cash_flow == pytest.approx(9990.0 - 2002.0 - 14.995)


def test_report_keeps_fills_and_rejects_together() -> None:
    account = Account(cash=100_000.0)
    bars = bars_of([(MAIN, 10.0, 11.0, 9.0)])

    report = BROKER.execute(
        [Order(MAIN, "buy", 300), Order(SZ_MAIN, "buy", 100)], bars, account
    )

    assert len(report.fills) == 1
    assert len(report.rejects) == 1
    assert report.rejects[0].reason is RejectReason.SUSPENDED
    assert report.buy_amount == pytest.approx(3003.0)
