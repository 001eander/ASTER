"""``quant.backtest.engine`` 单元测试：主循环时序、无前视、T+1、停牌、公司行为、拒单。

mini 案例全部用合成行情，nav 序列按费用 / 滑点 / 收盘价手算后写成字面量核对。
"""
from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from quant.backtest.broker import Broker, Order, RejectReason
from quant.backtest.engine import BacktestEngine, BacktestResult
from quant.backtest.fee import FeeModel
from quant.data.schema import CORPORATE_ACTIONS, DAILY_BARS, TRADE_CALENDAR

A = "600000.SH"
B = "000001.SZ"
C = "601398.SH"

DAY0 = date(2024, 3, 1)
DAYS_20 = [DAY0 + timedelta(days=i) for i in range(20)]

#: 三只票的收盘价序列（同时用作开盘价）：A 每两日涨 1 元，B 恒定，C 每日跌 0.1 元。
A_SERIES = {day: 10.0 + 0.5 * i for i, day in enumerate(DAYS_20)}
B_SERIES = {day: 20.0 for day in DAYS_20}
C_SERIES = {day: 5.0 - 0.1 * i for i, day in enumerate(DAYS_20)}


# ---------------------------------------------------------------------------
# 构造辅助
# ---------------------------------------------------------------------------


def engine() -> BacktestEngine:
    return BacktestEngine(Broker(FeeModel()))


def bar(
    day: date,
    instrument: str,
    close: float,
    *,
    open_price: float | None = None,
    limit_up: float | None = None,
    limit_down: float | None = None,
) -> tuple[date, str, float, float, float | None, float | None]:
    """一行行情：(date, instrument, open, close, limit_up, limit_down)。"""
    return (
        day,
        instrument,
        close if open_price is None else open_price,
        close,
        close * 1.1 if limit_up is None else limit_up,
        close * 0.9 if limit_down is None else limit_down,
    )


def build_bars(
    rows: list[tuple[date, str, float, float, float | None, float | None]],
) -> pl.DataFrame:
    """按 DAILY_BARS 全列构造行情表（未使用的列留空）。"""
    count = len(rows)
    return pl.DataFrame(
        {
            "date": [row[0] for row in rows],
            "instrument": [row[1] for row in rows],
            "open": [row[2] for row in rows],
            "high": [None] * count,
            "low": [None] * count,
            "close": [row[3] for row in rows],
            "vwap": [None] * count,
            "volume": [None] * count,
            "amount": [None] * count,
            "adjfactor": [None] * count,
            "limit_up": [row[4] for row in rows],
            "limit_down": [row[5] for row in rows],
        },
        schema=DAILY_BARS,
    )


def calendar_of(days: list[date], closed: frozenset[date] = frozenset()) -> pl.DataFrame:
    return pl.DataFrame(
        {"date": days, "is_open": [day not in closed for day in days]},
        schema=TRADE_CALENDAR,
    )


def build_actions(
    rows: list[tuple[date, str, float | None, float | None]],
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [row[0] for row in rows],
            "instrument": [row[1] for row in rows],
            "cash_per_share": [row[2] for row in rows],
            "share_per_share": [row[3] for row in rows],
        },
        schema=CORPORATE_ACTIONS,
    )


def no_actions() -> pl.DataFrame:
    return build_actions([])


def mini_market() -> pl.DataFrame:
    """20 个交易日、3 只票的全行情表。"""
    rows: list[tuple[date, str, float, float, float | None, float | None]] = []
    for day in DAYS_20:
        rows.append(bar(day, A, A_SERIES[day]))
        rows.append(bar(day, B, B_SERIES[day]))
        rows.append(bar(day, C, C_SERIES[day]))
    return build_bars(rows)


def hold_first_day_signal(orders: dict[date, list[Order]]):
    def signal(day: date, history: pl.DataFrame) -> list[Order]:
        return list(orders.get(day, []))

    return signal


# ---------------------------------------------------------------------------
# 端到端 mini 案例：第一天买入后持有
# ---------------------------------------------------------------------------


def test_hold_from_first_day_matches_hand_computed_nav() -> None:
    orders = {
        DAYS_20[0]: [
            Order(A, "buy", 1000),
            Order(B, "buy", 500),
            Order(C, "buy", 200),
        ]
    }

    result = engine().run(
        mini_market(),
        calendar_of(DAYS_20),
        no_actions(),
        hold_first_day_signal(orders),
        DAYS_20[0],
        DAYS_20[-1],
        100_000.0,
    )

    assert result.trading_days == 20
    assert result.nav["date"].to_list() == DAYS_20

    # 第 0 日空仓：nav = 初始资金。
    assert result.nav["nav"][0] == pytest.approx(100_000.0)
    assert result.nav["market_value"][0] == 0.0
    assert result.nav["turnover"][0] == 0.0

    # 三笔买单都在第 1 日开盘成交（T 日收盘下单 → T+1 开盘撮合）。
    assert [(fill.date, fill.instrument, fill.side) for fill in result.fills] == [
        (DAYS_20[1], A, "buy"),
        (DAYS_20[1], B, "buy"),
        (DAYS_20[1], C, "buy"),
    ]
    assert result.rejects == []

    # 成交价 = 开盘价 × (1 + 10bp)；佣金取最低 5 元（三笔名义额都很小）。
    fill_a, fill_b, fill_c = result.fills
    assert fill_a.price == pytest.approx(10.5 * 1.001)
    assert fill_b.price == pytest.approx(20.0 * 1.001)
    assert fill_c.price == pytest.approx(4.9 * 1.001)
    assert fill_a.fee.cash_cost == pytest.approx(5.0)
    assert fill_b.fee.cash_cost == pytest.approx(5.0)
    assert fill_c.fee.cash_cost == pytest.approx(5.0)

    # 手算现金：
    #   A: 1000 × 10.5105 + 5 = 10515.50
    #   B:  500 × 20.0200 + 5 = 10015.00
    #   C:  200 ×  4.9049 + 5 =   985.98
    cash = 100_000.0 - 10_515.50 - 10_015.00 - 985.98
    assert cash == pytest.approx(78_483.52)
    assert result.nav["cash"][1] == pytest.approx(cash)

    # 手算逐日 nav = 现金 + 1000×A 收盘 + 500×20 + 200×C 收盘
    #          = 78483.52 + 21000 + 480 × i = 99483.52 + 480 i（第 i 个交易日，i ≥ 1）。
    expected_nav = [100_000.0] + [99_483.52 + 480.0 * i for i in range(1, 20)]
    assert result.nav["nav"].to_list() == pytest.approx(expected_nav)
    assert result.final_nav == pytest.approx(99_483.52 + 480.0 * 19)
    assert result.total_return == pytest.approx(result.final_nav / 100_000.0 - 1.0)

    # 换手率 = 当日成交额 / 当日组合市值。
    traded = 10_510.50 + 10_010.00 + 980.98
    assert result.nav["traded_amount"][1] == pytest.approx(traded)
    assert result.nav["turnover"][1] == pytest.approx(traded / expected_nav[1])
    assert result.nav["traded_amount"].to_list()[2:] == [0.0] * 18


def test_closed_days_are_skipped_and_orders_wait_for_next_open_day() -> None:
    days = [DAY0 + timedelta(days=i) for i in range(4)]
    closed = frozenset({days[1]})
    bars = build_bars([bar(day, A, 10.0) for day in [days[0], days[2], days[3]]])
    orders = {days[0]: [Order(A, "buy", 100)]}

    result = engine().run(
        bars,
        calendar_of(days, closed),
        no_actions(),
        hold_first_day_signal(orders),
        days[0],
        days[-1],
        100_000.0,
    )

    assert result.nav["date"].to_list() == [days[0], days[2], days[3]]
    assert [fill.date for fill in result.fills] == [days[2]]


# ---------------------------------------------------------------------------
# 无前视：signal_fn 只看到 ≤ T 的数据
# ---------------------------------------------------------------------------


def test_signal_history_never_exceeds_signal_day() -> None:
    seen: list[tuple[date, date | None, int]] = []

    def signal(day: date, history: pl.DataFrame) -> list[Order]:
        assert (history["date"] > day).sum() == 0
        seen.append((day, history["date"].max(), history.height))
        return [Order(A, "buy", 100)] if day == DAYS_20[0] else []

    engine().run(
        mini_market(),
        calendar_of(DAYS_20),
        no_actions(),
        signal,
        DAYS_20[0],
        DAYS_20[-1],
        100_000.0,
    )

    # 区间最后一个开市日不再生成订单。
    assert [item[0] for item in seen] == DAYS_20[:-1]
    for day, latest, height in seen:
        assert latest == day
        assert height == 3 * (DAYS_20.index(day) + 1)


def test_signal_receives_pre_start_history() -> None:
    days = DAYS_20[:5]
    bars = mini_market().filter(pl.col("date") <= days[3])
    heights: list[int] = []

    def signal(day: date, history: pl.DataFrame) -> list[Order]:
        heights.append(history.height)
        return []

    engine().run(
        bars, calendar_of(days), no_actions(), signal, days[1], days[4], 100_000.0
    )

    # start 之前的行仍然进入 history（因子回溯窗口），只是不参与撮合与估值。
    assert heights == [3 * 2, 3 * 3, 3 * 4]


def test_signal_must_return_orders() -> None:
    days = DAYS_20[:3]

    def signal(day: date, history: pl.DataFrame) -> list[Order]:
        return ["not an order"]  # type: ignore[list-item]

    with pytest.raises(TypeError, match="Order"):
        engine().run(
            build_bars([bar(day, A, 10.0) for day in days]),
            calendar_of(days),
            no_actions(),
            signal,
            days[0],
            days[-1],
            100_000.0,
        )


# ---------------------------------------------------------------------------
# T+1 冻结贯穿
# ---------------------------------------------------------------------------


def test_sell_generated_after_buy_fills_next_day() -> None:
    days = DAYS_20[:5]
    bars = build_bars([bar(day, A, 10.0) for day in days])

    def signal(day: date, history: pl.DataFrame) -> list[Order]:
        if day == days[0]:
            return [Order(A, "buy", 300)]
        if day == days[1]:
            # 第 1 日买入的 300 股当日冻结，只能等第 2 日开盘解冻后卖出。
            return [Order(A, "sell", 300)]
        return []

    result = engine().run(
        bars, calendar_of(days), no_actions(), signal, days[0], days[-1], 100_000.0
    )

    # 若引擎漏掉开盘前的 settle_new_day，第 2 日的卖单会被拒 T1_FROZEN。
    assert result.rejects == []
    assert [(fill.date, fill.side, fill.volume) for fill in result.fills] == [
        (days[1], "buy", 300),
        (days[2], "sell", 300),
    ]
    sell = result.fills[1]
    # 卖价 = 开盘 10 × (1 − 10bp)；佣金 5 + 印花税 2997 × 万5。
    assert sell.price == pytest.approx(9.99)
    assert sell.fee.cash_cost == pytest.approx(5.0 + 2997.0 * 0.0005)

    # 买 300 股花 3003 + 5；卖 300 股收 2997 − 6.4985。
    expected_cash = 100_000.0 - 3008.0 + 2997.0 - 6.4985
    assert result.nav["cash"][-1] == pytest.approx(expected_cash)
    assert result.nav["nav"][-1] == pytest.approx(expected_cash)


# ---------------------------------------------------------------------------
# 停牌：按最近可得价估值，nav 不断档
# ---------------------------------------------------------------------------


def test_suspension_valued_at_last_price() -> None:
    days = DAYS_20[:6]
    quoted = {days[0]: 10.0, days[1]: 11.0, days[2]: 12.0, days[5]: 15.0}
    bars = build_bars([bar(day, A, price) for day, price in quoted.items()])

    def signal(day: date, history: pl.DataFrame) -> list[Order]:
        if day == days[0]:
            return [Order(A, "buy", 300)]
        if day == days[2]:
            # 次日（停牌日）要卖的这条单会被拒 SUSPENDED。
            return [Order(A, "sell", 100)]
        return []

    result = engine().run(
        bars, calendar_of(days), no_actions(), signal, days[0], days[-1], 100_000.0
    )

    assert result.nav["date"].to_list() == days  # 停牌日仍逐日记净值
    cash = 100_000.0 - (300 * 11.0 * 1.001 + 5.0)
    assert cash == pytest.approx(96_691.70)

    # 第 1 日按 11 收盘估值；第 2–4 日沿用最近的 12；第 5 日回到 15。
    expected_nav = [
        100_000.0,
        cash + 300 * 11.0,
        cash + 300 * 12.0,
        cash + 300 * 12.0,
        cash + 300 * 12.0,
        cash + 300 * 15.0,
    ]
    assert result.nav["nav"].to_list() == pytest.approx(expected_nav)

    assert len(result.rejects) == 1
    assert result.rejects[0].date == days[3]
    assert result.rejects[0].reason is RejectReason.SUSPENDED
    # 拒单不改账户，第 3、4 日 nav 与拒单前一致。
    assert result.nav["nav"][3] == pytest.approx(result.nav["nav"][2])


# ---------------------------------------------------------------------------
# 公司行为：除权日 nav 不产生假跳变
# ---------------------------------------------------------------------------


def dividend_market(ex_date: date, close_on_ex: float) -> pl.DataFrame:
    """除权日前收盘 10 元，除权日及之后维持除权后价格 close_on_ex。"""
    days = DAYS_20[:6]
    return build_bars(
        [bar(day, A, close_on_ex if day >= ex_date else 10.0) for day in days]
    )


def buy_first_day_signal(first_day: date, volume: int = 1000):
    def signal(day: date, history: pl.DataFrame) -> list[Order]:
        return [Order(A, "buy", volume)] if day == first_day else []

    return signal


def test_cash_dividend_credited_and_nav_continuous() -> None:
    days = DAYS_20[:6]
    ex_date = days[4]
    actions = build_actions([(ex_date, A, 0.5, 0.0)])

    result = engine().run(
        dividend_market(ex_date, 9.5),
        calendar_of(days),
        actions,
        buy_first_day_signal(days[0]),
        days[0],
        days[-1],
        100_000.0,
    )

    cash = 100_000.0 - (1000 * 10.01 + 5.0)
    assert cash == pytest.approx(89_985.0)
    assert result.nav["nav"].to_list() == pytest.approx([100_000.0] + [cash + 10_000.0] * 5)

    detail = result.corporate_actions[0]
    assert detail.date == ex_date
    assert detail.instrument == A
    assert detail.cash_received == pytest.approx(500.0)
    assert detail.price_before == pytest.approx(10.0)
    assert detail.market_value_before == pytest.approx(10_000.0)
    assert (detail.volume_before, detail.volume_after, detail.shares_added) == (
        1000,
        1000,
        0,
    )
    # 除权日现金多 500，持仓市值少 500，nav 与前一日相同。
    assert result.nav["nav"][4] == pytest.approx(result.nav["nav"][3])
    assert result.nav["cash"][4] == pytest.approx(result.nav["cash"][3] + 500.0)


def test_share_bonus_credited_and_nav_continuous() -> None:
    days = DAYS_20[:6]
    ex_date = days[4]
    actions = build_actions([(ex_date, A, 0.0, 1.0)])  # 10 送 10

    result = engine().run(
        dividend_market(ex_date, 5.0),
        calendar_of(days),
        actions,
        buy_first_day_signal(days[0]),
        days[0],
        days[-1],
        100_000.0,
    )

    cash = 100_000.0 - (1000 * 10.01 + 5.0)
    # 股数翻倍、价格腰斩，市值不变。
    assert result.nav["nav"].to_list() == pytest.approx([100_000.0] + [cash + 10_000.0] * 5)

    detail = result.corporate_actions[0]
    assert detail.share_per_share == pytest.approx(1.0)
    assert (detail.volume_before, detail.volume_after, detail.shares_added) == (
        1000,
        2000,
        1000,
    )
    assert detail.cash_received == 0.0


def test_cash_and_bonus_together_keep_nav_flat() -> None:
    days = DAYS_20[:6]
    ex_date = days[4]
    actions = build_actions([(ex_date, A, 0.5, 1.0)])

    result = engine().run(
        dividend_market(ex_date, 4.75),  # (10 − 0.5) / 2
        calendar_of(days),
        actions,
        buy_first_day_signal(days[0]),
        days[0],
        days[-1],
        100_000.0,
    )

    cash = 100_000.0 - (1000 * 10.01 + 5.0)
    assert result.nav["nav"].to_list() == pytest.approx([100_000.0] + [cash + 10_000.0] * 5)

    detail = result.corporate_actions[0]
    assert detail.cash_received == pytest.approx(500.0)  # 按送转前的 1000 股计提
    assert detail.shares_added == 1000


def test_corporate_action_ignored_for_unheld_instrument() -> None:
    days = DAYS_20[:4]
    bars = build_bars([bar(day, A, 10.0) for day in days])

    result = engine().run(
        bars,
        calendar_of(days),
        build_actions([(days[2], B, 0.5, 1.0)]),
        lambda day, history: [],
        days[0],
        days[-1],
        100_000.0,
    )

    assert result.corporate_actions == []
    assert result.nav["nav"].to_list() == pytest.approx([100_000.0] * 4)


# ---------------------------------------------------------------------------
# 拒单传导
# ---------------------------------------------------------------------------


def test_limit_up_buy_rejected_leaves_nav_unchanged() -> None:
    days = DAYS_20[:3]
    bars = build_bars(
        [
            bar(days[0], A, 10.0),
            bar(days[1], A, 11.0, open_price=11.0, limit_up=11.0, limit_down=9.0),
            bar(days[2], A, 11.0),
        ]
    )

    result = engine().run(
        bars,
        calendar_of(days),
        no_actions(),
        buy_first_day_signal(days[0], volume=100),
        days[0],
        days[-1],
        100_000.0,
    )

    assert result.fills == []
    assert len(result.rejects) == 1
    assert result.rejects[0].date == days[1]
    assert result.rejects[0].reason is RejectReason.LIMIT_UP
    assert result.rejects[0].requested == 100
    assert result.nav["nav"].to_list() == pytest.approx([100_000.0] * 3)
    assert result.nav["cash"].to_list() == pytest.approx([100_000.0] * 3)


# ---------------------------------------------------------------------------
# 入参与边界
# ---------------------------------------------------------------------------


def test_run_rejects_bars_without_required_columns() -> None:
    with pytest.raises(ValueError, match="行情表缺少列"):
        engine().run(
            pl.DataFrame({"date": [DAY0], "instrument": [A]}, schema={"date": pl.Date, "instrument": pl.String}),
            calendar_of([DAY0]),
            no_actions(),
            lambda day, history: [],
            DAY0,
            DAY0,
            100_000.0,
        )


def test_empty_range_yields_empty_nav() -> None:
    result = engine().run(
        mini_market(),
        calendar_of(DAYS_20),
        no_actions(),
        lambda day, history: [],
        DAY0 - timedelta(days=10),
        DAY0 - timedelta(days=5),
        50_000.0,
    )

    assert isinstance(result, BacktestResult)
    assert result.trading_days == 0
    assert result.final_nav == 50_000.0
    assert result.nav.columns == list(
        ["date", "cash", "market_value", "nav", "traded_amount", "turnover"]
    )


def test_negative_initial_cash_rejected() -> None:
    with pytest.raises(ValueError, match="初始资金"):
        engine().run(
            mini_market(),
            calendar_of(DAYS_20),
            no_actions(),
            lambda day, history: [],
            DAYS_20[0],
            DAYS_20[-1],
            -1.0,
        )
