"""回测引擎：按交易日推进的主循环（架构 3.5）。

主循环时序
----------
对区间 ``[start, end]`` 内每个开市日 ``d``（即 T+1）：

1. **开盘前**：``account.settle_new_day()`` 解冻昨日买入。
2. **开盘撮合**：``broker.execute(pending, d 当日行情, account)``；``pending`` 由上一
   交易日收盘时的 ``signal_fn`` 给出。
3. **日终公司行为**：除权日 = ``d``，调 :func:`quant.backtest.corporate.apply_corporate_actions`。
4. **日终估值**：按当日收盘价记 ``nav``；停牌（当日无行情）持仓沿用最近可得价。
5. **收盘决策**：``signal_fn(d, history)`` 生成次日订单，``history`` 只含 ``date <= d``
   的行。

无前视的结构性保证
------------------
撮合只读 T+1 的开盘价与涨跌停，估值只读 T+1 的收盘价，二者都在 T 的决策之后发生；
``signal_fn`` 拿到的 ``history`` 由引擎按日期前缀切片，调用方无从取得 ``> T`` 的数据。

性能
----
``bars`` 在 ``run`` 入口按 ``(date, instrument)`` 排序一次并 ``partition_by("date")``
切成逐日视图，``history`` 用 Arrow 前缀切片（零拷贝）取得，公司行为按日预分组，主循环
不做全表逐行扫描。
"""
from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Sequence

import polars as pl

from quant.backtest.account import Account
from quant.backtest.broker import Broker, ExecutionReport, Order, RejectReason
from quant.backtest.corporate import CorporateActionDetail, apply_corporate_actions
from quant.backtest.fee import Fee, Side

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

#: 行情表（DAILY_BARS）的必要列；``limit_up`` / ``limit_down`` 缺失时 broker 跳过涨跌停检查。
BAR_COLUMNS: tuple[str, ...] = ("date", "instrument", "open", "close")

#: 逐日净值表 schema。
NAV_SCHEMA: pl.Schema = pl.Schema(
    {
        "date": pl.Date,
        "cash": pl.Float64,
        "market_value": pl.Float64,
        "nav": pl.Float64,
        "traded_amount": pl.Float64,
        "turnover": pl.Float64,
    }
)

#: 当日无行情时交给 broker 的空表：所有订单按停牌拒绝。
EMPTY_BARS: pl.DataFrame = pl.DataFrame(
    schema={
        "instrument": pl.String,
        "open": pl.Float64,
        "limit_up": pl.Float64,
        "limit_down": pl.Float64,
    }
)

#: 信号函数：给定 T 日与 ``date <= T`` 的历史行情，返回 T+1 开盘的订单。
SignalFn = Callable[[date, pl.DataFrame], list[Order]]


# ---------------------------------------------------------------------------
# 产出类型
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FillRecord:
    """带日期的成交记录。``price`` 为含滑点成交价，``fee`` 为对应费用。"""

    date: date
    instrument: str
    side: Side
    volume: int
    price: float
    fee: Fee

    @property
    def notional(self) -> float:
        """成交名义金额（元）。"""
        return self.volume * self.price


@dataclass(frozen=True)
class RejectRecord:
    """带日期的拒单记录。``requested`` 为原意愿股数。"""

    date: date
    instrument: str
    side: Side
    reason: RejectReason
    requested: int = 0
    detail: str = ""


@dataclass(frozen=True)
class BacktestResult:
    """回测产出：逐日净值表 + 全部成交/拒单 + 公司行为明细。

    ``nav`` 列：``date`` / ``cash`` / ``market_value`` / ``nav`` / ``traded_amount``
    （当日买卖成交额之和）/ ``turnover``（``traded_amount / nav``）。
    """

    nav: pl.DataFrame
    fills: list[FillRecord] = field(default_factory=list)
    rejects: list[RejectRecord] = field(default_factory=list)
    corporate_actions: list[CorporateActionDetail] = field(default_factory=list)
    initial_cash: float = 0.0

    @property
    def trading_days(self) -> int:
        """回测覆盖的开市日数。"""
        return self.nav.height

    @property
    def final_nav(self) -> float:
        """最后一个交易日的净资产；无交易日时为初始资金。"""
        if self.nav.height == 0:
            return self.initial_cash
        return float(self.nav["nav"][-1])

    @property
    def total_return(self) -> float:
        """区间总收益（末值 / 初值 − 1）。"""
        if self.initial_cash <= 0.0:
            return 0.0
        return self.final_nav / self.initial_cash - 1.0


# ---------------------------------------------------------------------------
# 引擎
# ---------------------------------------------------------------------------


class BacktestEngine:
    """按交易日推进账户级回测。"""

    def __init__(self, broker: Broker) -> None:
        self.broker = broker

    def run(
        self,
        bars: pl.DataFrame,
        calendar: pl.DataFrame,
        actions: pl.DataFrame,
        signal_fn: SignalFn,
        start: date,
        end: date,
        initial_cash: float,
    ) -> BacktestResult:
        """跑完 ``[start, end]`` 区间，返回净值表与全部成交流水。

        ``bars`` 为 DAILY_BARS 全历史（可含 ``start`` 之前的行，作为因子回溯窗口），
        ``calendar`` 为 TRADE_CALENDAR，``actions`` 为 CORPORATE_ACTIONS。
        ``signal_fn(T, history)`` 在 T 日收盘后调用，返回 T+1 开盘的订单；区间最后一个
        开市日不再调用（生成的订单没有可执行的下一日）。
        """
        _check_columns(bars, BAR_COLUMNS, "行情表")
        if initial_cash < 0.0:
            raise ValueError(f"初始资金不能为负: {initial_cash}")
        if end < start:
            raise ValueError(f"区间非法: start={start} > end={end}")

        ordered = bars.sort(["date", "instrument"])
        partitions = ordered.partition_by("date", maintain_order=True, include_key=True)
        bars_by_day: dict[date, pl.DataFrame] = {}
        # history 前缀切片：bar_dates 为有行情的日期（升序），history_end 为对应的行偏移。
        bar_dates: list[date] = []
        history_end: list[int] = []
        offset = 0
        for frame in partitions:
            day = frame["date"][0]
            offset += frame.height
            bars_by_day[day] = frame
            bar_dates.append(day)
            history_end.append(offset)

        open_days = _open_days(calendar, start, end)

        # 公司行为按日预分组：只有除权日才需要扫行为表。
        action_days: set[date] = (
            set(actions["date"].to_list()) if "date" in actions.columns else set()
        )

        account = Account(cash=initial_cash)
        last_price: dict[str, float] = {}
        pending: list[Order] = []
        nav_rows: list[dict[str, object]] = []
        fills: list[FillRecord] = []
        rejects: list[RejectRecord] = []
        corporate_details: list[CorporateActionDetail] = []

        for index, day in enumerate(open_days):
            day_bars = bars_by_day.get(day, EMPTY_BARS)

            # T+1 开盘前：昨日买入解冻。
            account.settle_new_day()

            # T+1 开盘：撮合 T 日收盘生成的订单。
            report = self.broker.execute(pending, day_bars, account)
            pending = []
            _record(report, day, fills, rejects)

            # T+1 日终：公司行为（除权日 = day），除权前市值取最近可得价。
            if day in action_days:
                corporate_details.extend(
                    apply_corporate_actions(account, actions, day, last_price)
                )

            # T+1 日终：更新最近可得价（停牌持仓前向填充）。
            last_price.update(_close_map(day_bars, list(account.positions)))

            # T+1 日终：记净值与换手。
            market_value = account.market_value(last_price)
            nav = account.cash + market_value
            traded_amount = report.buy_amount + report.sell_amount
            nav_rows.append(
                {
                    "date": day,
                    "cash": account.cash,
                    "market_value": market_value,
                    "nav": nav,
                    "traded_amount": traded_amount,
                    "turnover": traded_amount / nav if nav > 0.0 else 0.0,
                }
            )

            # T 日收盘：生成次日订单（区间末尾没有下一个开市日，跳过）。
            if index + 1 < len(open_days):
                pending = _signal_orders(
                    signal_fn,
                    day,
                    _history(ordered, bar_dates, history_end, day),
                )

        return BacktestResult(
            nav=pl.DataFrame(nav_rows, schema=NAV_SCHEMA),
            fills=fills,
            rejects=rejects,
            corporate_actions=corporate_details,
            initial_cash=initial_cash,
        )


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------


def _open_days(calendar: pl.DataFrame, start: date, end: date) -> list[date]:
    """取 ``[start, end]`` 内的开市日，按日期升序。"""
    _check_columns(calendar, ("date", "is_open"), "交易日历")
    if calendar.height == 0:
        return []
    days = calendar.filter(
        pl.col("is_open").fill_null(False)
        & (pl.col("date") >= start)
        & (pl.col("date") <= end)
    )
    return days.sort("date")["date"].to_list()


def _record(
    report: ExecutionReport,
    day: date,
    fills: list[FillRecord],
    rejects: list[RejectRecord],
) -> None:
    """把 broker 的当次报告打上日期后并入流水。"""
    for fill in report.fills:
        fills.append(
            FillRecord(
                day, fill.instrument, fill.side, fill.volume, fill.price, fill.fee
            )
        )
    for reject in report.rejects:
        rejects.append(
            RejectRecord(
                day,
                reject.instrument,
                reject.side,
                reject.reason,
                reject.requested,
                reject.detail,
            )
        )


def _signal_orders(signal_fn: SignalFn, day: date, history: pl.DataFrame) -> list[Order]:
    """调用 ``signal_fn`` 并校验返回类型。"""
    orders = list(signal_fn(day, history))
    for order in orders:
        if not isinstance(order, Order):
            raise TypeError(
                f"signal_fn 必须返回 Order 列表，收到 {type(order).__name__}"
            )
    return orders


def _history(
    ordered: pl.DataFrame,
    bar_dates: list[date],
    history_end: list[int],
    day: date,
) -> pl.DataFrame:
    """取 ``date <= day`` 的行情前缀（Arrow 切片，零拷贝）。

    某日全市场都没有行情时 ``bar_dates`` 里没有该日期，用二分找到不晚于 ``day`` 的最后
    一个行情日，保证 ``signal_fn`` 的输入仍然只含历史数据。
    """
    position = bisect_right(bar_dates, day)
    if position == 0:
        return ordered.slice(0, 0)
    return ordered.slice(0, history_end[position - 1])


def _close_map(day_bars: pl.DataFrame, instruments: Sequence[str]) -> dict[str, float]:
    """取当日各持仓证券的估值价：优先收盘价，缺失时退回开盘价。

    当日无行情的证券不出现在返回字典里，由调用方沿用最近可得价（停牌前向填充）。
    """
    if day_bars.height == 0 or not instruments:
        return {}
    columns = ["instrument"] + [
        name for name in ("close", "open") if name in day_bars.columns
    ]
    window = day_bars.filter(pl.col("instrument").is_in(list(instruments))).select(
        columns
    )
    prices: dict[str, float] = {}
    for row in window.iter_rows(named=True):
        price = _finite(row.get("close"))
        if price <= 0.0:
            price = _finite(row.get("open"))
        if price > 0.0:
            prices[str(row["instrument"])] = price
    return prices


def _finite(value: object) -> float:
    """把可为 None / NaN / inf 的数值归一化为有限浮点，缺省 0.0。"""
    if value is None:
        return 0.0
    number = float(value)  # type: ignore[arg-type]
    return number if math.isfinite(number) else 0.0


def _check_columns(df: pl.DataFrame, required: tuple[str, ...], name: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{name}缺少列: {missing}")


__all__ = [
    "BAR_COLUMNS",
    "EMPTY_BARS",
    "NAV_SCHEMA",
    "BacktestEngine",
    "BacktestResult",
    "FillRecord",
    "RejectRecord",
    "SignalFn",
]
