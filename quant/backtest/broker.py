"""broker 撮合：T+1 / 涨跌停 / 整手 / 停牌 / 现金缩单。

定位（架构 3.5）
----------------
:class:`Broker` 接收调用方（回测引擎 / 调仓流程）给出的目标订单，按**当日行情**
撮合并直接改动 :class:`~quant.backtest.account.Account`。它不做调度、不碰 IO，
只负责「这笔单能不能成、按什么价成、成交多少」。

无前视口径
----------
撮合只用 T+1 当天的 ``open`` 与 ``limit_up`` / ``limit_down``，不读 ``close``：
T 日收盘后生成订单，T+1 开盘撮合，涨跌停判定基于 T+1 开盘价与预计算的涨跌停价。

规则口径
--------
- **停牌**：当日行情里没有该证券的行，或该行 ``open`` 缺失 / 非正，判 SUSPENDED。
- **涨跌停（严格版）**：买单开盘价触及涨停（``open >= limit_up``）拒，卖单触及跌停
  （``open <= limit_down``）拒；``limit_*`` 为 null（新股首日等）时跳过该检查。
  浮点比较留 :data:`PRICE_EPSILON` 容差。
- **整手**：主板 / 创业板 / 北交所买入按 100 股整数倍向下取整；科创板（688 / 689）
  买入最低 200 股，超过 200 股部分按 1 股递增。卖出不卡整手，允许零股出清，
  成交股数取 ``min(订单量, sellable)``。
- **T+1 冻结**：卖出先截断到 ``sellable``；截断后为零（含无持仓）才拒 T1_FROZEN，
  超出部分按部分成交处理，不整单拒。
- **现金不足缩单**：买单先按整手取整，再按费用后总额向下试探到买得起的最大整手量；
  缩到不足一手则拒 CASH_SHORT。
- **成交价**：:meth:`FeeModel.execution_price`，买入上浮、卖出下压（含滑点）；
  账户扣款用 :attr:`Fee.cash_cost`，滑点已含在成交价里。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import polars as pl

from quant.backtest.account import CASH_EPSILON, Account
from quant.backtest.fee import Fee, FeeModel, Side
from quant.data.schema import board_of

# ---------------------------------------------------------------------------
# 配置：整手与价格容差常量
# ---------------------------------------------------------------------------

#: 主板 / 创业板 / 北交所买入最小申报单位：100 股整数倍。
MAIN_LOT_SIZE: int = 100
#: 科创板买入最低申报数量（股），超过部分按 1 股递增。
KCB_MIN_BUY_VOLUME: int = 200
#: 科创板整手递增步长（股）。
KCB_LOT_STEP: int = 1
#: 价格比较容差（元）：涨跌停触价判定吸收浮点误差。
PRICE_EPSILON: float = 1e-9

#: 板块代码，科创板按 1 股递增，其余按主板的 100 股整数倍。
_KCB_BOARD: str = "kcb"

_VALID_SIDES: frozenset[str] = frozenset({"buy", "sell"})


# ---------------------------------------------------------------------------
# 订单与结果类型
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Order:
    """一笔调仓订单。

    ``volume`` 为**意愿股数**（买入时按整手规则取整前的目标量）。
    """

    instrument: str
    side: Side
    volume: int

    def __post_init__(self) -> None:
        if self.side not in _VALID_SIDES:
            raise ValueError(f"未知交易方向: {self.side!r}，应为 'buy' 或 'sell'")
        if self.volume <= 0:
            raise ValueError(f"订单股数必须为正: {self.volume}")


@dataclass(frozen=True)
class Fill:
    """一笔成交。``price`` 为含滑点的成交价，``fee`` 为对应费用。

    账户现金支出为 ``volume × price + fee.cash_cost``（滑点已在 ``price`` 里）。
    """

    instrument: str
    side: Side
    volume: int
    price: float
    fee: Fee

    @property
    def notional(self) -> float:
        """成交名义金额（元）。"""
        return self.volume * self.price


class RejectReason(str, Enum):
    """拒单原因。"""

    #: 停牌 / 当日无行情（含缺失或非法开盘价）。
    SUSPENDED = "suspended"
    #: 买单开盘价触及涨停。
    LIMIT_UP = "limit_up"
    #: 卖单开盘价触及跌停。
    LIMIT_DOWN = "limit_down"
    #: 可卖股数不足（T+1 冻结或持仓不足）。
    T1_FROZEN = "t1_frozen"
    #: 整手取整后不足一手。
    LOT_SIZE = "lot_size"
    #: 缩单后仍买不起一手。
    CASH_SHORT = "cash_short"


@dataclass(frozen=True)
class Reject:
    """一笔被拒的订单。``requested`` 为原意愿股数，``detail`` 便于排查。"""

    instrument: str
    side: Side
    reason: RejectReason
    requested: int = 0
    detail: str = ""


@dataclass
class ExecutionReport:
    """一次 :meth:`Broker.execute` 的成交与拒单清单及汇总金额。"""

    fills: list[Fill] = field(default_factory=list)
    rejects: list[Reject] = field(default_factory=list)

    @property
    def buy_amount(self) -> float:
        """买入成交名义金额合计（元）。"""
        return sum(fill.notional for fill in self.fills if fill.side == "buy")

    @property
    def sell_amount(self) -> float:
        """卖出成交名义金额合计（元）。"""
        return sum(fill.notional for fill in self.fills if fill.side == "sell")

    @property
    def cash_cost(self) -> float:
        """现金费用合计（佣金 + 印花税，元）。"""
        return sum(fill.fee.cash_cost for fill in self.fills)

    @property
    def total_cost(self) -> float:
        """全部交易成本合计（含滑点，元），仅用于口径统计。"""
        return sum(fill.fee.total for fill in self.fills)

    @property
    def net_cash_flow(self) -> float:
        """净现金流（元）：卖出收入 − 买入支出 − 现金费用。"""
        return self.sell_amount - self.buy_amount - self.cash_cost

    def filled_volume(self, instrument: str) -> int:
        """某证券本次成交股数（买卖合并，用于对账）。"""
        return sum(fill.volume for fill in self.fills if fill.instrument == instrument)


# ---------------------------------------------------------------------------
# 整手规则
# ---------------------------------------------------------------------------


def buy_lot_step(instrument: str) -> int:
    """买入缩单时的最小递减步长（股）：科创板 1，其余 100。"""
    return KCB_LOT_STEP if board_of(instrument) == _KCB_BOARD else MAIN_LOT_SIZE


def round_buy_volume(instrument: str, volume: int) -> int:
    """把买入意愿股数按板块整手规则**向下取整**。

    返回 0 表示取整后不足一手（主板不足 100 股，科创板不足 200 股）。
    """
    if volume <= 0:
        return 0
    if board_of(instrument) == _KCB_BOARD:
        return volume if volume >= KCB_MIN_BUY_VOLUME else 0
    return (volume // MAIN_LOT_SIZE) * MAIN_LOT_SIZE


# ---------------------------------------------------------------------------
# broker
# ---------------------------------------------------------------------------


class Broker:
    """按当日行情撮合订单，成交直接落账到 :class:`Account`。"""

    def __init__(self, fee_model: FeeModel) -> None:
        self.fee_model = fee_model

    def execute(
        self,
        orders: list[Order],
        bars: pl.DataFrame,
        account: Account,
    ) -> ExecutionReport:
        """依次撮合 ``orders``，就地更新 ``account``，返回成交报告。

        ``bars`` 为**当个交易日**各证券的行情行（至少含 ``instrument`` 与
        ``open``，可选 ``limit_up`` / ``limit_down``）。订单按给定顺序串行处理，
        因此同批次买单共享同一份实时现金。
        """
        report = ExecutionReport()
        bar_by_instrument = _index_bars(bars)
        for order in orders:
            if order.side == "buy":
                self._execute_buy(order, bar_by_instrument, account, report)
            else:
                self._execute_sell(order, bar_by_instrument, account, report)
        return report

    # -- 买单 ---------------------------------------------------------------

    def _execute_buy(
        self,
        order: Order,
        bar_by_instrument: dict[str, dict[str, Any]],
        account: Account,
        report: ExecutionReport,
    ) -> None:
        row = bar_by_instrument.get(order.instrument)
        open_price = _open_price(row)
        if open_price is None:
            report.rejects.append(
                _reject(order, RejectReason.SUSPENDED, "当日无行情或开盘价缺失")
            )
            return

        limit_up = _limit_value(row, "limit_up")
        if limit_up is not None and open_price >= limit_up - PRICE_EPSILON:
            report.rejects.append(
                _reject(order, RejectReason.LIMIT_UP, f"开盘 {open_price} 触及涨停 {limit_up}")
            )
            return

        target = round_buy_volume(order.instrument, order.volume)
        if target == 0:
            report.rejects.append(
                _reject(order, RejectReason.LOT_SIZE, "取整后不足一手")
            )
            return

        price = self.fee_model.execution_price("buy", open_price)
        volume = _affordable_volume(
            order.instrument, target, price, account.cash, self.fee_model
        )
        if volume == 0:
            report.rejects.append(
                _reject(order, RejectReason.CASH_SHORT, "现金不足最小一手")
            )
            return

        fee = self.fee_model.buy_cost(volume * price)
        account.buy(order.instrument, volume, price, fee)
        report.fills.append(Fill(order.instrument, "buy", volume, price, fee))

    # -- 卖单 ---------------------------------------------------------------

    def _execute_sell(
        self,
        order: Order,
        bar_by_instrument: dict[str, dict[str, Any]],
        account: Account,
        report: ExecutionReport,
    ) -> None:
        row = bar_by_instrument.get(order.instrument)
        open_price = _open_price(row)
        if open_price is None:
            report.rejects.append(
                _reject(order, RejectReason.SUSPENDED, "当日无行情或开盘价缺失")
            )
            return

        limit_down = _limit_value(row, "limit_down")
        if limit_down is not None and open_price <= limit_down + PRICE_EPSILON:
            report.rejects.append(
                _reject(
                    order, RejectReason.LIMIT_DOWN, f"开盘 {open_price} 触及跌停 {limit_down}"
                )
            )
            return

        position = account.position(order.instrument)
        sellable = position.sellable if position is not None else 0
        volume = min(order.volume, sellable)
        if volume <= 0:
            report.rejects.append(
                _reject(order, RejectReason.T1_FROZEN, f"可卖 {sellable} 股")
            )
            return

        price = self.fee_model.execution_price("sell", open_price)
        fee = self.fee_model.sell_cost(volume * price)
        account.sell(order.instrument, volume, price, fee)
        report.fills.append(Fill(order.instrument, "sell", volume, price, fee))


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------


def _reject(order: Order, reason: RejectReason, detail: str) -> Reject:
    return Reject(order.instrument, order.side, reason, requested=order.volume, detail=detail)


def _index_bars(bars: pl.DataFrame) -> dict[str, dict[str, Any]]:
    """把当日行情按 instrument 建索引，只取撮合需要的列。"""
    available = [
        name
        for name in ("instrument", "open", "limit_up", "limit_down")
        if name in bars.columns
    ]
    if "instrument" not in available or "open" not in available:
        raise ValueError("bars 至少需要 instrument 与 open 两列")
    if bars.height == 0:
        return {}
    return {
        str(row["instrument"]): row
        for row in bars.select(available).iter_rows(named=True)
    }


def _open_price(row: dict[str, Any] | None) -> float | None:
    """取有效开盘价；无行、缺失、非有限或非正一律视为不可交易。"""
    if row is None:
        return None
    value = row.get("open")
    if value is None:
        return None
    price = float(value)
    if not math.isfinite(price) or price <= 0.0:
        return None
    return price


def _limit_value(row: dict[str, Any] | None, column: str) -> float | None:
    """取涨跌停价；无行、缺失或非有限时为 None（跳过该项检查）。"""
    if row is None:
        return None
    value = row.get(column)
    if value is None:
        return None
    limit = float(value)
    return limit if math.isfinite(limit) else None


def _affordable_volume(
    instrument: str,
    volume: int,
    price: float,
    cash: float,
    fee_model: FeeModel,
) -> int:
    """在现金约束下把买入股数按整手步长下压，返回可买股数（0 表示买不起）。"""
    step = buy_lot_step(instrument)
    # 先用忽略费用的上界快速下压，再逐档校验含费总额。
    upper = int((cash + CASH_EPSILON) / price)
    if upper < volume:
        volume = round_buy_volume(instrument, upper)

    while volume > 0:
        volume = round_buy_volume(instrument, volume)
        if volume == 0:
            return 0
        fee = fee_model.buy_cost(volume * price)
        if volume * price + fee.cash_cost <= cash + CASH_EPSILON:
            return volume
        volume -= step
    return 0


__all__ = [
    "KCB_LOT_STEP",
    "KCB_MIN_BUY_VOLUME",
    "MAIN_LOT_SIZE",
    "PRICE_EPSILON",
    "Broker",
    "ExecutionReport",
    "Fill",
    "Order",
    "Reject",
    "RejectReason",
    "buy_lot_step",
    "round_buy_volume",
]
