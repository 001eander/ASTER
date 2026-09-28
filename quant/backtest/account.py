"""账户模拟器：现金 + 持仓（``volume`` / ``sellable`` 分离）。

设计目标（架构 3.5 节）
----------------------
同一套账户逻辑既服务**回测批量推进**，也服务**生产虚拟账户每日单步推进**：

- 批量回测：引擎按历史日推，逐日调用 :meth:`Account.buy` / :meth:`Account.sell`，
  日终调用 :meth:`Account.settle_new_day` 解冻、:meth:`Account.apply_corporate_action`
  处理分红送转，再用 :meth:`Account.nav` 记录净值。
- 生产跟单：每天在处理完调仓与成交后，用 :meth:`Account.to_dict` 落盘，
  下一日启动时 :meth:`Account.from_dict` 恢复，状态即完整可续。

因此账户本身不做调度、不碰 IO：只维护现金与持仓的会计恒等式，把「什么时候买、
买多少、按什么价」交给调用方（回测引擎 / broker）。

持仓约定
--------
- ``volume``：总股数；``sellable``：可卖股数。T+1 冻结体现在二者之差，当日买入只加
  ``volume``，次日 :meth:`settle_new_day` 后 ``sellable`` 追平 ``volume``。
- ``cost_basis``：该持仓的**摊薄总成本**（元，含买入费用）。卖出时按股数比例结转，
  分红送转时总额不变，因此每股成本随送转自然摊薄。
- 现金不允许为负；买入前由调用方（broker）负责缩单，账户内部超支会抛
  :class:`InsufficientCashError`，卖出超过 ``sellable`` 抛
  :class:`InsufficientPositionError`。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from quant.backtest.fee import Fee

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

#: 现金比较容差，吸收浮点误差（元）。
CASH_EPSILON: float = 1e-6


# ---------------------------------------------------------------------------
# 异常
# ---------------------------------------------------------------------------


class AccountError(Exception):
    """账户操作非法。"""


class InsufficientCashError(AccountError):
    """现金不足以完成买入。"""


class InsufficientPositionError(AccountError):
    """卖出股数超过可卖数量或持仓不存在。"""


# ---------------------------------------------------------------------------
# 持仓
# ---------------------------------------------------------------------------


@dataclass
class Position:
    """单只证券的持仓。

    ``sellable`` 不超过 ``volume``；``cost_basis`` 为摊薄总成本（元）。
    """

    instrument: str
    volume: int = 0
    sellable: int = 0
    cost_basis: float = 0.0

    @property
    def avg_cost(self) -> float:
        """每股摊薄成本（元/股），空仓为 0。"""
        if self.volume <= 0:
            return 0.0
        return self.cost_basis / self.volume

    def to_dict(self) -> dict[str, Any]:
        return {
            "instrument": self.instrument,
            "volume": self.volume,
            "sellable": self.sellable,
            "cost_basis": self.cost_basis,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Position":
        return cls(
            instrument=str(data["instrument"]),
            volume=int(data["volume"]),
            sellable=int(data["sellable"]),
            cost_basis=float(data["cost_basis"]),
        )


# ---------------------------------------------------------------------------
# 账户
# ---------------------------------------------------------------------------


@dataclass
class Account:
    """现金 + 持仓的账户模拟器。"""

    cash: float = 0.0
    positions: dict[str, Position] = field(default_factory=dict)

    # -- 查询 ---------------------------------------------------------------

    def position(self, instrument: str) -> Position | None:
        """返回持仓对象，未持有时为 ``None``。"""
        return self.positions.get(instrument)

    def market_value(self, prices: Mapping[str, float]) -> float:
        """按给定价格对全部持仓估值（元）。缺少持仓价格即报错，避免静默按 0。"""
        total = 0.0
        for instrument, position in self.positions.items():
            if instrument not in prices:
                raise KeyError(f"估值缺少 {instrument} 的价格")
            total += position.volume * prices[instrument]
        return total

    def nav(self, prices: Mapping[str, float]) -> float:
        """净资产 = 现金 + 持仓市值。"""
        return self.cash + self.market_value(prices)

    # -- 交易 ---------------------------------------------------------------

    def buy(self, instrument: str, volume: int, price: float, fees: Fee) -> None:
        """买入：扣 ``volume × price`` 与现金费用，当日买入不计入 ``sellable``。

        ``price`` 为成交价（含滑点，见 :meth:`FeeModel.execution_price`）。现金费用
        取 :attr:`Fee.cash_cost`（佣金 + 印花税），滑点已含在成交价中不重复扣。
        现金不足抛 :class:`InsufficientCashError`。
        """
        if volume <= 0:
            raise AccountError(f"买入股数必须为正: {volume}")
        if price <= 0:
            raise AccountError(f"买入价格必须为正: {price}")

        cash_out = volume * price + fees.cash_cost
        if cash_out > self.cash + CASH_EPSILON:
            raise InsufficientCashError(
                f"买入 {instrument} 需 {cash_out:.2f} 元，现金仅 {self.cash:.2f} 元"
            )

        self.cash -= cash_out
        position = self.positions.get(instrument)
        if position is None:
            # 当日买入只加 volume，sellable 保持 0（T+1 冻结）。
            self.positions[instrument] = Position(
                instrument=instrument,
                volume=volume,
                sellable=0,
                cost_basis=cash_out,
            )
        else:
            position.volume += volume
            position.cost_basis += cash_out

    def sell(self, instrument: str, volume: int, price: float, fees: Fee) -> float:
        """卖出：只能卖 ``sellable`` 部分，加现金并扣现金费用，返回净收入。

        ``price`` 为成交价（含滑点）。净收入 = ``volume × price − Fee.cash_cost``。
        持仓不存在或股数超过 ``sellable`` 抛 :class:`InsufficientPositionError`。
        """
        if volume <= 0:
            raise AccountError(f"卖出股数必须为正: {volume}")

        position = self.positions.get(instrument)
        if position is None:
            raise InsufficientPositionError(f"无 {instrument} 持仓，无法卖出")
        if volume > position.sellable:
            raise InsufficientPositionError(
                f"{instrument} 可卖 {position.sellable} 股，试图卖出 {volume} 股"
            )

        proceeds = volume * price - fees.cash_cost
        self.cash += proceeds

        # 按股数比例结转成本，剩余持仓总成本不变口径。
        position.cost_basis -= position.avg_cost * volume
        position.volume -= volume
        position.sellable -= volume
        if position.volume <= 0:
            del self.positions[instrument]
        return proceeds

    # -- 日切与公司行为 -----------------------------------------------------

    def settle_new_day(self) -> None:
        """日终 / 次日开盘前调用：昨日买入解冻，``sellable`` 追平 ``volume``。"""
        for position in self.positions.values():
            position.sellable = position.volume

    def apply_corporate_action(
        self,
        instrument: str,
        cash_per_share: float = 0.0,
        share_per_share: float = 0.0,
    ) -> None:
        """除权日调整持仓：现金分红入现金，送转股增加数量并按比例摊薄成本。

        - 现金分红：``现金 += cash_per_share × volume``（税前口径，红利税不在此处理）。
        - 送转：新增股数 = ``round(volume × share_per_share)``，同时计入 ``volume`` 与
          ``sellable``；``cost_basis`` 总额不变，每股成本随股数增加而摊薄。

        未持有该证券时静默跳过。
        """
        position = self.positions.get(instrument)
        if position is None:
            return

        if cash_per_share:
            self.cash += cash_per_share * position.volume

        if share_per_share:
            added = int(round(position.volume * share_per_share))
            position.volume += added
            position.sellable += added

    # -- 序列化（为 #15 虚拟账户持久化预留）--------------------------------

    def to_dict(self) -> dict[str, Any]:
        """导出为可 JSON 序列化的状态字典。"""
        return {
            "cash": self.cash,
            "positions": [position.to_dict() for position in self.positions.values()],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Account":
        """从 :meth:`to_dict` 的状态字典恢复账户。"""
        positions = {
            position.instrument: position
            for position in (Position.from_dict(item) for item in data["positions"])
        }
        return cls(cash=float(data["cash"]), positions=positions)


__all__ = [
    "CASH_EPSILON",
    "Account",
    "AccountError",
    "InsufficientCashError",
    "InsufficientPositionError",
    "Position",
]
