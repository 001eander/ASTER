"""A 股费用模型：佣金 / 印花税 / 滑点。

口径（2026-09 现行）
--------------------
- **佣金**：买卖双边收取，费率默认万 2.5（``0.00025``），单笔最低 5 元。
  一笔委托的名义金额无论多小，佣金都不低于 5 元。
- **印花税**：仅卖出收取，默认 ``0.0005``（2023-08-28 起由千 1 减半为万 5）。
- **滑点**：默认 10bp（``0.001``）。体现在**成交价**上：买入加价、卖出减价，
  由 :meth:`FeeModel.execution_price` 给出；:class:`Fee` 里的 ``slippage`` 字段是
  按名义金额折算出的滑点成本，用于归因与展示。

金额约定
--------
:meth:`FeeModel.buy_cost` / :meth:`FeeModel.sell_cost` 的 ``notional`` 是**成交名义金额**
（股数 × 成交价，成交价由 :meth:`execution_price` 得到）。返回的 :class:`Fee` 中：

- ``commission`` / ``stamp_tax`` 是现金费用，账户扣款时应使用 :attr:`Fee.cash_cost`
  （两者之和）；
- ``slippage`` 已反映在成交价里，**不重复计入现金**；
- ``total`` 是三项之和，代表这笔交易相对未滑点参考价的全部交易成本，仅用于口径统计。

即：现金支出 = ``notional + Fee.cash_cost``，交易成本 = ``Fee.total``。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Side = Literal["buy", "sell"]

# ---------------------------------------------------------------------------
# 配置：默认费用参数（模块顶部集中，便于按账户覆盖）
# ---------------------------------------------------------------------------

#: 佣金费率（双边），万 2.5。
DEFAULT_COMMISSION_RATE: float = 0.00025
#: 单笔最低佣金，元。
DEFAULT_MIN_COMMISSION: float = 5.0
#: 印花税率（仅卖出），万 5。
DEFAULT_STAMP_TAX_RATE: float = 0.0005
#: 滑点，基点（1bp = 0.0001），默认 10bp。
DEFAULT_SLIPPAGE_BP: float = 10.0

#: 基点换算系数。
BP_PER_UNIT: float = 10_000.0

_VALID_SIDES: frozenset[str] = frozenset({"buy", "sell"})


@dataclass(frozen=True)
class Fee:
    """一笔交易的结构化费用（元）。

    ``total`` 为三项之和（含滑点，用于成本归因）；``cash_cost`` 为需要从现金中
    实际扣减的现金费用（佣金 + 印花税），滑点已包含在成交价里，不重复扣。
    """

    commission: float = 0.0
    stamp_tax: float = 0.0
    slippage: float = 0.0

    @property
    def cash_cost(self) -> float:
        """现金费用合计：佣金 + 印花税。"""
        return self.commission + self.stamp_tax

    @property
    def total(self) -> float:
        """全部交易成本：佣金 + 印花税 + 滑点。"""
        return self.commission + self.stamp_tax + self.slippage


@dataclass(frozen=True)
class FeeModel:
    """A 股费用模型。默认取值见模块顶部常量。"""

    commission_rate: float = DEFAULT_COMMISSION_RATE
    min_commission: float = DEFAULT_MIN_COMMISSION
    stamp_tax_rate: float = DEFAULT_STAMP_TAX_RATE
    slippage_bp: float = DEFAULT_SLIPPAGE_BP

    # -- 价格 ---------------------------------------------------------------

    @property
    def slippage_rate(self) -> float:
        """滑点比例（10bp -> 0.001）。"""
        return self.slippage_bp / BP_PER_UNIT

    def execution_price(self, side: Side, price: float) -> float:
        """含滑点的成交价：买入上浮、卖出下压。"""
        _check_side(side)
        if side == "buy":
            return price * (1.0 + self.slippage_rate)
        return price * (1.0 - self.slippage_rate)

    # -- 费用 ---------------------------------------------------------------

    def commission(self, notional: float) -> float:
        """佣金：``notional × 费率``，单笔不低于 ``min_commission``；无成交为 0。"""
        if notional <= 0.0:
            return 0.0
        return max(notional * self.commission_rate, self.min_commission)

    def buy_cost(self, notional: float) -> Fee:
        """买入费用：佣金（双边）+ 滑点，无印花税。"""
        return Fee(
            commission=self.commission(notional),
            stamp_tax=0.0,
            slippage=notional * self.slippage_rate,
        )

    def sell_cost(self, notional: float) -> Fee:
        """卖出费用：佣金（双边）+ 印花税 + 滑点。"""
        return Fee(
            commission=self.commission(notional),
            stamp_tax=notional * self.stamp_tax_rate,
            slippage=notional * self.slippage_rate,
        )


def _check_side(side: str) -> None:
    if side not in _VALID_SIDES:
        raise ValueError(f"未知交易方向: {side!r}，应为 'buy' 或 'sell'")


__all__ = [
    "BP_PER_UNIT",
    "DEFAULT_COMMISSION_RATE",
    "DEFAULT_MIN_COMMISSION",
    "DEFAULT_SLIPPAGE_BP",
    "DEFAULT_STAMP_TAX_RATE",
    "Fee",
    "FeeModel",
    "Side",
]
