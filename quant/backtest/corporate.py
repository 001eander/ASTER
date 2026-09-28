"""公司行为：除权日的分红送转处理。

口径
----
``CORPORATE_ACTIONS`` 的 ``date`` 是**除权除息日**。当天日终对持仓做两件事（见
:meth:`quant.backtest.account.Account.apply_corporate_action`）：

- 现金分红：``现金 += cash_per_share × 除权前股数``（税前口径，红利税不在回测处理）。
- 送转股：``股数 += round(除权前股数 × share_per_share)``；成本总额不变，每股成本随
  股数增加自然摊薄。

估值连续性：行情表存的是未复权价，除权日收盘价会自然下移；分红送转把对应的现金与
股数补回账户，因此 ``nav`` 在除权日不产生假跳变。

编排顺序
--------
:func:`apply_corporate_actions` 对当日除权且有持仓的证券：先记录除权前市值，再调账户，
最后汇总明细。``prices`` 是**除权前价格**（引擎传最近可得价，通常是除权日前一交易日
收盘价），只用于明细里的 ``market_value_before``，不参与会计恒等式。

结算基数取除权日日终的实际持仓（当日成交之后）。A 股股权登记日在除权日前一天，若除权
日当天卖出，本实现按剩余持仓计提分红；如需按登记日持仓计提，由调用方在引擎层快照。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Mapping

import polars as pl

from quant.backtest.account import Account

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

#: 公司行为表（CORPORATE_ACTIONS）的必要列。
ACTION_COLUMNS: tuple[str, ...] = (
    "date",
    "instrument",
    "cash_per_share",
    "share_per_share",
)


# ---------------------------------------------------------------------------
# 处理明细
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorporateActionDetail:
    """一次公司行为对账户的实际影响（供引擎记日志 / 简报）。

    ``volume_before`` / ``volume_after`` 为除权前后股数，``price_before`` 为除权前价格，
    ``market_value_before`` 为按该价格估的除权前市值。
    """

    date: date
    instrument: str
    cash_per_share: float
    share_per_share: float
    price_before: float
    volume_before: int
    volume_after: int
    shares_added: int
    cash_received: float
    market_value_before: float


# ---------------------------------------------------------------------------
# 查询
# ---------------------------------------------------------------------------


def actions_on(actions: pl.DataFrame, day: date) -> pl.DataFrame:
    """返回 ``day``（除权日）当天的公司行为，按 ``instrument`` 排序。

    当日无行为时返回空表（保留原列）。
    """
    missing = [column for column in ACTION_COLUMNS if column not in actions.columns]
    if missing:
        raise ValueError(f"公司行为表缺少列: {missing}")
    return actions.filter(pl.col("date") == day).sort("instrument")


# ---------------------------------------------------------------------------
# 应用
# ---------------------------------------------------------------------------


def apply_corporate_actions(
    account: Account,
    actions: pl.DataFrame,
    day: date,
    prices: Mapping[str, float],
) -> list[CorporateActionDetail]:
    """对 ``day`` 除权、且账户持有的证券调整持仓，返回处理明细。

    未持有该证券的行、以及现金分红与送转都为 0 的行不产生明细。
    """
    details: list[CorporateActionDetail] = []
    if actions.height == 0:
        return details

    rows = actions_on(actions, day).select(
        "instrument", "cash_per_share", "share_per_share"
    )
    for row in rows.iter_rows(named=True):
        instrument = str(row["instrument"])
        position = account.position(instrument)
        if position is None:
            continue

        cash_per_share = _finite(row["cash_per_share"])
        share_per_share = _finite(row["share_per_share"])
        if cash_per_share == 0.0 and share_per_share == 0.0:
            continue

        volume_before = position.volume
        price_before = max(_finite(prices.get(instrument)), 0.0)
        account.apply_corporate_action(instrument, cash_per_share, share_per_share)
        details.append(
            CorporateActionDetail(
                date=day,
                instrument=instrument,
                cash_per_share=cash_per_share,
                share_per_share=share_per_share,
                price_before=price_before,
                volume_before=volume_before,
                volume_after=position.volume,
                shares_added=position.volume - volume_before,
                cash_received=cash_per_share * volume_before,
                market_value_before=price_before * volume_before,
            )
        )
    return details


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------


def _finite(value: object) -> float:
    """把可为 None / NaN / inf 的数值归一化为有限浮点，缺省 0.0。"""
    if value is None:
        return 0.0
    number = float(value)  # type: ignore[arg-type]
    return number if math.isfinite(number) else 0.0


__all__ = [
    "ACTION_COLUMNS",
    "CorporateActionDetail",
    "actions_on",
    "apply_corporate_actions",
]
