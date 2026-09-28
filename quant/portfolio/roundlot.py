"""权重 → 可执行股数（整手向下取整）与剩余现金估计。

输入
----
- ``weights``：目标权重 ``{instrument: weight}``，多头、之和应约等于 1（允许因
  剔除约零项略小于 1；超过 1 会报错，以保证不透支）。
- ``prices``：各证券最新可得价（T 日收盘或最近价；T+1 开盘价此时未知）。
- ``nav``：组合总市值（元）。
- ``boards``：可选 ``{instrument: Board}`` 映射，缺省按代码用
  :func:`quant.data.schema.board_of` 判定。

输出
----
``RoundLotResult.orders`` 为 polars 表，列
``instrument, board, price, target_value, volume, est_value``；
``cash_left = nav − Σ est_value`` 为剩余现金估计（元）。

口径
----
每票目标市值 ``target_value = w × nav``，按整手规则**向下**取整得 ``volume``
（规则见 :mod:`quant.backtest.lot`），``est_value = volume × price``。向下取整保证
``Σ est_value ≤ nav``，只留现金不透支，与架构 3.6 的 buffer 设计一致。
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import polars as pl

from quant.backtest.lot import round_lot_down
from quant.data.schema import Board, board_of

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

#: 权重之和允许超出 1 的容差；超出即报错，避免估出的股数透支现金。
WEIGHT_SUM_TOL: float = 1e-6

#: 订单表 schema。
ORDERS_SCHEMA = pl.Schema(
    {
        "instrument": pl.String,
        "board": pl.String,
        "price": pl.Float64,
        "target_value": pl.Float64,
        "volume": pl.Int64,
        "est_value": pl.Float64,
    }
)


# ---------------------------------------------------------------------------
# 结果对象
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoundLotResult:
    """取整结果。

    Attributes
    ----------
    orders:
        每只证券的整手目标，列见 :data:`ORDERS_SCHEMA`。
    invested_value:
        预计建仓市值 ``Σ volume × price``。
    cash_left:
        剩余现金估计 ``nav − invested_value``。
    """

    orders: pl.DataFrame
    invested_value: float
    cash_left: float


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------


def round_weights_to_lots(
    weights: Mapping[str, float],
    prices: Mapping[str, float],
    nav: float,
    *,
    boards: Mapping[str, Board] | None = None,
) -> RoundLotResult:
    """把目标权重转成整手股数，返回订单表与剩余现金估计。"""
    if not math.isfinite(nav) or nav <= 0:
        raise ValueError(f"nav 必须为正的有限值: {nav}")

    weight_sum = float(sum(weights.values()))
    if weight_sum > 1 + WEIGHT_SUM_TOL:
        raise ValueError(f"权重之和 {weight_sum:.6f} 超过 1，请先归一化")

    rows: list[dict[str, object]] = []
    invested = 0.0
    for instrument, weight in weights.items():
        if instrument not in prices:
            raise ValueError(f"缺少 {instrument} 的价格")
        price = float(prices[instrument])
        if not math.isfinite(price) or price <= 0:
            raise ValueError(f"{instrument} 的价格非法: {price}")
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"{instrument} 的权重非法: {weight}")

        board: Board = boards[instrument] if boards and instrument in boards else board_of(instrument)
        target_value = float(weight) * nav
        volume = round_lot_down(target_value / price, board)
        est_value = volume * price
        invested += est_value
        rows.append(
            {
                "instrument": instrument,
                "board": board,
                "price": price,
                "target_value": target_value,
                "volume": volume,
                "est_value": est_value,
            }
        )

    orders = pl.DataFrame(rows, schema=ORDERS_SCHEMA)
    return RoundLotResult(
        orders=orders,
        invested_value=invested,
        cash_left=nav - invested,
    )
