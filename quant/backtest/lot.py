"""A 股整手（lot）规则：权重取整与 broker 撮合共用的最小口径。

规则（与 issue #10 broker 对齐）：

- 主板 ``main``、创业板 ``cyb``：买入数量为 100 股整数倍。
- 科创板 ``kcb``（688/689）：单笔买入不低于 200 股，超过部分以 1 股递增。
- 北交所 ``bj``：单笔买入不低于 100 股，超过部分以 1 股递增。

数量为 0 表示不持有 / 不交易，视为合法。

TODO(#10 / #12 统一)：broker 与 roundlot 由并行任务各自实现了一份等价规则，
后续以本模块为唯一口径，broker 改为 import 这里。本模块不依赖 broker。
"""
from __future__ import annotations

import math

from quant.data.schema import Board

#: 单笔买入的最小数量。
MIN_VOLUME: dict[Board, int] = {"main": 100, "cyb": 100, "kcb": 200, "bj": 100}

#: 超过最小数量后的递增单位（1 表示可任意整数递增）。
LOT_STEP: dict[Board, int] = {"main": 100, "cyb": 100, "kcb": 1, "bj": 1}


def min_volume(board: Board) -> int:
    """返回该板块单笔买入的最小数量。"""
    return MIN_VOLUME[board]


def lot_step(board: Board) -> int:
    """返回该板块超过最小数量后的递增单位。"""
    return LOT_STEP[board]


def round_lot_down(volume: float, board: Board) -> int:
    """把目标股数按板块整手规则**向下**取整为可执行数量。

    向下取整只会少买，不会超过目标市值，从而不会透支现金。
    """
    if volume <= 0 or not math.isfinite(volume):
        return 0
    whole = math.floor(volume)
    minimum = MIN_VOLUME[board]
    if whole < minimum:
        return 0
    step = LOT_STEP[board]
    if step == 1:
        return whole
    return (whole // step) * step


def is_valid_volume(volume: int, board: Board) -> bool:
    """判断数量是否符合该板块整手规则（0 视为合法）。"""
    if volume < 0:
        return False
    if volume == 0:
        return True
    minimum = MIN_VOLUME[board]
    if volume < minimum:
        return False
    step = LOT_STEP[board]
    return volume % step == 0
