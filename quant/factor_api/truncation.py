"""截断重算检测：抽样检测日，把数据截断到该日重算因子，与全量结果对比，抓前视。

检测原理
    因子契约要求 ``compute(data)`` 只能使用传入数据里「当日及之前」的历史
    （见 :mod:`quant.factor_api.spec`）。如果因子在 ``t`` 日的结果偷偷用到了
    ``t`` 之后的行情，那么把输入截断到 ``t`` 之后再算，``t`` 日的结果就会与全量
    数据下的结果不同：全量输入包含 ``t`` 之后的未来行，截断输入只到 ``t``。
    于是「同一检测日在两份输入下的截面是否一致」就成为前视的行为级判据，
    无须读取因子源码。

    逐日全量重算代价高，因此等距抽 ``n_checks`` 个检测日。抽样用等距取点，
    不用随机数，保证同一输入每次得到同一结果。

预热期
    有窗口的因子在数据起步阶段受窗口不足影响，前 ``warmup`` 个交易日的值本就不稳定，
    检测日一律取在 ``dates[warmup:]``，避免把「窗口还没铺满」误判成前视。
    因子模块可声明模块级常量 ``WARMUP: int`` 覆盖默认预热期，读取方（#20 评估管线）
    负责读取该常量并传给 ``warmup`` 参数。

null 与浮点
    一边 null 另一边非 null，或某一证券只在一侧出现（另一侧整体缺行），都算不一致。
    两侧都有值时用 ``max(rtol * max(|a|, |b|), atol)`` 作容差，与 ``math.isclose``
    语义一致，容差内视为一致，用来吸收浮点计算路径上的噪声。
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import polars as pl

from quant.factor_api.schema import validate_output
from quant.factor_api.spec import FactorCompute

#: 默认预热交易日数：检测日避开因子起步的前 60 个交易日。
DEFAULT_WARMUP_DAYS: int = 60

#: 默认抽样检测日数。
DEFAULT_N_CHECKS: int = 5

#: 数值比较的相对容差。
DEFAULT_RTOL: float = 1e-9

#: 数值比较的绝对容差。
DEFAULT_ATOL: float = 1e-12

__all__ = [
    "DEFAULT_ATOL",
    "DEFAULT_N_CHECKS",
    "DEFAULT_RTOL",
    "DEFAULT_WARMUP_DAYS",
    "TruncationFailure",
    "TruncationResult",
    "check_truncation",
]


@dataclass(frozen=True)
class TruncationFailure:
    """某个检测日的截断重算不一致明细。

    :param date: 检测日。
    :param n_mismatch: 不一致的证券数。
    :param max_abs_diff: 两侧都有值的那些不一致行的最大绝对偏差；
        不一致全部由 null 或整行缺失造成时没有可计算的偏差，取 None。
    """

    date: dt.date
    n_mismatch: int
    max_abs_diff: float | None


@dataclass(frozen=True)
class TruncationResult:
    """截断重算检测的整体结果。

    :param ok: 全部检测日都通过时为 True。
    :param n_checks: 实际执行的检测日数；数据太短而无法抽样时为 0。
    :param failures: 不一致的检测日明细，按检测日升序。
    :param skipped_reason: 无法执行检测时的说明（如交易日数不足预热期），
        可执行检测时为 None。调用方据此区分「数据太短没法检」与「检出前视」：
        前者 ``ok is False`` 且 ``skipped_reason is not None``，
        后者 ``ok is False`` 且 ``failures`` 非空。
    """

    ok: bool
    n_checks: int
    failures: tuple[TruncationFailure, ...]
    skipped_reason: str | None = None


def _select_check_days(
    dates: list[dt.date], warmup: int, n_checks: int
) -> list[dt.date]:
    """从 ``dates[warmup:]`` 等距取 ``n_checks`` 个检测日。

    区间长度不足 ``n_checks`` 时取区间内全部日期；区间为空返回空列表。
    等距取点用整数下标线性插值，首尾都落在区间端点上。
    """
    candidates = dates[warmup:]
    if not candidates:
        return []
    if len(candidates) <= n_checks:
        return list(candidates)
    if n_checks == 1:
        return [candidates[len(candidates) // 2]]
    last = len(candidates) - 1
    return [candidates[round(i * last / (n_checks - 1))] for i in range(n_checks)]


def _compare_day(
    full: pl.DataFrame,
    recomputed: pl.DataFrame,
    day: dt.date,
    *,
    rtol: float,
    atol: float,
) -> TruncationFailure | None:
    """对比 ``full`` 与 ``recomputed`` 在 ``day`` 当天的截面，无差异返回 None。

    用 full outer join 对齐 ``instrument``：一侧整行缺失时该侧 ``*_present`` 为 null，
    与「一侧有值、另一侧为 null」一并判为不一致。数值行用
    ``max(rtol * max(|a|, |b|), atol)`` 作容差。
    """
    full_day = (
        full.filter(pl.col("date") == day)
        .select("instrument", pl.col("value").alias("full_value"))
        .with_columns(pl.lit(True).alias("full_present"))
    )
    re_day = (
        recomputed.filter(pl.col("date") == day)
        .select("instrument", pl.col("value").alias("re_value"))
        .with_columns(pl.lit(True).alias("re_present"))
    )
    joined = full_day.join(re_day, on="instrument", how="full", coalesce=True)

    both_present = pl.col("full_value").is_not_null() & pl.col("re_value").is_not_null()
    tol = pl.max_horizontal(
        pl.lit(rtol)
        * pl.max_horizontal(
            pl.col("full_value").abs(), pl.col("re_value").abs()
        ),
        pl.lit(atol),
    )
    not_close = (pl.col("full_value") - pl.col("re_value")).abs() > tol
    mismatch = (
        (pl.col("full_present").is_null() != pl.col("re_present").is_null())
        | (pl.col("full_value").is_null() != pl.col("re_value").is_null())
        | (both_present & not_close)
    ).fill_null(False)

    bad = joined.filter(mismatch)
    n_mismatch = bad.height
    if n_mismatch == 0:
        return None

    numeric = bad.filter(
        pl.col("full_value").is_not_null() & pl.col("re_value").is_not_null()
    )
    max_abs_diff: float | None
    if numeric.height == 0:
        max_abs_diff = None
    else:
        max_abs_diff = numeric.select(
            (pl.col("full_value") - pl.col("re_value")).abs().max()
        ).item()
    return TruncationFailure(date=day, n_mismatch=n_mismatch, max_abs_diff=max_abs_diff)


def check_truncation(
    compute: FactorCompute,
    data: pl.DataFrame,
    *,
    warmup: int = DEFAULT_WARMUP_DAYS,
    n_checks: int = DEFAULT_N_CHECKS,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> TruncationResult:
    """抽样截断重算，检测 ``compute`` 是否使用了检测日之后的数据。

    流程：全量算一次 ``full = compute(data)`` 并校验输出；在
    ``dates[warmup:]`` 里等距抽 ``n_checks`` 个检测日；对每个检测日 ``t``，
    用 ``data.filter(date <= t)`` 重算一次，与 ``full`` 在 ``t`` 当天的截面
    逐证券对比。全部检测日一致则 ``ok=True``。

    :param compute: 因子计算函数，须满足 :mod:`quant.factor_api.spec` 的契约。
    :param data: 因子输入面板，列见 :data:`quant.factor_api.spec.FACTOR_INPUT_SCHEMA`。
    :param warmup: 预热交易日数，检测日跳过最前面的这么多天。
    :param n_checks: 抽样检测日数，须至少为 1。
    :param rtol: 数值比较相对容差。
    :param atol: 数值比较绝对容差。
    """
    if n_checks < 1:
        raise ValueError(f"n_checks 至少为 1，实际 {n_checks}")
    if warmup < 0:
        raise ValueError(f"warmup 不能为负，实际 {warmup}")

    full = compute(data)
    validate_output(full)

    dates: list[dt.date] = data["date"].unique().sort().to_list()
    check_days = _select_check_days(dates, warmup, n_checks)
    if not check_days:
        return TruncationResult(
            ok=False,
            n_checks=0,
            failures=(),
            skipped_reason=(
                f"交易日数 {len(dates)} 不足：检测日需要跳过前 {warmup} 个预热交易日"
                f"后再有至少 1 个交易日，无法抽样，本次未检测"
            ),
        )

    failures: list[TruncationFailure] = []
    for day in check_days:
        truncated = data.filter(pl.col("date") <= day)
        recomputed = compute(truncated)
        failure = _compare_day(full, recomputed, day, rtol=rtol, atol=atol)
        if failure is not None:
            failures.append(failure)

    return TruncationResult(
        ok=not failures,
        n_checks=len(check_days),
        failures=tuple(failures),
        skipped_reason=None,
    )
