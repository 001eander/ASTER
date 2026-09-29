"""``quant.factor_api.truncation`` 的单元测试：截断重算检测前视。

全部使用合成的多证券 × 多交易日面板，不触网、不读 ``data/`` 真实缓存。
"""
from __future__ import annotations

import dataclasses
import datetime as dt

import polars as pl
import pytest

from quant.factor_api.spec import FACTOR_INPUT_SCHEMA
from quant.factor_api.truncation import (
    DEFAULT_ATOL,
    DEFAULT_N_CHECKS,
    DEFAULT_RTOL,
    DEFAULT_WARMUP_DAYS,
    TruncationResult,
    check_truncation,
)

INSTRUMENTS: tuple[str, ...] = ("000001.SZ", "300750.SZ", "600000.SH")


def _make_input(n_days: int, n_instruments: int = len(INSTRUMENTS)) -> pl.DataFrame:
    """合成面板：``n_instruments`` 只证券 × ``n_days`` 个交易日，按 (instrument, date) 排序。

    价格单调递增，便于构造 shift / rolling 类因子；日期用连续日历日，检测不依赖真实交易日历。
    """
    rows: list[tuple] = []
    start = dt.date(2021, 9, 29)
    for index, instrument in enumerate(INSTRUMENTS[:n_instruments]):
        for step in range(n_days):
            day = start + dt.timedelta(days=step)
            base = 10.0 + 0.1 * step + float(index)
            rows.append(
                (
                    day,
                    instrument,
                    base,
                    base + 1.0,
                    base - 1.0,
                    base + 0.5,
                    base,
                    1000.0,
                    base * 1000.0,
                    1.0,
                )
            )
    return pl.DataFrame(rows, schema=FACTOR_INPUT_SCHEMA, orient="row")


def _causal_factor(data: pl.DataFrame) -> pl.DataFrame:
    """无前视因子：close 的一日滞后。"""
    return (
        data.sort(["instrument", "date"])
        .with_columns(pl.col("close").shift(1).over("instrument").alias("value"))
        .select("date", "instrument", "value")
    )


def _lookahead_factor(data: pl.DataFrame) -> pl.DataFrame:
    """前视因子：close 的下一日值，用到了检测日之后的数据。"""
    return (
        data.sort(["instrument", "date"])
        .with_columns(pl.col("close").shift(-1).over("instrument").alias("value"))
        .select("date", "instrument", "value")
    )


class TestCausalFactorPasses:
    def test_causal_factor_passes(self) -> None:
        data = _make_input(120)
        result = check_truncation(_causal_factor, data)
        assert result.ok is True
        assert result.n_checks == DEFAULT_N_CHECKS
        assert result.failures == ()
        assert result.skipped_reason is None

    def test_shift_and_rolling_are_causal(self) -> None:
        def factor(data: pl.DataFrame) -> pl.DataFrame:
            ordered = data.sort(["instrument", "date"])
            return ordered.with_columns(
                (
                    pl.col("close").shift(1).over("instrument")
                    + pl.col("close").rolling_mean(5).over("instrument")
                ).alias("value")
            ).select("date", "instrument", "value")

        result = check_truncation(factor, _make_input(120))
        assert result.ok is True
        assert result.failures == ()


class TestLookaheadDetected:
    def test_lookahead_factor_flagged(self) -> None:
        data = _make_input(120)
        result = check_truncation(_lookahead_factor, data)
        assert result.ok is False
        assert result.skipped_reason is None
        assert len(result.failures) > 0

        dates = data["date"].unique().sort().to_list()
        for failure in result.failures:
            assert failure.date in dates
            assert dates.index(failure.date) >= DEFAULT_WARMUP_DAYS
            assert failure.n_mismatch == len(INSTRUMENTS)

    def test_future_aggregate_flagged_with_numeric_diff(self) -> None:
        def future_mean_factor(data: pl.DataFrame) -> pl.DataFrame:
            """用全样本 close 均值做中心化，均值里混入了检测日之后的行。"""
            return (
                data.sort(["instrument", "date"])
                .with_columns(
                    (pl.col("close") - pl.col("close").mean()).alias("value")
                )
                .select("date", "instrument", "value")
            )

        result = check_truncation(
            future_mean_factor, _make_input(120), warmup=5, n_checks=1
        )
        assert result.ok is False
        failure = result.failures[0]
        assert failure.n_mismatch == len(INSTRUMENTS)
        # 两侧都有值，偏差可计算。
        assert failure.max_abs_diff is not None
        assert failure.max_abs_diff > DEFAULT_ATOL

    def test_failure_dates_are_sampled_check_days(self) -> None:
        data = _make_input(120)
        result = check_truncation(_lookahead_factor, data, n_checks=3)
        assert result.n_checks == 3
        assert 0 < len(result.failures) <= 3
        assert list(result.failures) == sorted(
            result.failures, key=lambda failure: failure.date
        )


class TestWarmup:
    def test_check_days_avoid_warmup_window(self) -> None:
        calls: list[dt.date] = []

        def probe(data: pl.DataFrame) -> pl.DataFrame:
            calls.append(data["date"].max())
            return (
                data.sort(["instrument", "date"])
                .with_columns(
                    pl.col("close").rolling_mean(60).over("instrument").alias("value")
                )
                .select("date", "instrument", "value")
            )

        data = _make_input(120)
        result = check_truncation(probe, data, n_checks=5)
        assert result.ok is True
        assert result.n_checks == 5

        dates = data["date"].unique().sort().to_list()
        # 首次调用是全量数据，之后每个检测日各调用一次。
        assert len(calls) == 1 + result.n_checks
        # 因子接口 60 日窗口在起步阶段本就不稳，检测日必须落在预热期之后。
        assert all(dates.index(call_day) >= DEFAULT_WARMUP_DAYS for call_day in calls)

    def test_custom_warmup_shifts_check_window(self) -> None:
        data = _make_input(120)
        dates = data["date"].unique().sort().to_list()

        def run(warmup: int) -> dt.date:
            calls: list[dt.date] = []

            def probe(frame: pl.DataFrame) -> pl.DataFrame:
                calls.append(frame["date"].max())
                return _causal_factor(frame)

            check_truncation(probe, data, warmup=warmup, n_checks=1)
            return calls[-1]

        early_index = dates.index(run(10))
        late_index = dates.index(run(DEFAULT_WARMUP_DAYS))
        assert early_index >= 10
        # 预热期调小后，检测窗口整体前移。
        assert early_index < late_index


class TestNotCheckable:
    def test_too_short_data_is_distinguishable(self) -> None:
        data = _make_input(DEFAULT_WARMUP_DAYS - 1)
        result = check_truncation(_lookahead_factor, data)
        assert result.ok is False
        assert result.skipped_reason is not None
        assert result.failures == ()
        assert result.n_checks == 0

    def test_exactly_warmup_days_is_not_checkable(self) -> None:
        data = _make_input(DEFAULT_WARMUP_DAYS)
        result = check_truncation(_causal_factor, data, warmup=DEFAULT_WARMUP_DAYS)
        assert result.skipped_reason is not None
        assert result.n_checks == 0


class TestNullAndTolerance:
    def test_null_mismatch_counted(self) -> None:
        # 单个检测日落在区间中部，前视因子在截断输入下该日值变 null。
        data = _make_input(120)
        result = check_truncation(_lookahead_factor, data, n_checks=1)
        assert result.ok is False
        assert len(result.failures) == 1
        failure = result.failures[0]
        assert failure.n_mismatch == len(INSTRUMENTS)
        # 不一致全部来自「全量有值、截断为 null」，没有可计算的数值偏差。
        assert failure.max_abs_diff is None

    def test_missing_instrument_counted(self) -> None:
        data = _make_input(120)
        last_day = data["date"].max()

        def dropping_factor(frame: pl.DataFrame) -> pl.DataFrame:
            """截断输入下丢掉一只证券，模拟「一侧有此证券、另一侧整体缺行」。"""
            out = _causal_factor(frame)
            if frame["date"].max() != last_day:
                return out.filter(pl.col("instrument") != INSTRUMENTS[0])
            return out

        result = check_truncation(dropping_factor, data, n_checks=3)
        assert result.ok is False
        assert all(failure.n_mismatch == 1 for failure in result.failures)
        # 不一致来自整行缺失，没有可计算的数值偏差。
        assert all(failure.max_abs_diff is None for failure in result.failures)

    def test_tiny_path_noise_within_tolerance_passes(self) -> None:
        def near_causal(data: pl.DataFrame) -> pl.DataFrame:
            """对 max date 有极小依赖，模拟数值路径噪声，幅度远低于默认容差。"""
            span_days = (pl.col("date").max() - pl.col("date")).dt.total_days()
            return (
                data.sort(["instrument", "date"])
                .with_columns((pl.col("close") + span_days * 1e-15).alias("value"))
                .select("date", "instrument", "value")
            )

        data = _make_input(120)
        default_result = check_truncation(near_causal, data)
        assert default_result.ok is True
        assert default_result.failures == ()

        # 容差收紧到 0 后同一因子被检出，证明容差确实参与了判定。
        strict_result = check_truncation(near_causal, data, rtol=0.0, atol=0.0)
        assert strict_result.ok is False
        assert len(strict_result.failures) > 0

    def test_constants_exposed(self) -> None:
        assert DEFAULT_WARMUP_DAYS == 60
        assert DEFAULT_N_CHECKS == 5
        assert DEFAULT_RTOL == pytest.approx(1e-9)
        assert DEFAULT_ATOL == pytest.approx(1e-12)


class TestResultTypes:
    def test_frozen_dataclasses(self) -> None:
        result = check_truncation(_causal_factor, _make_input(120))
        assert isinstance(result, TruncationResult)
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.ok = False  # type: ignore[misc]
