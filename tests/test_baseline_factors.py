"""``factor_library`` 基线因子集的单元测试。

全部用合成数据，不触网、不读 ``data/``。3 只证券 × 100 天的确定性价格 / 成交量
序列，核对项：

- 每个因子输出 ``(date, instrument, value)`` schema 正确、不丢行；
- 同一输入两次 ``compute`` 结果一致（无随机、无时间依赖）；
- 无前视：截断未来数据后重算，保留区间内取值不变；
- 窗口局部性：只保留尾部数据重算，预热期之后取值与全量一致；
- 关键因子手算核对（``mom_5`` / ``ma_bias_10``）；
- 除权日不误导：``adjfactor`` 跳变时 ``mom_5`` 不出现假跳变。
"""
from __future__ import annotations

import datetime as dt
import importlib
import math
from pathlib import Path

import polars as pl
import pytest

from quant.data.schema import DAILY_BARS

# ---------------------------------------------------------------------------
# 合成数据
# ---------------------------------------------------------------------------

A = "600000.SH"
B = "000001.SZ"
C = "300750.SZ"

N_DAYS = 100
START = dt.date(2026, 1, 1)

FACTOR_DIR = Path(__file__).resolve().parents[1] / "factor_library"
FACTOR_NAMES = sorted(
    p.stem for p in FACTOR_DIR.glob("*.py") if not p.stem.startswith("_")
)


def _dates(n: int = N_DAYS) -> list[dt.date]:
    return [START + dt.timedelta(days=i) for i in range(n)]


def _closes() -> dict[str, list[float]]:
    """三只票的确定性价格序列：A 上行、B 下行、C 周期波动。"""
    return {
        A: [10.0 * (1.01**i) for i in range(N_DAYS)],
        B: [50.0 - 0.2 * i for i in range(N_DAYS)],
        C: [20.0 + 3.0 * math.sin(i / 5.0) for i in range(N_DAYS)],
    }


def _volumes() -> dict[str, list[float]]:
    return {
        A: [1000.0 + 50.0 * (i % 7) for i in range(N_DAYS)],
        B: [2000.0 - 30.0 * (i % 5) for i in range(N_DAYS)],
        C: [1500.0 + 100.0 * (i % 3) for i in range(N_DAYS)],
    }


def _make_bars(*, exdiv: bool = False) -> pl.DataFrame:
    """合成完整 ``DAILY_BARS``。

    ``exdiv=True`` 时，证券 A 从第 50 天起 ``adjfactor`` 跳到 1.1、未复权价
    同步除以 1.1，使 ``close × adjfactor`` 保持连续（模拟除权）。
    """
    dates = _dates()
    closes = _closes()
    volumes = _volumes()
    rows: list[dict[str, object]] = []
    for instrument, series in closes.items():
        for i, day in enumerate(dates):
            close = series[i]
            adjfactor = 1.0
            if exdiv and instrument == A and i >= 50:
                adjfactor = 1.1
                close = close / 1.1
            high = close * 1.02
            low = close * 0.98
            vwap = close * (1.0 + 0.001 * (i % 5))
            volume = volumes[instrument][i]
            rows.append(
                {
                    "date": day,
                    "instrument": instrument,
                    "open": close * 0.997,
                    "high": high,
                    "low": low,
                    "close": close,
                    "vwap": vwap,
                    "volume": volume,
                    "amount": vwap * volume,
                    "adjfactor": adjfactor,
                    "limit_up": None,
                    "limit_down": None,
                }
            )
    return pl.DataFrame(rows, schema=DAILY_BARS).sort(["instrument", "date"])


def _compute(name: str, data: pl.DataFrame) -> pl.DataFrame:
    module = importlib.import_module(f"factor_library.{name}")
    return module.compute(data)


def _value(df: pl.DataFrame, instrument: str, day: dt.date) -> float | None:
    return (
        df.filter(
            (pl.col("instrument") == instrument) & (pl.col("date") == day)
        )["value"].item()
    )


def _assert_close(a: pl.Series, b: pl.Series, tol: float = 1e-9) -> None:
    assert a.len() == b.len(), f"长度不一致：{a.len()} vs {b.len()}"
    for x, y in zip(a.to_list(), b.to_list()):
        if x is None or y is None:
            assert x is None and y is None, f"null 不匹配：{x!r} vs {y!r}"
        else:
            assert math.isclose(x, y, rel_tol=tol, abs_tol=tol), f"{x!r} vs {y!r}"


# ---------------------------------------------------------------------------
# 因子集契约
# ---------------------------------------------------------------------------


def test_factor_set_has_twelve_factors() -> None:
    assert len(FACTOR_NAMES) == 12, f"因子数应为 12，实际 {FACTOR_NAMES}"


@pytest.mark.parametrize("name", FACTOR_NAMES)
def test_output_schema(name: str) -> None:
    data = _make_bars()
    out = _compute(name, data)
    assert out.columns == ["date", "instrument", "value"], name
    assert out.schema["date"] == pl.Date, name
    assert out.schema["instrument"] == pl.String, name
    assert out.schema["value"] == pl.Float64, name
    # 不丢行：每个 (date, instrument) 都保留，窗口不足处为 null
    assert out.height == data.height, name
    assert out.select("date", "instrument").equals(
        data.select("date", "instrument")
    ), name


@pytest.mark.parametrize("name", FACTOR_NAMES)
def test_deterministic(name: str) -> None:
    data = _make_bars()
    first = _compute(name, data)
    second = _compute(name, data)
    assert first.equals(second), name


# ---------------------------------------------------------------------------
# 无前视
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", FACTOR_NAMES)
def test_no_lookahead_future_truncation(name: str) -> None:
    """截断未来数据后重算，保留区间内取值必须与全量一致。"""
    data = _make_bars()
    cutoff = _dates()[70]
    truncated = data.filter(pl.col("date") <= cutoff)

    full = _compute(name, data).filter(pl.col("date") <= cutoff)
    part = _compute(name, truncated)

    full = full.sort(["instrument", "date"])
    part = part.sort(["instrument", "date"])
    assert full.select("date", "instrument").equals(
        part.select("date", "instrument")
    ), name
    _assert_close(full["value"], part["value"])


@pytest.mark.parametrize("name", FACTOR_NAMES)
def test_window_locality_suffix_recompute(name: str) -> None:
    """只保留尾部 80 天重算，最后 20 天（预热期之后）取值与全量一致。"""
    data = _make_bars()
    suffix_start = _dates()[20]
    suffix = data.filter(pl.col("date") >= suffix_start)

    window_dates = _dates()[80:]
    full = (
        _compute(name, data)
        .filter(pl.col("date").is_in(window_dates))
        .sort(["instrument", "date"])
    )
    part = (
        _compute(name, suffix)
        .filter(pl.col("date").is_in(window_dates))
        .sort(["instrument", "date"])
    )
    # 预热期在尾部 40 天内已满足，最后 20 天三个窗口类因子都应非 null
    assert full["value"].null_count() == 0, name
    _assert_close(full["value"], part["value"])


# ---------------------------------------------------------------------------
# 手算核对
# ---------------------------------------------------------------------------


def test_mom_5_handcheck() -> None:
    data = _make_bars()
    out = _compute("mom_5", data)
    i = 60
    expected = (10.0 * 1.01**i) / (10.0 * 1.01 ** (i - 5)) - 1.0
    assert _value(out, A, _dates()[i]) == pytest.approx(expected, rel=1e-12)
    # 窗口预热期应为 null
    assert _value(out, A, _dates()[4]) is None


def test_ma_bias_10_handcheck() -> None:
    data = _make_bars()
    out = _compute("ma_bias_10", data)
    i = 60
    window = [10.0 * 1.01**k for k in range(i - 9, i + 1)]
    expected = window[-1] / (sum(window) / 10.0) - 1.0
    assert _value(out, A, _dates()[i]) == pytest.approx(expected, rel=1e-12)
    assert _value(out, A, _dates()[8]) is None


def test_reversal_5_is_negative_of_open_momentum() -> None:
    data = _make_bars()
    out = _compute("reversal_5", data)
    i = 60
    opens = [10.0 * 1.01**k * 0.997 for k in range(i - 5, i + 1)]
    # reversal = open(T-5) / open(T) - 1，与 mom 反号但复利口径不是简单取负
    expected = opens[0] / opens[-1] - 1.0
    assert _value(out, A, _dates()[i]) == pytest.approx(expected, rel=1e-12)
    assert expected < 0.0  # A 上行走势下反转信号为负


# ---------------------------------------------------------------------------
# 除权
# ---------------------------------------------------------------------------


def test_mom_5_not_fooled_by_ex_dividend() -> None:
    """adjfactor 跳变日复权口径下 mom_5 与无除权对照一致，不产生假跳变。"""
    control = _compute("mom_5", _make_bars(exdiv=False))
    adjusted = _compute("mom_5", _make_bars(exdiv=True))
    day = _dates()[50]

    expected = _value(control, A, day)
    assert expected is not None
    assert _value(adjusted, A, day) == pytest.approx(expected, rel=1e-12)

    # 若错误地用未复权价，除权日会出现约 1/1.1 的假跳变；确认本因子没有
    raw = _make_bars(exdiv=True).filter(pl.col("instrument") == A)
    close = raw["close"].to_list()
    idx = raw["date"].to_list().index(day)
    naive = close[idx] / close[idx - 5] - 1.0
    assert not math.isclose(naive, expected, rel_tol=1e-6)
