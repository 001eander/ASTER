"""``quant.labels.open_to_open`` 的单元测试。

全部用合成数据，不触网、不依赖 ``data/``。核对项：

- 基本正确性：3 只证券 5 天标签逐值手算
- 除权：中间一天送转使未复权价跳空，标签不受影响
- 停牌：缺中间一天时标签顺延，``delay_days`` 暴露延迟
- 尾部 null：最后 ``horizon + 1`` 天 label 为 null
- horizon=2
- 输入未排序：函数内部先排序，结果一致
"""
from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from quant.data.schema import DAILY_BARS
from quant.labels import DEFAULT_HORIZON, attach_label, open_to_open_label

D1 = dt.date(2026, 1, 5)  # 周一
D2 = dt.date(2026, 1, 6)
D3 = dt.date(2026, 1, 7)
D4 = dt.date(2026, 1, 8)
D5 = dt.date(2026, 1, 9)
DAYS = [D1, D2, D3, D4, D5]

A = "600000.SH"
B = "000001.SZ"
C = "300750.SZ"


def _bars(rows: list[tuple[dt.date, str, float, float]]) -> pl.DataFrame:
    """由 ``(date, instrument, open, adjfactor)`` 合成完整 DAILY_BARS。"""
    return pl.DataFrame(
        [
            {
                "date": d,
                "instrument": ins,
                "open": o,
                "high": o,
                "low": o,
                "close": o,
                "vwap": o,
                "volume": 1.0,
                "amount": o,
                "adjfactor": af,
                "limit_up": None,
                "limit_down": None,
            }
            for d, ins, o, af in rows
        ],
        schema=DAILY_BARS,
    )


def _as_map(df: pl.DataFrame) -> dict[tuple[str, dt.date], tuple[float | None, int | None]]:
    return {
        (row["instrument"], row["date"]): (row["label"], row["delay_days"])
        for row in df.iter_rows(named=True)
    }


class TestBasicCorrectness:
    """3 只证券各 5 天，horizon=1 逐值手算核对。"""

    @staticmethod
    def _frame() -> pl.DataFrame:
        rows: list[tuple[dt.date, str, float, float]] = []
        series = {
            A: [10.0, 11.0, 12.0, 13.0, 14.0],
            B: [20.0, 22.0, 24.0, 26.0, 28.0],
            C: [100.0, 90.0, 81.0, 72.9, 65.61],
        }
        for instrument, opens in series.items():
            for day, open_ in zip(DAYS, opens):
                rows.append((day, instrument, open_, 1.0))
        return _bars(rows)

    def test_hand_computed_horizon_1(self) -> None:
        out = open_to_open_label(self._frame())
        assert out.columns == ["date", "instrument", "label", "delay_days"]
        got = _as_map(out)

        # 每个信号日 T 的 label = open(T+2) / open(T+1) - 1
        expected: dict[tuple[str, dt.date], float] = {
            (A, D1): 12.0 / 11.0 - 1.0,
            (A, D2): 13.0 / 12.0 - 1.0,
            (A, D3): 14.0 / 13.0 - 1.0,
            (B, D1): 24.0 / 22.0 - 1.0,
            (B, D2): 26.0 / 24.0 - 1.0,
            (B, D3): 28.0 / 26.0 - 1.0,
            (C, D1): 81.0 / 90.0 - 1.0,
            (C, D2): 72.9 / 81.0 - 1.0,
            (C, D3): 65.61 / 72.9 - 1.0,
        }
        for key, value in expected.items():
            label, delay = got[key]
            assert label == pytest.approx(value)
            assert delay == 1

        # 尾部两天无未来行情 -> null
        for instrument in (A, B, C):
            assert got[(instrument, D4)][0] is None
            assert got[(instrument, D5)][0] is None
        assert out.height == 15

    def test_default_horizon_is_1(self) -> None:
        assert DEFAULT_HORIZON == 1
        assert open_to_open_label(self._frame()).equals(
            open_to_open_label(self._frame(), horizon=1)
        )

    def test_output_sorted(self) -> None:
        out = open_to_open_label(self._frame())
        assert out.equals(out.sort(["instrument", "date"]))

    def test_requires_columns(self) -> None:
        bars = self._frame().drop("adjfactor")
        with pytest.raises(ValueError, match="缺少列"):
            open_to_open_label(bars)

    def test_rejects_bad_horizon(self) -> None:
        with pytest.raises(ValueError, match="horizon"):
            open_to_open_label(self._frame(), horizon=0)


class TestAdjfactor:
    """除权日未复权价跳空，标签必须用后复权价，不受影响。"""

    def test_ex_date_does_not_distort_label(self) -> None:
        # 第 3 天 10 送 10：未复权价 10 -> 5，adjfactor 1 -> 2，后复权价恒为 10。
        opens = [10.0, 10.0, 10.0, 5.0, 5.0]
        adjs = [1.0, 1.0, 1.0, 2.0, 2.0]
        bars = _bars([(d, A, o, af) for d, o, af in zip(DAYS, opens, adjs)])

        got = _as_map(open_to_open_label(bars))
        # D1 的 T+1 在 D2、T+2 在 D3，两者后复权价都是 10
        assert got[(A, D1)][0] == pytest.approx(0.0)
        # D2 的 T+2 落在除权日 D4，未复权会得到 -50%，后复权则为 0
        assert got[(A, D2)][0] == pytest.approx(0.0)
        assert got[(A, D3)][0] == pytest.approx(0.0)

        # 反证：若误用未复权价，D2 的标签会是 -0.5
        raw_label = 5.0 / 10.0 - 1.0
        assert raw_label == pytest.approx(-0.5)
        assert got[(A, D2)][0] != pytest.approx(raw_label)


class TestSuspension:
    """某票缺中间一天：标签顺延到下一个实际交易日，delay_days 暴露延迟。"""

    @staticmethod
    def _frame() -> pl.DataFrame:
        # D3 停牌，无该行；opens: D1=10, D2=11, D4=12, D5=13
        return _bars(
            [
                (D1, A, 10.0, 1.0),
                (D2, A, 11.0, 1.0),
                (D4, A, 12.0, 1.0),
                (D5, A, 13.0, 1.0),
            ]
        )

    def test_label_and_delay(self) -> None:
        got = _as_map(open_to_open_label(self._frame()))

        # T=D1：T+1=D2，T+2=D4，label = 12/11-1
        assert got[(A, D1)][0] == pytest.approx(12.0 / 11.0 - 1.0)
        assert got[(A, D1)][1] == 1

        # T=D2：T+1=D4（停牌顺延），T+2=D5，label = 13/12-1
        # 建仓日距信号日 2 个日历天，delay_days 暴露停牌
        assert got[(A, D2)][0] == pytest.approx(13.0 / 12.0 - 1.0)
        assert got[(A, D2)][1] == 2

        # 尾部无未来行情
        assert got[(A, D4)][0] is None
        assert got[(A, D5)][0] is None

    def test_delay_days_uses_calendar_days(self) -> None:
        # 周二到周五的日历间隔为 3，delay_days 按日历天计
        bars = _bars([(D2, A, 10.0, 1.0), (D5, A, 11.0, 1.0), (D1, B, 1.0, 1.0)])
        out = open_to_open_label(bars)
        # A 只有两行，D2 的建仓日 D5 间隔 3 天
        a_row = out.filter((pl.col("instrument") == A) & (pl.col("date") == D2))
        assert a_row["delay_days"][0] == 3


class TestTailNull:
    """尾部 horizon+1 天 label 为 null，且不丢行。"""

    @pytest.mark.parametrize("horizon", [1, 2, 3])
    def test_last_rows_null(self, horizon: int) -> None:
        bars = _bars([(d, A, 10.0 + i, 1.0) for i, d in enumerate(DAYS)])
        out = open_to_open_label(bars, horizon=horizon)
        assert out.height == len(DAYS)
        labels = out.sort("date")["label"].to_list()
        for value in labels[: len(DAYS) - horizon - 1]:
            assert value is not None
        for value in labels[len(DAYS) - horizon - 1 :]:
            assert value is None


class TestHorizon2:
    """horizon=2：T+1 开盘 → T+3 开盘。"""

    def test_hand_computed(self) -> None:
        bars = _bars([(d, A, 10.0 + i, 1.0) for i, d in enumerate(DAYS)])
        got = _as_map(open_to_open_label(bars, horizon=2))
        # D1: entry D2=11, exit D4=13 -> 13/11-1
        assert got[(A, D1)][0] == pytest.approx(13.0 / 11.0 - 1.0)
        assert got[(A, D1)][1] == 1
        # D2: entry D3=12, exit D5=14 -> 14/12-1
        assert got[(A, D2)][0] == pytest.approx(14.0 / 12.0 - 1.0)
        # D3/D4/D5 未来不足
        assert got[(A, D3)][0] is None
        assert got[(A, D4)][0] is None
        assert got[(A, D5)][0] is None


class TestUnsortedInput:
    """输入未按 (instrument, date) 排序时，函数内部先排序，结果仍正确。"""

    def _sorted_bars(self) -> pl.DataFrame:
        rows: list[tuple[dt.date, str, float, float]] = []
        for instrument, base in ((A, 10.0), (B, 20.0)):
            for i, day in enumerate(DAYS):
                rows.append((day, instrument, base + i, 1.0))
        return _bars(rows)

    def test_shuffled_matches_sorted(self) -> None:
        sorted_bars = self._sorted_bars()
        shuffled = sorted_bars.sample(fraction=1.0, shuffle=True, seed=42)
        assert not shuffled.equals(sorted_bars)

        expected = open_to_open_label(sorted_bars)
        got = open_to_open_label(shuffled)
        assert got.equals(expected)


class TestAttachLabel:
    """attach_label 保留 DAILY_BARS 全部列并附加 label / delay_days。"""

    def test_columns_and_values(self) -> None:
        bars = _bars([(d, A, 10.0 + i, 1.0) for i, d in enumerate(DAYS)])
        out = attach_label(bars)

        for col in DAILY_BARS:
            assert col in out.columns
        assert "label" in out.columns
        assert "delay_days" in out.columns
        assert out.height == bars.height

        direct = open_to_open_label(bars)
        got = out.select("date", "instrument", "label", "delay_days")
        assert got.equals(direct)

    def test_unmatched_rows_kept(self) -> None:
        # 单行输入，标签为 null，行仍保留
        bars = _bars([(D1, A, 10.0, 1.0)])
        out = attach_label(bars)
        assert out.height == 1
        assert out["label"][0] is None
