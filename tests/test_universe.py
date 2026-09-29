"""``quant.universe.members`` 的单元测试（issue #66）。

全部用合成 parquet / csv，不依赖真实 ``data/``。覆盖：

- 命名池 PIT：调样日前后成员不同；
- 命名池在成分表发布前返回空集（如 932000 在 2023-08 之前）；
- 自定义静态池（任意日同一集合）与动态池（按日精确匹配，PIT）；
- 自定义池的 ``instrument`` 归一化；
- 未知池名 / 不存在路径 / 缺列报 ``UniverseError``；
- ``members_range`` 跨调样日的区间展开、去重、排序与 schema；
- 静态池区间展开用交易日历、缺日历时报错。
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from quant.data.schema import INDEX_MEMBERS, TRADE_CALENDAR
from quant.universe.members import (
    NAMED_UNIVERSES,
    UniverseError,
    members,
    members_range,
)

A = "600000.SH"
B = "000001.SZ"
C = "300750.SZ"


# ---------------------------------------------------------------------------
# 合成数据
# ---------------------------------------------------------------------------


def _write_index_members(data_dir: Path, rows: list[dict[str, object]]) -> None:
    pl.DataFrame(rows, schema=INDEX_MEMBERS).write_parquet(
        data_dir / "index_members.parquet"
    )


def _member(day: date, instrument: str, index_code: str = "000300") -> dict[str, object]:
    return {"date": day, "instrument": instrument, "index_code": index_code}


def _write_calendar(data_dir: Path, open_days: list[date], closed: list[date]) -> None:
    days = [*open_days, *closed]
    pl.DataFrame(
        {"date": days, "is_open": [day in open_days for day in days]},
        schema=TRADE_CALENDAR,
    ).write_parquet(data_dir / "calendar.parquet")


def _write_static_parquet(
    data_dir: Path, instruments: list[str], name: str = "pool.parquet"
) -> Path:
    path = data_dir / name
    pl.DataFrame({"instrument": instruments}).write_parquet(path)
    return path


def _write_dynamic_parquet(
    data_dir: Path,
    rows: list[tuple[date, str]],
    name: str = "dyn.parquet",
) -> Path:
    path = data_dir / name
    pl.DataFrame(
        {"date": [day for day, _ in rows], "instrument": [code for _, code in rows]},
        schema=pl.Schema({"date": pl.Date, "instrument": pl.String}),
    ).write_parquet(path)
    return path


# ---------------------------------------------------------------------------
# 命名池
# ---------------------------------------------------------------------------


def _named_fixture(data_dir: Path) -> None:
    """000300 在 6/5 调样：B 被剔除。"""
    _write_index_members(
        data_dir,
        [
            _member(date(2024, 6, 3), A),
            _member(date(2024, 6, 3), B),
            _member(date(2024, 6, 4), A),
            _member(date(2024, 6, 4), B),
            _member(date(2024, 6, 5), A),
            _member(date(2024, 6, 3), C, index_code="000905"),
        ],
    )


def test_members_named_pit_before_after_rebalance(tmp_path: Path) -> None:
    _named_fixture(tmp_path)
    assert members("hs300", date(2024, 6, 4), data_dir=tmp_path) == {A, B}
    assert members("hs300", date(2024, 6, 5), data_dir=tmp_path) == {A}
    assert members("zz500", date(2024, 6, 3), data_dir=tmp_path) == {C}


def test_members_named_empty_before_first_snapshot(tmp_path: Path) -> None:
    # 932000（中证 2000）2023-08 前无成分，不应报错。
    _named_fixture(tmp_path)
    assert members("zz2000", date(2023, 1, 4), data_dir=tmp_path) == set()


def test_members_named_empty_for_missing_file(tmp_path: Path) -> None:
    assert members("hs300", date(2024, 6, 3), data_dir=tmp_path) == set()


def test_members_unknown_name_lists_available(tmp_path: Path) -> None:
    with pytest.raises(UniverseError) as excinfo:
        members("csi300", date(2024, 6, 3), data_dir=tmp_path)
    message = str(excinfo.value)
    for name in NAMED_UNIVERSES:
        assert name in message


def test_members_nonexistent_path_raises(tmp_path: Path) -> None:
    with pytest.raises(UniverseError):
        members(str(tmp_path / "missing.parquet"), date(2024, 6, 3), data_dir=tmp_path)


# ---------------------------------------------------------------------------
# 自定义池
# ---------------------------------------------------------------------------


def test_members_static_custom_any_day(tmp_path: Path) -> None:
    _write_static_parquet(tmp_path, ["600000", "000001"])
    path = str(tmp_path / "pool.parquet")
    # 任意日返回同一集合并完成归一化。
    assert members(path, date(2024, 6, 3), data_dir=tmp_path) == {A, B}
    assert members(path, date(2030, 1, 1), data_dir=tmp_path) == {A, B}


def test_members_dynamic_custom_pit(tmp_path: Path) -> None:
    path = str(
        _write_dynamic_parquet(
            tmp_path,
            [
                (date(2024, 6, 3), A),
                (date(2024, 6, 3), B),
                (date(2024, 6, 5), A),
            ],
        )
    )
    assert members(path, date(2024, 6, 3), data_dir=tmp_path) == {A, B}
    assert members(path, date(2024, 6, 4), data_dir=tmp_path) == set()
    assert members(path, date(2024, 6, 5), data_dir=tmp_path) == {A}


def test_members_static_csv(tmp_path: Path) -> None:
    pl.DataFrame({"instrument": ["600000", "000001"]}).write_csv(tmp_path / "pool.csv")
    assert members(str(tmp_path / "pool.csv"), date(2024, 6, 3), data_dir=tmp_path) == {
        A,
        B,
    }


def test_members_dynamic_csv(tmp_path: Path) -> None:
    pl.DataFrame(
        {"date": ["2024-06-03", "2024-06-05"], "instrument": ["600000", "000001"]}
    ).write_csv(tmp_path / "dyn.csv")
    path = str(tmp_path / "dyn.csv")
    assert members(path, date(2024, 6, 3), data_dir=tmp_path) == {A}
    assert members(path, date(2024, 6, 4), data_dir=tmp_path) == set()
    assert members(path, date(2024, 6, 5), data_dir=tmp_path) == {B}


def test_members_custom_missing_instrument_column_raises(tmp_path: Path) -> None:
    pl.DataFrame({"code": ["600000"]}).write_parquet(tmp_path / "bad.parquet")
    with pytest.raises(UniverseError):
        members(str(tmp_path / "bad.parquet"), date(2024, 6, 3), data_dir=tmp_path)


# ---------------------------------------------------------------------------
# members_range
# ---------------------------------------------------------------------------


def test_members_range_named_across_rebalance(tmp_path: Path) -> None:
    _named_fixture(tmp_path)
    frame = members_range("hs300", date(2024, 6, 3), date(2024, 6, 5), data_dir=tmp_path)
    assert frame.schema == pl.Schema({"date": pl.Date, "instrument": pl.String})
    got = set(zip(frame["date"].to_list(), frame["instrument"].to_list()))
    assert got == {
        (date(2024, 6, 3), A),
        (date(2024, 6, 3), B),
        (date(2024, 6, 4), A),
        (date(2024, 6, 4), B),
        (date(2024, 6, 5), A),
    }
    assert frame.equals(frame.sort(["date", "instrument"]))


def test_members_range_dedups_and_filters(tmp_path: Path) -> None:
    _write_index_members(
        tmp_path,
        [
            _member(date(2024, 6, 3), A),
            _member(date(2024, 6, 3), A),  # 重复行应被去重
            _member(date(2024, 6, 7), A),  # 区间外
        ],
    )
    frame = members_range("hs300", date(2024, 6, 3), date(2024, 6, 5), data_dir=tmp_path)
    assert frame.height == 1


def test_members_range_empty_when_no_records(tmp_path: Path) -> None:
    _named_fixture(tmp_path)
    frame = members_range("hs300", date(2025, 1, 1), date(2025, 1, 31), data_dir=tmp_path)
    assert frame.height == 0
    assert frame.schema == pl.Schema({"date": pl.Date, "instrument": pl.String})


def test_members_range_dynamic_custom(tmp_path: Path) -> None:
    path = str(
        _write_dynamic_parquet(
            tmp_path,
            [
                (date(2024, 6, 3), A),
                (date(2024, 6, 5), B),
                (date(2024, 6, 7), C),
            ],
        )
    )
    frame = members_range(path, date(2024, 6, 3), date(2024, 6, 5), data_dir=tmp_path)
    assert set(zip(frame["date"].to_list(), frame["instrument"].to_list())) == {
        (date(2024, 6, 3), A),
        (date(2024, 6, 5), B),
    }


def test_members_range_static_uses_open_days(tmp_path: Path) -> None:
    _write_static_parquet(tmp_path, ["600000", "000001"])
    open_days = [date(2024, 6, 3), date(2024, 6, 4), date(2024, 6, 5)]
    _write_calendar(tmp_path, open_days, closed=[date(2024, 6, 1)])
    frame = members_range(
        str(tmp_path / "pool.parquet"), date(2024, 6, 1), date(2024, 6, 5), data_dir=tmp_path
    )
    # 6/1 为休市日，不展开；每天两只票，共 3 天 × 2。
    assert frame.height == 6
    assert set(frame["date"].to_list()) == set(open_days)
    assert frame.equals(frame.sort(["date", "instrument"]))


def test_members_range_static_missing_calendar_raises(tmp_path: Path) -> None:
    _write_static_parquet(tmp_path, ["600000"])
    with pytest.raises(UniverseError):
        members_range(
            str(tmp_path / "pool.parquet"),
            date(2024, 6, 3),
            date(2024, 6, 5),
            data_dir=tmp_path,
        )
