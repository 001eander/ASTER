"""``scripts/build_flagship_universe.py`` 的单元测试（issue #113）。

全部用合成行情 / 证券信息 / ST 区间，不依赖真实 ``data/``。覆盖：

- 季度末刷新日的推导与「成员有效期到下一刷新日」的逐日展开；
- 五条过滤规则各自的边界：ST 区间 PIT、上市天数、板块、价格上限、成交额窗口与门槛；
- top N 截断；
- 数据不完整的刷新日被跳过（成员沿用上一刷新日）；
- 产物能被 ``quant.universe.members`` 的自定义动态池路径解析（PIT 生效）；
- ``config/strategy-flagship.json`` 能被 ``load_strategy_config`` 解析，
  且与 ``backtest_e2e.py`` 的命令行参数一致性检查兼容。
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from quant.data.limit import ST_INTERVALS
from quant.data.schema import DAILY_BARS, INSTRUMENT_INFO
from quant.daily.strategy import build_stock_optimizer, load_strategy_config
from quant.universe.members import members, members_range
from scripts.build_flagship_universe import (
    AMOUNT_WINDOW,
    MAX_CLOSE,
    MIN_AVG_AMOUNT,
    MIN_LISTED_TRADING_DAYS,
    POOL_START,
    TOP_N,
    build_pool,
    quarter_end_refresh_days,
    refresh_days_in_use,
    select_members,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FLAGSHIP_POOL = REPO_ROOT / "config" / "universe" / "flagship_pool.parquet"
FLAGSHIP_STRATEGY = REPO_ROOT / "config" / "strategy-flagship.json"

#: 合成数据里的默认上市日：早于日历窗口，上市天数过滤直接放行。
DEFAULT_LIST_DATE = date(2015, 1, 5)

#: 合成行情里「每天成交额都很大」的默认值，保证只被目标规则拦下。
DEFAULT_AMOUNT = 5e8


# ---------------------------------------------------------------------------
# 合成数据
# ---------------------------------------------------------------------------


def _open_days(start: date, end: date) -> list[date]:
    """``[start, end]`` 的工作日序列，充当交易日历的开市日。"""
    days: list[date] = []
    day = start
    while day <= end:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return days


def _spec(
    instrument: str,
    *,
    close: float | object = 20.0,
    amount: float | object = DEFAULT_AMOUNT,
    list_date: date = DEFAULT_LIST_DATE,
    first_day: date | None = None,
    name: str = "测试",
) -> dict[str, object]:
    """一只证券的合成设定；``close`` / ``amount`` 可传常量或 ``(day) -> float``。"""
    return {
        "instrument": instrument,
        "name": name,
        "list_date": list_date,
        "first_day": first_day,
        "close": close,
        "amount": amount,
    }


def _value(value: object, day: date) -> float:
    return float(value(day)) if callable(value) else float(value)  # type: ignore[arg-type]


def _instruments(specs: list[dict[str, object]]) -> pl.DataFrame:
    rows = [
        {
            "instrument": spec["instrument"],
            "name": spec["name"],
            "board": _board_of(str(spec["instrument"])),
            "list_date": spec["list_date"],
            "delist_date": None,
        }
        for spec in specs
    ]
    return pl.DataFrame(rows, schema=INSTRUMENT_INFO)


def _board_of(instrument: str) -> str:
    digits, exchange = instrument.split(".")
    head3 = digits[:3]
    if exchange == "BJ":
        return "bj"
    if head3 in {"688", "689"}:
        return "kcb"
    if head3 in {"300", "301", "302"}:
        return "cyb"
    return "main"


def _bars(specs: list[dict[str, object]], open_days: list[date]) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in specs:
        first_day = spec["first_day"]
        for day in open_days:
            if first_day is not None and day < first_day:  # type: ignore[operator]
                continue
            close = _value(spec["close"], day)
            amount = _value(spec["amount"], day)
            rows.append(
                {
                    "date": day,
                    "instrument": spec["instrument"],
                    "open": close,
                    "high": close,
                    "low": close,
                    "close": close,
                    "vwap": close,
                    "volume": amount / close,
                    "amount": amount,
                    "adjfactor": 1.0,
                    "limit_up": None,
                    "limit_down": None,
                }
            )
    return pl.DataFrame(rows, schema=DAILY_BARS)


def _st_intervals(rows: list[tuple[str, date, date | None]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "instrument": [instrument for instrument, _, _ in rows],
            "start_date": [start for _, start, _ in rows],
            "end_date": [end for _, _, end in rows],
        },
        schema=ST_INTERVALS,
    )


def _no_st() -> pl.DataFrame:
    return pl.DataFrame(schema=ST_INTERVALS)


def _scenario(
    specs: list[dict[str, object]], open_days: list[date]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    return _bars(specs, open_days), _instruments(specs)


# ---------------------------------------------------------------------------
# 刷新日
# ---------------------------------------------------------------------------


def test_quarter_end_refresh_days_takes_last_open_day_of_each_quarter() -> None:
    days = _open_days(date(2023, 1, 2), date(2023, 12, 29))
    assert quarter_end_refresh_days(days, date(2023, 1, 1)) == [
        date(2023, 3, 31),
        date(2023, 6, 30),
        date(2023, 9, 29),
        date(2023, 12, 29),
    ]


def test_quarter_end_refresh_days_respects_start() -> None:
    days = _open_days(date(2023, 1, 2), date(2023, 12, 29))
    # start 落在 Q1 之后时，Q1 的刷新日不出现在结果里。
    assert quarter_end_refresh_days(days, date(2023, 4, 1)) == [
        date(2023, 6, 30),
        date(2023, 9, 29),
        date(2023, 12, 29),
    ]


def test_quarter_end_refresh_days_skips_holiday_quarter_end() -> None:
    # 3/31 是周日时，取该季度最后一个开市日 3/29。
    days = _open_days(date(2024, 1, 2), date(2024, 6, 28))
    assert quarter_end_refresh_days(days, date(2024, 1, 1)) == [
        date(2024, 3, 29),
        date(2024, 6, 28),
    ]


# ---------------------------------------------------------------------------
# 单刷新日的过滤规则
# ---------------------------------------------------------------------------


def test_st_excluded_when_interval_covers_refresh_day() -> None:
    days = _open_days(date(2023, 1, 2), date(2023, 12, 29))
    refresh_days = quarter_end_refresh_days(days, date(2023, 1, 1))
    specs = [_spec("600000.SH"), _spec("600001.SH")]
    bars, instruments = _scenario(specs, days)
    st = _st_intervals(
        [
            ("600001.SH", date(2023, 1, 4), date(2023, 6, 15)),
            ("600001.SH", date(2023, 9, 1), None),  # 至今（开区间）
        ]
    )

    first = set(select_members(bars, instruments, st, days, refresh_days[0])["instrument"])
    second = set(select_members(bars, instruments, st, days, refresh_days[1])["instrument"])
    third = set(select_members(bars, instruments, st, days, refresh_days[2])["instrument"])
    # Q1 处于 ST 区间内 → 剔除；Q2 区间在 6/15 结束而刷新日 6/30 已摘帽 → 收；Q3 起仍 ST。
    assert first == {"600000.SH"}
    assert second == {"600000.SH", "600001.SH"}
    assert third == {"600000.SH"}


def test_st_interval_starting_after_refresh_day_does_not_exclude() -> None:
    days = _open_days(date(2023, 1, 2), date(2023, 6, 30))
    refresh_day = quarter_end_refresh_days(days, date(2023, 1, 1))[0]
    bars, instruments = _scenario([_spec("600001.SH")], days)
    st = _st_intervals([("600001.SH", date(2023, 6, 1), None)])
    assert select_members(bars, instruments, st, days, refresh_day).height == 1


def test_recent_listing_excluded_until_enough_trading_days() -> None:
    # 日历要比 120 个交易日更长，才能构造「恰好够 121 个交易日」的上市日。
    days = _open_days(date(2022, 6, 1), date(2023, 6, 30))
    refresh_day = quarter_end_refresh_days(days, date(2022, 10, 1))[0]
    index = days.index(refresh_day)
    assert index >= MIN_LISTED_TRADING_DAYS
    fresh_list_date = days[index - 30]  # 上市 31 个交易日
    seasoned_list_date = days[index - MIN_LISTED_TRADING_DAYS]  # 上市 121 个交易日
    specs = [
        _spec("600010.SH", list_date=fresh_list_date, first_day=fresh_list_date),
        _spec("600011.SH", list_date=seasoned_list_date, first_day=seasoned_list_date),
        _spec("600012.SH", list_date=DEFAULT_LIST_DATE),
    ]
    bars, instruments = _scenario(specs, days)
    got = set(select_members(bars, instruments, _no_st(), days, refresh_day)["instrument"])
    assert got == {"600011.SH", "600012.SH"}


def test_excluded_boards_are_dropped() -> None:
    days = _open_days(date(2023, 1, 2), date(2023, 6, 30))
    refresh_day = quarter_end_refresh_days(days, date(2023, 1, 1))[0]
    specs = [
        _spec("600000.SH"),
        _spec("000001.SZ"),
        _spec("300750.SZ"),
        _spec("301236.SZ"),
        _spec("688981.SH"),  # 科创板
        _spec("689009.SH"),  # 科创板
        _spec("830799.BJ"),  # 北交所
        _spec("430047.BJ"),  # 北交所
    ]
    bars, instruments = _scenario(specs, days)
    got = set(select_members(bars, instruments, _no_st(), days, refresh_day)["instrument"])
    assert got == {"600000.SH", "000001.SZ", "300750.SZ", "301236.SZ"}


def test_price_cap_boundary() -> None:
    days = _open_days(date(2023, 1, 2), date(2023, 6, 30))
    refresh_day = quarter_end_refresh_days(days, date(2023, 1, 1))[0]
    specs = [
        _spec("600000.SH", close=MAX_CLOSE),  # 恰好等于上限 → 保留
        _spec("600001.SH", close=MAX_CLOSE + 0.01),  # 高出上限 → 剔除
        _spec("600002.SH", close=3.5),
    ]
    bars, instruments = _scenario(specs, days)
    got = set(select_members(bars, instruments, _no_st(), days, refresh_day)["instrument"])
    assert got == {"600000.SH", "600002.SH"}


def test_refresh_day_without_quote_is_dropped() -> None:
    # 停牌（刷新日无行情行）的票取不到收盘价，价格过滤天然剔除。
    days = _open_days(date(2023, 1, 2), date(2023, 6, 30))
    refresh_day = quarter_end_refresh_days(days, date(2023, 1, 1))[0]
    specs = [_spec("600000.SH"), _spec("600001.SH")]
    bars, instruments = _scenario(specs, days)
    bars = bars.filter(
        ~((pl.col("instrument") == "600001.SH") & (pl.col("date") == refresh_day))
    )
    got = set(select_members(bars, instruments, _no_st(), days, refresh_day)["instrument"])
    assert got == {"600000.SH"}


def test_liquidity_threshold_boundary() -> None:
    days = _open_days(date(2023, 1, 2), date(2023, 6, 30))
    refresh_day = quarter_end_refresh_days(days, date(2023, 1, 1))[0]
    specs = [
        _spec("600000.SH", amount=MIN_AVG_AMOUNT),  # 恰好等于门槛 → 保留
        _spec("600001.SH", amount=MIN_AVG_AMOUNT - 1.0),  # 略低 → 剔除
    ]
    bars, instruments = _scenario(specs, days)
    got = set(select_members(bars, instruments, _no_st(), days, refresh_day)["instrument"])
    assert got == {"600000.SH"}


def test_liquidity_window_only_looks_back_60_trading_days() -> None:
    days = _open_days(date(2023, 1, 2), date(2023, 6, 30))
    refresh_day = quarter_end_refresh_days(days, date(2023, 1, 1))[0]
    index = days.index(refresh_day)
    window_start = days[index - AMOUNT_WINDOW + 1]
    specs = [
        # 窗口之前成交额极高、窗口内几乎没量：不能靠历史成交额混进来。
        _spec(
            "600001.SH",
            amount=lambda day: 5e9 if day < window_start else 1e6,
        ),
        _spec("600002.SH"),
    ]
    bars, instruments = _scenario(specs, days)
    got = set(select_members(bars, instruments, _no_st(), days, refresh_day)["instrument"])
    assert got == {"600002.SH"}


def test_suspended_days_count_as_zero_amount_in_window() -> None:
    # 窗口里停牌一半的票，分母仍是 60 个开市日，日均被折半。
    days = _open_days(date(2023, 1, 2), date(2023, 6, 30))
    refresh_day = quarter_end_refresh_days(days, date(2023, 1, 1))[0]
    index = days.index(refresh_day)
    window_days = days[index - AMOUNT_WINDOW + 1 : index + 1]
    resume = window_days[AMOUNT_WINDOW // 2]
    specs = [
        _spec("600001.SH", amount=1.5e8, first_day=resume),  # 只有半个窗口有行情
        _spec("600002.SH", amount=1.1e8),
    ]
    bars, instruments = _scenario(specs, days)
    got = set(select_members(bars, instruments, _no_st(), days, refresh_day)["instrument"])
    assert got == {"600002.SH"}


def test_top_n_truncation_keeps_most_liquid() -> None:
    days = _open_days(date(2023, 1, 2), date(2023, 6, 30))
    refresh_day = quarter_end_refresh_days(days, date(2023, 1, 1))[0]
    specs = [
        _spec("600000.SH", amount=3e8),
        _spec("600001.SH", amount=2e8),
        _spec("600002.SH", amount=1.5e8),
    ]
    bars, instruments = _scenario(specs, days)
    got = select_members(bars, instruments, _no_st(), days, refresh_day, top_n=2)
    assert got["instrument"].to_list() == ["600000.SH", "600001.SH"]
    assert select_members(
        bars, instruments, _no_st(), days, refresh_day
    ).height == len(specs) < TOP_N


# ---------------------------------------------------------------------------
# 池展开与 members 接口
# ---------------------------------------------------------------------------


def _two_quarter_scenario() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, list[date]]:
    """2023 全年工作日 + 两只票（其中一只 Q2 摘帽）的合成场景。"""
    days = _open_days(date(2023, 1, 2), date(2023, 12, 29))
    specs = [_spec("600000.SH"), _spec("600001.SH", close=35.0)]
    bars, instruments = _scenario(specs, days)
    st = _st_intervals([("600001.SH", date(2022, 1, 3), date(2023, 6, 15))])
    return bars, instruments, st, days


def test_build_pool_expands_members_to_every_open_day(tmp_path: Path) -> None:
    bars, instruments, st, days = _two_quarter_scenario()
    pool = build_pool(bars, instruments, st, days, start=date(2023, 1, 1))
    refresh_days = quarter_end_refresh_days(days, date(2023, 1, 1))

    assert pool.schema == pl.Schema({"date": pl.Date, "instrument": pl.String})
    assert pool.equals(pool.sort(["date", "instrument"]))
    # 首个刷新日之前没有成员行；从它起到数据末尾每个开市日都有行。
    expected_days = [day for day in days if day >= refresh_days[0]]
    assert pool["date"].unique().sort().to_list() == expected_days
    per_day = pool.group_by("date").len().sort("date")
    assert per_day["len"][0] == 1  # Q1 只有 600000.SH
    assert per_day["len"][-1] == 2  # 末段两只
    assert per_day["len"].max() == 2

    path = tmp_path / "pool.parquet"
    pool.write_parquet(path)
    assert members(str(path), refresh_days[0], data_dir=tmp_path) == {"600000.SH"}
    # 刷新日之后、下一刷新日之前的任意开市日取到同一批成员（PIT 有效期）。
    between = [day for day in days if refresh_days[0] < day < refresh_days[1]][3]
    assert members(str(path), between, data_dir=tmp_path) == {"600000.SH"}
    assert members(str(path), refresh_days[1], data_dir=tmp_path) == {
        "600000.SH",
        "600001.SH",
    }
    # 池生效前返回空集。
    assert members(str(path), date(2022, 12, 30), data_dir=tmp_path) == set()
    frame = members_range(
        str(path), refresh_days[0], refresh_days[0], data_dir=tmp_path
    )
    assert frame.height == 1
    assert frame.schema == pl.Schema({"date": pl.Date, "instrument": pl.String})


def test_build_pool_start_before_first_refresh_day_is_empty() -> None:
    bars, instruments, st, days = _two_quarter_scenario()
    pool = build_pool(bars, instruments, st, days, start=date(2099, 1, 1))
    assert pool.height == 0
    assert pool.schema == pl.Schema({"date": pl.Date, "instrument": pl.String})


def test_incomplete_refresh_day_is_skipped_and_previous_members_extend() -> None:
    days = _open_days(date(2023, 1, 2), date(2023, 6, 30))
    specs = [_spec(f"6000{index:02d}.SH") for index in range(10)]
    bars, instruments = _scenario(specs, days)
    # 末日本人只有一只票有行情（增量更新停在半天），该刷新日不应成立。
    last_day = days[-1]
    bars = bars.filter(
        (pl.col("date") < last_day) | (pl.col("instrument") == "600000.SH")
    )
    refresh_days = quarter_end_refresh_days(days, date(2023, 1, 1))
    assert refresh_days[-1] == last_day

    usable = refresh_days_in_use(bars, days, start=date(2023, 1, 1))
    assert usable == [refresh_days[0]]

    pool = build_pool(bars, instruments, _no_st(), days, start=date(2023, 1, 1))
    assert pool["date"].max() == last_day
    # 末段沿用 Q1 的成员，不因数据残缺少选票。
    last_members = pool.filter(pl.col("date") == last_day)["instrument"].n_unique()
    first_members = pool.filter(pl.col("date") == refresh_days[0])["instrument"].n_unique()
    assert last_members == first_members == len(specs)


def test_build_pool_skips_boards_and_st_together() -> None:
    days = _open_days(date(2023, 1, 2), date(2023, 6, 30))
    specs = [
        _spec("600000.SH"),
        _spec("688981.SH"),
        _spec("830799.BJ"),
        _spec("600009.SH", close=100.0),
    ]
    bars, instruments = _scenario(specs, days)
    st = _st_intervals([("600000.SH", date(2023, 1, 1), None)])
    pool = build_pool(bars, instruments, st, days, start=date(2023, 1, 1))
    assert pool["instrument"].unique().to_list() == []
    assert pool.height == 0


def test_build_pool_respects_top_n() -> None:
    days = _open_days(date(2023, 1, 2), date(2023, 6, 30))
    specs = [
        _spec("600000.SH", amount=3e8),
        _spec("600001.SH", amount=2e8),
        _spec("600002.SH", amount=1.5e8),
    ]
    bars, instruments = _scenario(specs, days)
    pool = build_pool(
        bars, instruments, _no_st(), days, start=date(2023, 1, 1), top_n=2
    )
    first_day = pool["date"].min()
    assert (
        pool.filter(pl.col("date") == first_day)["instrument"].to_list()
        == ["600000.SH", "600001.SH"]
    )
    assert pool["instrument"].n_unique() == 2


# ---------------------------------------------------------------------------
# 策略配置
# ---------------------------------------------------------------------------


def test_flagship_strategy_config_loads_and_builds_optimizer() -> None:
    config = load_strategy_config(FLAGSHIP_STRATEGY)
    assert config.strategy == "stock_selection"
    assert config.universe == "config/universe/flagship_pool.parquet"
    assert config.benchmark == "000300"
    assert config.top_k == 15
    assert config.rebalance_freq == "W"
    assert config.optimize == {"w_max": 0.1, "max_turnover": 0.6, "kappa": 0.002}

    optimizer = build_stock_optimizer(config)
    assert optimizer.w_max == pytest.approx(0.1)
    assert optimizer.max_turnover == pytest.approx(0.6)
    assert optimizer.kappa == pytest.approx(0.002)


def test_flagship_strategy_config_matches_backtest_cli() -> None:
    """``--strategy-config`` 与命令行五项同名参数必须逐项一致。"""
    from scripts.backtest_e2e import E2EConfig, E2EError

    common = {
        "data_dir": Path("data"),
        "model_dir": Path("runs/automl/flagship"),
        "out_dir": Path("runs/flagship"),
        "start": date(2022, 4, 1),
        "end": date(2026, 9, 28),
        "strategy": "stock_selection",
        "universe": "config/universe/flagship_pool.parquet",
        "benchmark": "000300",
        "top_k": 15,
        "rebalance_freq": "W",
        "strategy_config_path": FLAGSHIP_STRATEGY,
    }
    loaded = E2EConfig(**common).resolved_strategy_config()
    assert loaded.top_k == 15
    assert loaded.universe == "config/universe/flagship_pool.parquet"
    assert loaded.optimize["w_max"] == pytest.approx(0.1)

    with pytest.raises(E2EError, match="不一致"):
        E2EConfig(**{**common, "top_k": 50}).resolved_strategy_config()


def test_committed_flagship_pool_parses_as_dynamic_pool(tmp_path: Path) -> None:
    assert FLAGSHIP_POOL.exists(), "先跑 scripts/build_flagship_universe.py 生成池文件"
    frame = members_range(
        str(FLAGSHIP_POOL), date(2022, 3, 31), date(2022, 3, 31), data_dir=tmp_path
    )
    assert 0 < frame.height <= TOP_N
    assert frame.schema == pl.Schema({"date": pl.Date, "instrument": pl.String})
    # 池生效之前（POOL_START 早于首刷新日）没有任何成员。
    before = members(
        str(FLAGSHIP_POOL), date(2022, 3, 30), data_dir=tmp_path
    )
    assert before == set()
    assert POOL_START == date(2022, 1, 1)
