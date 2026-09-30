"""个人交易者旗舰策略的股票池构建 CLI（issue #113）。

样板口径：本金 10 万、``stock_selection``、周频调仓、``top_k`` 15 的集中持仓。
整手（100 股/手）约束下 10 万本金单票预算约 6600 元，高价股一手就顶掉整票预算，
因此池在流动性之外额外卡了一道价格上限，收窄到 200~300 只。

构造口径
--------
- **刷新频率**：每季度末一个刷新日（3/6/9/12 月的最后一个交易日），成员有效期到
  下一刷新日。刷新日集合由 ``data/calendar.parquet`` 的开市日推导，不用自然日。
- **样本区间**：``POOL_START``（2022-01-01）起到数据末尾。首刷新日为 2022-03-31，
  此时日历里已有 121 个开市日，末 60 个正好覆盖到 2021-12-30，落在数据起点
  （2021-09-29）之后，成交额窗口与上市天数窗口都不缺数据。
- **刷新日可用性**：刷新日当日行情覆盖不足前一开市日的
  :data:`MIN_REFRESH_COVERAGE_RATIO` 时判定当天数据没抓完（增量更新停在半天就是
  这个形态），该刷新日不成立，成员沿用上一刷新日。数据补齐后重跑本脚本即可。
- **过滤**（全部只用刷新日及之前的数据，无前视）：

  1. 剔除 ST：按 ``data/st_intervals.parquet`` 在刷新日处于 ST 区间的证券。
     该表是 CSMAR 逐日交易状态折叠出的 PIT 区间，不拿当前名称回填历史（用名称
     过滤等于把「将来会 ST」的信息提前用上）。已知覆盖范围限于主板，创业板 ST
     不在表内。
  2. 剔除刷新日上市不足 ``MIN_LISTED_TRADING_DAYS`` 个交易日的。上市日在日历
     窗口之前的证券不受此限——窗口内的计数被日历截断，不代表真实上市天数。
  3. 剔除科创板（``688`` / ``689`` 开头）与北交所（``.BJ`` 后缀，含 4/8/92 开头），
     保留沪深主板与创业板：整手与流动性口径都按这两块取。
  4. 刷新日收盘价 ``<= MAX_CLOSE``（80 元）。10 万本金 top_k 15 时单票预算约
     6600 元，80 元/股 × 100 股 = 8000 元仍偏紧但可接受，超过就一手都买不起。
  5. 过去 ``AMOUNT_WINDOW``（60）个交易日日均成交额 ``>= MIN_AVG_AMOUNT``
     （1 亿元）。分母是窗口内的开市日数，停牌日按成交额 0 计入，避免「停牌一半、
     活跃日成交额高」的票被算成高流动性。
  6. 通过的候选按 60 日日均成交额降序取前 ``TOP_N``（300）只。

输出
----
``config/universe/flagship_pool.parquet``，两列 ``(date, instrument)``，每个开市日
一行一只成员——``quant.universe.members`` 的自定义动态池按 ``day`` 精确匹配，
稀疏的「只有刷新日有行」写法会让非刷新日返回空集。``instrument`` 沿用数据层的
``normalize_instrument`` 形式（``600000.SH``），文件按 ``(date, instrument)`` 排序。

用法::

    uv run python scripts/build_flagship_universe.py \
        --data-dir C:\\Users\\陶唐\\Workspace\\ASTER\\data

配套策略配置 ``config/strategy-flagship.json``。``backtest_e2e.py`` 给了
``--strategy-config`` 时会逐项核对命令行参数与 JSON，故调用方式为::

    uv run python scripts/backtest_e2e.py \
        --strategy-config config/strategy-flagship.json \
        --strategy stock_selection \
        --universe config/universe/flagship_pool.parquet \
        --benchmark 000300 --top-k 15 --rebalance-freq W \
        --initial-cash 100000 --data-dir data \
        --model-dir runs/automl/<旗舰模型> \
        --start 2022-04-01 --end 2026-09-28 --out-dir runs/flagship

``strategy`` / ``universe`` / ``benchmark`` / ``top_k`` / ``rebalance_freq`` 五项必须
与 JSON 完全一致，缺一项即报不一致。训练配置的 ``universe`` 也要同值。
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from quant.data import cache  # noqa: E402
from quant.data.limit import ST_INTERVALS  # noqa: E402

logger = logging.getLogger("build_flagship_universe")

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

#: 池的起始日：首刷新日取该日之后第一个季度末交易日。
POOL_START: date = date(2022, 1, 1)

#: 刷新月份（季度末）。
REFRESH_MONTHS: tuple[int, ...] = (3, 6, 9, 12)

#: 刷新日上市不足该交易日数的证券剔除。
MIN_LISTED_TRADING_DAYS: int = 120

#: 日均成交额的窗口交易日数。
AMOUNT_WINDOW: int = 60

#: 60 日日均成交额下限（元）。
MIN_AVG_AMOUNT: float = 1e8

#: 刷新日收盘价上限（元）：超过则一手买不起。
MAX_CLOSE: float = 80.0

#: 每个刷新日保留的成员数上限。
TOP_N: int = 300

#: 刷新日行情覆盖下限：当日有行情的证券数低于前一开市日的该比例时，视为数据没抓完，
#: 该刷新日不成立（成员沿用上一刷新日，避免用半天的行情选出「今天的池」）。
MIN_REFRESH_COVERAGE_RATIO: float = 0.8

#: 剔除板块：科创板代码段与北交所后缀。
EXCLUDED_PREFIXES: tuple[str, ...] = ("688", "689")
EXCLUDED_SUFFIX: str = ".BJ"

#: 默认数据目录与产出路径。
DEFAULT_DATA_DIR: str = "data"
DEFAULT_OUT: Path = Path("config/universe/flagship_pool.parquet")

#: ST 区间表文件名（与 ``scripts/build_st_intervals_from_csmar.py`` 约定一致）。
ST_INTERVALS_FILE: str = "st_intervals.parquet"

#: 行情读取后只保留的四列，控制 10 年全市场行情的内存占用。
BAR_COLUMNS: tuple[str, ...] = ("date", "instrument", "close", "amount")

#: 池文件 schema：自定义动态池的最小约定。
OUTPUT_SCHEMA: pl.Schema = pl.Schema({"date": pl.Date, "instrument": pl.String})

#: 候选明细 schema（仅中间结果与打印用，不落盘）。
CANDIDATE_SCHEMA: pl.Schema = pl.Schema(
    {
        "instrument": pl.String,
        "avg_amount": pl.Float64,
        "close": pl.Float64,
    }
)


# ---------------------------------------------------------------------------
# 刷新日与窗口
# ---------------------------------------------------------------------------


def quarter_end_refresh_days(open_days: Sequence[date], start: date) -> list[date]:
    """``start`` 之后（含）每个季度的最后一个开市日，按时间升序。"""
    latest: dict[tuple[int, int], date] = {}
    for day in sorted(open_days):
        if day < start or day.month not in REFRESH_MONTHS:
            continue
        # 只认季度末月，且落在该季度的最后一个月里；用月份直接分桶。
        key = (day.year, (day.month - 1) // 3)
        current = latest.get(key)
        if current is None or day > current:
            latest[key] = day
    return [latest[key] for key in sorted(latest)]


def _window_start(open_days: Sequence[date], refresh_day: date, window: int) -> date:
    """刷新日往前 ``window`` 个开市日（含刷新日）的窗口首日。"""
    index = open_days.index(refresh_day)
    return open_days[max(0, index - window + 1)]


def _empty_candidates() -> pl.DataFrame:
    return pl.DataFrame(schema=CANDIDATE_SCHEMA)


# ---------------------------------------------------------------------------
# 刷新日可用性
# ---------------------------------------------------------------------------


def _day_coverage(bars: pl.DataFrame) -> dict[date, int]:
    """每个开市日有行情的证券数。"""
    if bars.height == 0:
        return {}
    return {
        row["date"]: int(row["len"])
        for row in bars.group_by("date").len().iter_rows(named=True)
    }


def refresh_days_in_use(
    bars: pl.DataFrame, open_days: Sequence[date], *, start: date = POOL_START
) -> list[date]:
    """季度末刷新日中数据完整的那些。

    刷新日当日有行情的证券数不足前一开市日的 :data:`MIN_REFRESH_COVERAGE_RATIO`
    时判定当天数据没抓完（增量更新跑到一半就是这种形态），该刷新日不成立：成员沿用
    上一刷新日，最后一段自动延到数据末尾。数据完整时返回全部季度末刷新日。
    """
    days = quarter_end_refresh_days(open_days, start)
    coverage = _day_coverage(bars)
    if not coverage:
        return []
    usable: list[date] = []
    for refresh_day in days:
        index = open_days.index(refresh_day)
        previous = open_days[index - 1] if index > 0 else None
        reference = coverage.get(previous, 0) if previous is not None else 0
        today = coverage.get(refresh_day, 0)
        if reference > 0 and today < reference * MIN_REFRESH_COVERAGE_RATIO:
            logger.warning(
                "刷新日 %s 有行情的证券 %d 只，不足前一开市日 %s 的 %d 只的 %.0f%%，"
                "判定数据不完整并跳过（数据补齐后重跑本脚本）",
                refresh_day,
                today,
                previous,
                reference,
                MIN_REFRESH_COVERAGE_RATIO * 100.0,
            )
            continue
        usable.append(refresh_day)
    return usable


# ---------------------------------------------------------------------------
# 单刷新日选样
# ---------------------------------------------------------------------------


def select_members(
    bars: pl.DataFrame,
    instruments: pl.DataFrame,
    st_intervals: pl.DataFrame,
    open_days: Sequence[date],
    refresh_day: date,
    *,
    top_n: int = TOP_N,
) -> pl.DataFrame:
    """按刷新日选出通过全部过滤的候选，返回 ``(instrument, avg_amount, close)``。

    只用 ``date <= refresh_day`` 的行情：成交额窗口取刷新日往前
    :data:`AMOUNT_WINDOW` 个开市日，收盘价取刷新日当日。按 ``avg_amount`` 降序
    截断到 ``top_n``，同额时按代码升序，保证可复现。
    """
    open_array = np.asarray(open_days, dtype="datetime64[D]")
    refresh_ordinal = int(np.searchsorted(open_array, np.datetime64(refresh_day)))
    if refresh_ordinal >= open_array.shape[0] or open_array[refresh_ordinal] != np.datetime64(
        refresh_day
    ):
        raise ValueError(f"刷新日 {refresh_day} 不在开市日序列内")

    window_start = _window_start(open_days, refresh_day, AMOUNT_WINDOW)
    window_days = refresh_ordinal - open_days.index(window_start) + 1

    window = bars.filter(
        (pl.col("date") >= window_start) & (pl.col("date") <= refresh_day)
    )
    if window.height == 0:
        return _empty_candidates()

    traded = window.group_by("instrument").agg(
        (pl.col("amount").fill_null(0.0).sum() / window_days).alias("avg_amount")
    )
    on_refresh_day = window.filter(pl.col("date") == refresh_day).select(
        "instrument", pl.col("close").alias("close")
    )
    candidates = on_refresh_day.join(traded, on="instrument", how="inner").join(
        instruments.select("instrument", "list_date"),
        on="instrument",
        how="inner",
    )
    # 上市日到刷新日之间的开市日数：searchsorted 给出窗口内早于上市日的开市日个数。
    listed_days = refresh_ordinal + 1 - np.searchsorted(
        open_array, candidates["list_date"].to_numpy(), side="left"
    )
    candidates = candidates.with_columns(
        # 上市日在日历窗口之前时，窗口内计数被日历截断，不参与 120 日判定。
        (pl.col("list_date") >= open_days[0]).alias("_listed_in_window"),
        pl.Series("listed_days", listed_days),
    )

    st_now = set(
        st_intervals.filter(
            (pl.col("start_date") <= refresh_day)
            & (pl.col("end_date").is_null() | (pl.col("end_date") >= refresh_day))
        )["instrument"].to_list()
    )

    keep = (
        (pl.col("close") > 0.0)
        & (pl.col("close") <= MAX_CLOSE)
        & (pl.col("avg_amount") >= MIN_AVG_AMOUNT)
        & (~pl.col("instrument").str.slice(0, 3).is_in(list(EXCLUDED_PREFIXES)))
        & (~pl.col("instrument").str.ends_with(EXCLUDED_SUFFIX))
        & (~pl.col("instrument").is_in(st_now))
        & (
            (pl.col("listed_days") >= MIN_LISTED_TRADING_DAYS)
            | (~pl.col("_listed_in_window"))
        )
    )
    return (
        candidates.filter(keep)
        .sort(["avg_amount", "instrument"], descending=[True, False])
        .head(top_n)
        .select(*CANDIDATE_SCHEMA.keys())
    )


# ---------------------------------------------------------------------------
# 池构建
# ---------------------------------------------------------------------------


def build_pool(
    bars: pl.DataFrame,
    instruments: pl.DataFrame,
    st_intervals: pl.DataFrame,
    open_days: Sequence[date],
    *,
    start: date = POOL_START,
    top_n: int = TOP_N,
    refresh_days: Sequence[date] | None = None,
) -> pl.DataFrame:
    """构建 PIT 动态池，返回 ``(date, instrument)`` 两列，按 ``(date, instrument)`` 排序。

    每个刷新日选出的成员展开到 ``[刷新日, 下一刷新日)`` 的每个开市日；最后一个
    刷新日展开到开市日序列末尾。``start`` 之前不会有任何行——自定义动态池按日精确
    匹配，``start`` 之前的日期查池返回空集。

    ``refresh_days`` 缺省取 :func:`refresh_days_in_use`（已剔除数据不完整的刷新日）。
    """
    days_in_use = (
        list(refresh_days)
        if refresh_days is not None
        else refresh_days_in_use(bars, open_days, start=start)
    )
    if not days_in_use:
        logger.warning("开市日序列在 %s 之后没有可用的刷新日", start)
        return pl.DataFrame(schema=OUTPUT_SCHEMA)

    open_frame = pl.DataFrame({"date": list(open_days)}, schema={"date": pl.Date})
    frames: list[pl.DataFrame] = []
    for index, refresh_day in enumerate(days_in_use):
        members = select_members(
            bars, instruments, st_intervals, open_days, refresh_day, top_n=top_n
        )
        upper_bound = days_in_use[index + 1] if index + 1 < len(days_in_use) else None
        condition = pl.col("date") >= refresh_day
        if upper_bound is not None:
            condition = condition & (pl.col("date") < upper_bound)
        days = open_frame.filter(condition)
        logger.info(
            "刷新日 %s：候选 %d 只，展开 %d 个开市日",
            refresh_day,
            members.height,
            days.height,
        )
        frames.append(days.join(members.select("instrument"), how="cross"))
    return (
        pl.concat(frames, how="vertical")
        .unique()
        .sort(["date", "instrument"])
        .cast(OUTPUT_SCHEMA)
    )


# ---------------------------------------------------------------------------
# 校验与打印
# ---------------------------------------------------------------------------


def _check_normalized(frame: pl.DataFrame) -> None:
    """池的 ``instrument`` 必须是 ``normalize_instrument`` 形式，否则拒绝落盘。"""
    bad = frame.filter(
        ~pl.col("instrument").str.contains(r"^\d{6}\.(SH|SZ|BJ)$")
    )
    if bad.height:
        raise ValueError(
            f"instrument 未归一化（应为 600000.SH 形式）：{bad['instrument'].head(5).to_list()}"
        )


def _load_st_intervals(data_dir: Path) -> pl.DataFrame:
    path = data_dir / ST_INTERVALS_FILE
    if not path.exists():
        raise FileNotFoundError(f"缺少 ST 区间表 {path}，先跑 scripts/build_st_intervals_from_csmar.py")
    return pl.read_parquet(path).select(*ST_INTERVALS.keys()).cast(ST_INTERVALS)


def _summarize(
    pool: pl.DataFrame,
    instruments: pl.DataFrame,
    data_dir: Path,
    refresh_days: Sequence[date],
) -> None:
    """打印池规模、板块分布与末刷新日的行业前若干名。"""
    print("\n# 池规模")
    print(
        f"  总计 {pool.height} 行，{pool['instrument'].n_unique()} 只证券，"
        f"{pool['date'].min()} ~ {pool['date'].max()}"
    )
    print(f"  刷新日 {len(refresh_days)} 个：{refresh_days[0]} ~ {refresh_days[-1]}")
    print("  刷新日成员数：")
    per_day = (
        pool.group_by("date")
        .agg(pl.col("instrument").n_unique().alias("n"))
        .sort("date")
    )
    refresh_dates = set(refresh_days)
    for row in per_day.iter_rows(named=True):
        if row["date"] in refresh_dates:
            print(f"    {row['date']}: {row['n']} 只")

    last_day = refresh_days[-1]
    last_members = pool.filter(pl.col("date") == last_day).select("instrument")
    board = (
        last_members.join(
            instruments.select("instrument", "board"), on="instrument", how="left"
        )
        .group_by("board")
        .len()
        .sort("len", descending=True)
    )
    print(f"\n# 末刷新日 {last_day} 板块分布")
    for row in board.iter_rows(named=True):
        print(f"  {row['board']}: {row['len']} 只")

    industry = cache.load_industry(data_dir)
    if industry.height:
        top = (
            last_members.join(
                industry.select("instrument", "industry_l1"),
                on="instrument",
                how="left",
            )
            .group_by("industry_l1")
            .len()
            .sort("len", descending=True)
            .head(10)
        )
        print(
            "  行业前 10（东财一级）："
            + ", ".join(
                f"{row['industry_l1']}={row['len']}"
                for row in top.iter_rows(named=True)
            )
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="构建个人交易者旗舰策略的 PIT 股票池")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="缓存目录，默认 data/")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help=f"产出路径，默认 {DEFAULT_OUT}")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划不落盘")
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        pass

    args = _build_parser().parse_args(argv)
    data_dir = Path(args.data_dir)
    out_path = Path(args.out)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    open_days = (
        cache.load_calendar(data_dir)
        .filter(pl.col("is_open"))
        .sort("date")["date"]
        .to_list()
    )
    if not open_days:
        logger.error("%s 的日历里没有开市日", data_dir)
        return 2

    refresh_days = quarter_end_refresh_days(open_days, POOL_START)
    if not refresh_days:
        logger.error("开市日序列在 %s 之后没有季度末刷新日", POOL_START)
        return 2
    window_start = _window_start(open_days, refresh_days[0], AMOUNT_WINDOW)

    print(f"# build_flagship_universe data_dir={data_dir} out={out_path}")
    print(
        f"开市日 {len(open_days)} 个（{open_days[0]} ~ {open_days[-1]}），"
        f"季度末刷新日 {len(refresh_days)} 个（{refresh_days[0]} ~ {refresh_days[-1]}），"
        f"成交额窗口自 {window_start}"
    )
    if args.dry_run:
        print("dry-run：不读取行情、不落盘")
        return 0

    instruments = cache.load_instruments(data_dir)
    if not (data_dir / ST_INTERVALS_FILE).exists():
        logger.error(
            "缺少 ST 区间表 %s，先跑 scripts/build_st_intervals_from_csmar.py",
            data_dir / ST_INTERVALS_FILE,
        )
        return 2
    st_intervals = _load_st_intervals(data_dir)
    bars = cache.load_bars(data_dir, start=window_start).select(*BAR_COLUMNS)

    usable = refresh_days_in_use(bars, open_days)
    if not usable:
        logger.error("没有数据完整的刷新日，检查 %s 的行情是否抓全", data_dir)
        return 2
    pool = build_pool(bars, instruments, st_intervals, open_days, refresh_days=usable)
    _check_normalized(pool)
    _summarize(pool, instruments, data_dir, usable)

    cache._atomic_write_parquet(out_path, pool)
    print(f"\n已写入 {out_path}（{pool.height} 行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
