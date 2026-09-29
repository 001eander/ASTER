"""缓存数据体检：完整性 / 异常值 / 复权一致性。

用途
----
跑批前对 ``data/`` 缓存做一次结构化体检（AGENTS.md 量化纪律第 6 条），
:func:`validate` 返回 :class:`ValidationReport`，调用方看 ``report.ok`` 决定是否中止。
三类检查：

1. **完整性**：上市未退市证券在开市日是否有行情行（停牌允许缺行，整票无数据报警）；
   全市场每日覆盖率；单票相邻行情之间的开市日缺口；未退市证券的东财行业覆盖率
   （issue #65，低于 :data:`INDUSTRY_COVERAGE_MIN_RATIO` 报 error，缺失清单报 warning）。
2. **异常值**：价格非正、``high < low``、``open`` / ``close`` 越界、``vwap`` 越界、
   涨跌幅越过涨跌停带、``volume`` / ``amount`` 矛盾。
3. **复权一致性**：``adjfactor`` 非正、跳变是否由公司行为解释、后复权因子是否非递减。

severity 约定
-------------
- ``error``：数据不可能合法，调用方应硬中止。价格 / 成交量非正、``high < low``、
  重复键、schema 不符、涨跌幅越过涨跌停带容差、``adjfactor <= 0``、
  未退市证券行业覆盖率低于 :data:`INDUSTRY_COVERAGE_MIN_RATIO`。
- ``warning``：需要人判断的。覆盖率偏低、缺口、整票无数据、``vwap`` 越界、
  无法由公司行为解释的因子跳变、因子递减、缺行业归属的证券清单。

口径与取舍
----------
- 涨跌幅用**后复权收益** ``close × adjfactor / (prev_close × prev_adjfactor) - 1`` 判断，
  避免除权日未复权价格的跳空造成误报。``adjfactor`` 发生变化的那天（除权或因子基准
  跳变）需要除权参考价才能还原涨跌停带，本模块直接跳过这些行，交由复权一致性检查。
- 上限优先取 ``limit_up`` / ``limit_down``（已预计算时），否则按板块通用上限
  （常量直接复用 ``quant.data.limit``，保证与预计算同源）。
- 新股上市首日无前收（``prev_close`` 为 null）自然跳过；科创板 / 创业板 / 北交所
  上市初期无涨跌幅限制，用 ``list_date`` 之后的宽限窗口跳过，避免误报。
- ``instruments.list_date`` 为 null 时视为「窗口内一直可交易」，分母偏大，
  方向是覆盖率偏保守（可能多报）。
- 缺口口径：相邻两行的开市日间隔超过 :data:`GAP_OPEN_DAYS` 即报，长期停牌不单独
  豁免，由人看区间判断。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

import polars as pl

from quant.data import cache
from quant.data.limit import (
    BJ_LIMIT_RATIO,
    CYB_LIMIT_RATIO,
    CYB_REFORM_DATE,
    KCB_LIMIT_RATIO,
    MAIN_LIMIT_RATIO,
)
from quant.data.schema import (
    CORPORATE_ACTIONS,
    DAILY_BARS,
    INDUSTRY,
    INSTRUMENT_INFO,
    TRADE_CALENDAR,
    SchemaError,
    board_of,
    check_schema,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

#: 全市场每日行情覆盖率下限，低于此值报 warning。
COVERAGE_MIN_RATIO: float = 0.80
#: 未退市证券的行业覆盖率下限（issue #65），低于此值报 error。
INDUSTRY_COVERAGE_MIN_RATIO: float = 0.99
#: 单票相邻行情间隔超过这么多个开市日即视为缺口。
GAP_OPEN_DAYS: int = 10
#: 涨跌幅越过涨跌停带的容差（比例）。
LIMIT_TOLERANCE: float = 0.01
#: vwap 落在 [low, high] 的容差（比例）。
VWAP_TOLERANCE: float = 0.01
#: 价格浮点比较的绝对容差（元）。
PRICE_EPS: float = 1e-6
#: 相邻两日 adjfactor 比值偏离 1 超过此值视为跳变。
ADJ_TOLERANCE: float = 0.001
#: 「整票无数据」判定的最小应交易日数，避免对窗口极短的票误报。
MIN_EXPECTED_DAYS_NO_DATA: int = 5
#: 最近窗口模式下向前多读的天数，用于给 shift(1) 与缺口检查留前文。
LOOKBACK_PAD_DAYS: int = 400
#: 新股上市后无涨跌幅限制的宽限自然日数（按板块）。全面注册制后各板块新股
#: 上市初期均无涨跌幅限制（前 5 个交易日），统一宽限 14 个自然日。
NEW_LISTING_GRACE_DAYS: dict[str, int] = {"main": 14, "cyb": 14, "kcb": 14, "bj": 14}
#: 无涨跌幅限制日的行为阈值：全市场最大带宽是北交所 30%，|后复权收益| 超过
#: 此值加裕量在任何涨跌幅制度下都不可能发生（退市整理首日 / 复牌首日 / 新股
#: 窗口外的特殊安排），豁免 limit_move 的 error 并降级为 warning 供人工抽查。
UNRESTRICTED_BAND: float = 0.35
#: 汇总消息里最多列出的样例条数。
MAX_EXAMPLES: int = 3

Severity = Literal["error", "warning"]


# ---------------------------------------------------------------------------
# 报告结构
# ---------------------------------------------------------------------------


@dataclass
class ValidationIssue:
    """一条体检结论。``count`` 为涉及行数（按检查口径可能是证券数或天数）。"""

    severity: Severity
    check: str
    message: str
    count: int


@dataclass
class ValidationReport:
    """一次体检的结果。``ok`` 为 True 表示没有 error 级 issue。"""

    issues: list[ValidationIssue] = field(default_factory=list)
    checked_rows: int = 0
    checked_instruments: int = 0

    @property
    def ok(self) -> bool:
        """无 error 级 issue 即为通过，warning 不阻塞跑批。"""
        return not any(issue.severity == "error" for issue in self.issues)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]


def _add(
    issues: list[ValidationIssue],
    severity: Severity,
    check: str,
    message: str,
    count: int,
) -> None:
    issues.append(
        ValidationIssue(
            severity=severity, check=check, message=message, count=int(count)
        )
    )


# ---------------------------------------------------------------------------
# 读取：不走 cache.load_* ，以便把 schema / 重复键问题降级为报告项而不是抛异常
# ---------------------------------------------------------------------------


def _read_table(
    data_dir: Path,
    filename: str,
    schema: pl.Schema,
    name: str,
    issues: list[ValidationIssue],
    *,
    required: bool = True,
) -> pl.DataFrame:
    """读一个基础 parquet 表；缺失 / 读不动 / schema 不符只记账。"""
    path = data_dir / filename
    empty = pl.DataFrame(schema=schema)
    if not path.exists():
        _add(
            issues,
            "error" if required else "warning",
            f"{name}_missing",
            f"缺少 {name} 文件：{path}",
            0,
        )
        return empty
    try:
        raw = pl.read_parquet(path)
    except Exception as exc:  # noqa: BLE001 - 损坏文件记为体检结论
        _add(issues, "error", f"{name}_unreadable", f"{path} 读取失败：{exc}", 0)
        return empty
    try:
        check_schema(raw, schema, name=name)
    except SchemaError as exc:
        _add(issues, "error", "schema_mismatch", str(exc), raw.height)
        try:
            raw = raw.select(list(schema.keys())).cast(schema)
        except Exception:  # noqa: BLE001 - 列不全无法继续
            return empty
    return raw.select(list(schema.keys())).cast(schema)


def _read_bars(
    data_dir: Path,
    issues: list[ValidationIssue],
    *,
    start: date | None = None,
    end: date | None = None,
) -> pl.DataFrame:
    """读行情年文件并拼接，重复键与 schema 问题记 error 后尽量继续。"""
    bars_dir = data_dir / cache.BARS_DIR
    empty = pl.DataFrame(schema=DAILY_BARS)
    if not bars_dir.exists():
        _add(issues, "error", "bars_missing", f"缺少行情目录：{bars_dir}", 0)
        return empty

    frames: list[pl.DataFrame] = []
    for path in sorted(bars_dir.glob("*.parquet")):
        try:
            year = int(path.stem)
        except ValueError:
            continue
        if start is not None and year < start.year:
            continue
        if end is not None and year > end.year:
            continue
        try:
            raw = pl.read_parquet(path)
        except Exception as exc:  # noqa: BLE001 - 损坏文件记为体检结论
            _add(issues, "error", "bars_unreadable", f"{path} 读取失败：{exc}", 0)
            continue
        try:
            check_schema(raw, DAILY_BARS, name=f"bars/{year}")
        except SchemaError as exc:
            _add(issues, "error", "schema_mismatch", str(exc), raw.height)
            try:
                raw = raw.select(list(DAILY_BARS.keys())).cast(DAILY_BARS)
            except Exception:  # noqa: BLE001 - 列不全无法继续
                continue
        frames.append(raw)

    if not frames:
        _add(issues, "error", "bars_empty", f"{bars_dir} 下没有可用的行情文件", 0)
        return empty

    df = (
        pl.concat(frames, how="vertical_relaxed")
        .select(list(DAILY_BARS.keys()))
        .cast(DAILY_BARS)
    )
    if start is not None:
        df = df.filter(pl.col("date") >= start)
    if end is not None:
        df = df.filter(pl.col("date") <= end)

    duplicated = int(df.select(["instrument", "date"]).is_duplicated().sum())
    if duplicated:
        _add(
            issues,
            "error",
            "duplicate_key",
            f"日线存在重复的 (instrument, date)：{duplicated} 行",
            duplicated,
        )
        df = df.unique(subset=["instrument", "date"], keep="last")
    return df.sort(["instrument", "date"])


# ---------------------------------------------------------------------------
# 窗口与上下文
# ---------------------------------------------------------------------------


def _open_days(calendar: pl.DataFrame, end: date | None) -> pl.DataFrame:
    """日历里的开市日（date 升序去重），``end`` 截断。"""
    if calendar.height == 0:
        return pl.DataFrame(schema=TRADE_CALENDAR)
    days = calendar.filter(pl.col("is_open")).select("date")
    if end is not None:
        days = days.filter(pl.col("date") <= end)
    return days.unique().sort("date")


def _annotate(bars: pl.DataFrame, open_days: pl.DataFrame | None) -> pl.DataFrame:
    """给行情补开市日序号与同证券前值，供缺口 / 涨跌幅 / 复权检查使用。"""
    out = bars.sort(["instrument", "date"])
    if open_days is not None and open_days.height:
        ordinals = open_days.select(
            pl.col("date"),
            pl.int_range(pl.len(), dtype=pl.Int64).alias("_ord"),
        )
        out = out.join(ordinals, on="date", how="left")
    else:
        out = out.with_columns(pl.lit(None, dtype=pl.Int64).alias("_ord"))
    return out.with_columns(
        pl.col("_ord").shift(1).over("instrument", order_by="date").alias("_prev_ord"),
        pl.col("date").shift(1).over("instrument", order_by="date").alias("_prev_date"),
        pl.col("close").shift(1).over("instrument", order_by="date").alias("_prev_close"),
        pl.col("adjfactor")
        .shift(1)
        .over("instrument", order_by="date")
        .alias("_prev_adj"),
    )


def _examples(df: pl.DataFrame, columns: tuple[str, ...] = ("instrument", "date")) -> str:
    """取前几行拼成 ``a@b`` 形式的样例串。"""
    if df.height == 0:
        return ""
    rows = df.select(*columns).head(MAX_EXAMPLES).iter_rows()
    return ", ".join(" @ ".join(str(item) for item in row) for row in rows)


def _board_of_safe(instrument: str) -> str:
    try:
        return board_of(instrument)
    except ValueError:
        return "main"


def _check_completeness(
    issues: list[ValidationIssue],
    window: pl.DataFrame,
    win: pl.DataFrame,
    instruments: pl.DataFrame,
) -> None:
    """完整性：整票无数据、每日覆盖率、单票开市日缺口。"""
    if window.height == 0:
        return
    inst = instruments.select("instrument", "list_date", "delist_date")
    cross = window.join(inst, how="cross").filter(
        (pl.col("list_date").is_null() | (pl.col("list_date") <= pl.col("date")))
        & (pl.col("delist_date").is_null() | (pl.col("delist_date") >= pl.col("date")))
    )
    expected_inst = cross.group_by("instrument").agg(pl.len().alias("_expected"))
    expected_day = cross.group_by("date").agg(pl.len().alias("_eligible")).sort("date")

    actual = win.group_by("instrument").agg(
        pl.col("date").n_unique().alias("_actual_days")
    )
    no_data = (
        expected_inst.filter(pl.col("_expected") >= MIN_EXPECTED_DAYS_NO_DATA)
        .join(actual, on="instrument", how="left")
        .filter(pl.col("_actual_days").is_null() | (pl.col("_actual_days") == 0))
    )
    if no_data.height:
        _add(
            issues,
            "warning",
            "instrument_no_data",
            "窗口内应交易但完全无行情的证券 "
            f"{no_data.height} 只：{_examples(no_data, ('instrument',))}",
            no_data.height,
        )

    traded = win.group_by("date").agg(
        pl.col("instrument").n_unique().alias("_traded")
    )
    coverage = (
        expected_day.join(traded, on="date", how="left")
        .with_columns(pl.col("_traded").fill_null(0))
        .with_columns((pl.col("_traded") / pl.col("_eligible")).alias("_ratio"))
    )
    low = coverage.filter(pl.col("_ratio") < COVERAGE_MIN_RATIO)
    if low.height:
        worst = low.sort("_ratio").row(0, named=True)
        _add(
            issues,
            "warning",
            "coverage_low",
            f"{low.height} 个开市日行情覆盖率低于 {COVERAGE_MIN_RATIO:.0%}，"
            f"最低 {worst['_ratio']:.1%}（{worst['date']} 仅 {worst['_traded']}/"
            f"{worst['_eligible']} 只）",
            low.height,
        )


def _check_industry_coverage(
    issues: list[ValidationIssue],
    instruments: pl.DataFrame,
    industry: pl.DataFrame,
) -> None:
    """未退市证券的东财行业覆盖率：低于阈值报 error，缺失清单报 warning。"""
    if instruments.height == 0:
        return
    active = instruments.filter(pl.col("delist_date").is_null()).select("instrument")
    if active.height == 0:
        return
    known = (
        industry.filter(pl.col("industry_l1").is_not_null())
        .select("instrument")
        .unique()
    )
    missing = active.join(known, on="instrument", how="anti").sort("instrument")
    ratio = 1.0 - missing.height / active.height
    if ratio < INDUSTRY_COVERAGE_MIN_RATIO:
        _add(
            issues,
            "error",
            "industry_coverage_low",
            f"未退市证券东财行业覆盖率 {ratio:.2%} 低于 "
            f"{INDUSTRY_COVERAGE_MIN_RATIO:.0%}："
            f"{missing.height}/{active.height} 只没有归属",
            missing.height,
        )
    if missing.height:
        _add(
            issues,
            "warning",
            "industry_uncovered",
            f"{missing.height} 只未退市证券缺东财行业归属："
            f"{_examples(missing, ('instrument',))}",
            missing.height,
        )


def _check_gaps(
    issues: list[ValidationIssue],
    win: pl.DataFrame,
) -> None:
    """单票相邻行情之间缺失的开市日数超过阈值即报。"""
    if "_prev_ord" not in win.columns or "_ord" not in win.columns:
        return
    gaps = win.filter(
        pl.col("_ord").is_not_null()
        & pl.col("_prev_ord").is_not_null()
        & (pl.col("_ord") - pl.col("_prev_ord") - 1 > GAP_OPEN_DAYS)
    )
    if gaps.height == 0:
        return
    example = _examples(
        gaps.with_columns(
            (pl.col("_ord") - pl.col("_prev_ord") - 1).alias("_miss")
        ),
        ("instrument", "_prev_date", "date", "_miss"),
    )
    _add(
        issues,
        "warning",
        "date_gap",
        f"{gaps.height} 处相邻行情间隔超过 {GAP_OPEN_DAYS} 个开市日，"
        f"需人工确认是否长期停牌；示例（证券 @ 上一行 @ 当前行 @ 缺失开市日数）：{example}",
        gaps.height,
    )


def _check_prices(issues: list[ValidationIssue], win: pl.DataFrame) -> None:
    """价格异常：非正、最高最低倒挂、开收越界、vwap 越界。"""
    if win.height == 0:
        return

    price_cols = ["open", "high", "low", "close", "vwap"]
    nonpositive = win.filter(
        pl.any_horizontal([pl.col(col) <= 0 for col in price_cols])
    )
    if nonpositive.height:
        _add(
            issues,
            "error",
            "price_nonpositive",
            f"{nonpositive.height} 行存在 <= 0 的价格 / vwap：{_examples(nonpositive)}",
            nonpositive.height,
        )

    inverted = win.filter(pl.col("high") < pl.col("low"))
    if inverted.height:
        _add(
            issues,
            "error",
            "high_low_inverted",
            f"{inverted.height} 行 high < low：{_examples(inverted)}",
            inverted.height,
        )

    out_of_range = win.filter(
        (pl.col("open") < pl.col("low") - PRICE_EPS)
        | (pl.col("open") > pl.col("high") + PRICE_EPS)
        | (pl.col("close") < pl.col("low") - PRICE_EPS)
        | (pl.col("close") > pl.col("high") + PRICE_EPS)
    )
    if out_of_range.height:
        _add(
            issues,
            "error",
            "open_close_out_of_range",
            f"{out_of_range.height} 行 open / close 不在 [low, high] 内："
            f"{_examples(out_of_range)}",
            out_of_range.height,
        )

    bad_vwap = win.filter(
        (pl.col("vwap") < pl.col("low") * (1 - VWAP_TOLERANCE))
        | (pl.col("vwap") > pl.col("high") * (1 + VWAP_TOLERANCE))
    )
    if bad_vwap.height:
        _add(
            issues,
            "warning",
            "vwap_out_of_range",
            f"{bad_vwap.height} 行 vwap 超出 [low, high] 的 "
            f"{VWAP_TOLERANCE:.0%} 容差（疑似成交量单位错位）：{_examples(bad_vwap)}",
            bad_vwap.height,
        )


def _check_volume_amount(issues: list[ValidationIssue], win: pl.DataFrame) -> None:
    """成交量 / 成交额非负与矛盾组合。"""
    if win.height == 0:
        return
    negative = win.filter((pl.col("volume") < 0) | (pl.col("amount") < 0))
    if negative.height:
        _add(
            issues,
            "error",
            "negative_value",
            f"{negative.height} 行 volume / amount 为负：{_examples(negative)}",
            negative.height,
        )

    mismatch = win.filter(
        ((pl.col("volume") > 0) & (pl.col("amount") == 0))
        | ((pl.col("volume") == 0) & (pl.col("amount") > 0))
    )
    if mismatch.height:
        _add(
            issues,
            "error",
            "volume_amount_mismatch",
            f"{mismatch.height} 行 volume 与 amount 矛盾（一方为 0 另一方为正）："
            f"{_examples(mismatch)}",
            mismatch.height,
        )


def _check_limit_move(
    issues: list[ValidationIssue],
    win: pl.DataFrame,
    instruments: pl.DataFrame,
) -> None:
    """用后复权收益判断是否越过涨跌停带上限 + 容差。"""
    if win.height == 0:
        return

    frame = win.select("instrument", "date", "close", "adjfactor", "limit_up",
                       "limit_down", "_prev_close", "_prev_adj", "_ord", "_prev_ord")
    if instruments.height:
        frame = frame.join(
            instruments.select(
                "instrument",
                pl.col("board").alias("_info_board"),
                "list_date",
            ),
            on="instrument",
            how="left",
        )
    else:
        frame = frame.with_columns(
            pl.lit(None, dtype=pl.String).alias("_info_board"),
            pl.lit(None, dtype=pl.Date).alias("list_date"),
        )
    code_board = frame.select("instrument").unique().with_columns(
        pl.col("instrument")
        .map_elements(_board_of_safe, return_dtype=pl.String)
        .alias("_code_board")
    )
    frame = frame.join(code_board, on="instrument", how="left")
    board = pl.coalesce(pl.col("_info_board"), pl.col("_code_board"))

    up_ratio = (
        pl.when(pl.col("limit_up").is_not_null() & (pl.col("_prev_close") > 0))
        .then((pl.col("limit_up") / pl.col("_prev_close") - 1.0).clip(lower_bound=0.0))
        .otherwise(None)
    )
    down_ratio = (
        pl.when(pl.col("limit_down").is_not_null() & (pl.col("_prev_close") > 0))
        .then((1.0 - pl.col("limit_down") / pl.col("_prev_close")).clip(lower_bound=0.0))
        .otherwise(None)
    )
    from_limits = pl.max_horizontal(up_ratio, down_ratio)
    generic = (
        pl.when(board == "kcb")
        .then(pl.lit(KCB_LIMIT_RATIO))
        .when(board == "bj")
        .then(pl.lit(BJ_LIMIT_RATIO))
        .when((board == "cyb") & (pl.col("date") >= CYB_REFORM_DATE))
        .then(pl.lit(CYB_LIMIT_RATIO))
        .otherwise(pl.lit(MAIN_LIMIT_RATIO))
    )
    ratio = pl.when(from_limits > 0).then(from_limits).otherwise(generic)
    grace = (
        pl.when(board == "kcb")
        .then(pl.lit(NEW_LISTING_GRACE_DAYS["kcb"]))
        .when(board == "cyb")
        .then(pl.lit(NEW_LISTING_GRACE_DAYS["cyb"]))
        .when(board == "bj")
        .then(pl.lit(NEW_LISTING_GRACE_DAYS["bj"]))
        .otherwise(pl.lit(NEW_LISTING_GRACE_DAYS["main"]))
    )
    in_grace = pl.col("list_date").is_not_null() & (
        (pl.col("date") - pl.col("list_date")).dt.total_days() <= grace
    )
    adjusted_return = (
        (pl.col("close") * pl.col("adjfactor"))
        / (pl.col("_prev_close") * pl.col("_prev_adj"))
        - 1.0
    )
    checked = (
        frame.with_columns(adjusted_return.alias("_adj_ret"), ratio.alias("_ratio"))
        .filter(
            (pl.col("_prev_close") > 0)
            & (pl.col("_prev_adj") > 0)
            & (pl.col("adjfactor") > 0)
            & (pl.col("close") > 0)
        )
        # adjfactor 变化日（除权或基准跳变）需要除权参考价才能算涨跌停带，
        # 未复权价格在这天不可比；这类行交由复权一致性检查负责。
        .filter(
            ((pl.col("adjfactor") / pl.col("_prev_adj")) - 1.0).abs() <= ADJ_TOLERANCE
        )
        .filter(~in_grace)
    )
    exceeded = checked.filter(
        pl.col("_adj_ret").abs() - pl.col("_ratio") > LIMIT_TOLERANCE
    )
    # 无涨跌幅限制日豁免（issue #59 用 CSMAR 交易所执行数据全量核对：这些日子
    # 按交易所口径均不违规）：
    # 1) 相邻行情间隔超过 GAP_OPEN_DAYS 个开市日后的首个交易日——长期停牌复牌
    #    或退市整理期首日，无涨跌幅限制（同批票同时会出现在 date_gap 里，不另报）。
    # 2) |后复权收益| 超过 UNRESTRICTED_BAND——超过北交所 30% 这一全市场最大带宽，
    #    任何涨跌幅制度下都不可能，必为无限制日（gap 较小的退市整理首日走这条），
    #    降级为 warning 供人工抽查。
    resumed = (pl.col("_ord") - pl.col("_prev_ord") - 1 > GAP_OPEN_DAYS).fill_null(False)
    unrestricted = exceeded.filter(resumed | (pl.col("_adj_ret").abs() > UNRESTRICTED_BAND))
    violations = exceeded.filter(
        ~resumed & (pl.col("_adj_ret").abs() <= UNRESTRICTED_BAND)
    )
    if unrestricted.height:
        _add(
            issues,
            "warning",
            "unrestricted_move",
            f"{unrestricted.height} 行后复权涨跌幅越过自研带宽但属于无涨跌幅限制日"
            f"（复牌 / 退市整理首日或 |收益| > {UNRESTRICTED_BAND:.0%}）："
            f"{_examples(unrestricted)}",
            unrestricted.height,
        )
    if violations.height:
        worst = violations.with_columns(
            (pl.col("_adj_ret") - pl.col("_ratio")).alias("_excess")
        ).sort("_excess", descending=True)
        example = _examples(worst, ("instrument", "date"))
        _add(
            issues,
            "error",
            "limit_move_exceeded",
            f"{violations.height} 行后复权涨跌幅越过涨跌停带 "
            f"{LIMIT_TOLERANCE:.0%} 容差（疑似复权口径或单位错误）：{example}",
            violations.height,
        )


def _check_adjustments(
    issues: list[ValidationIssue],
    win: pl.DataFrame,
    actions: pl.DataFrame,
) -> None:
    """adjfactor 非正 / 缺失 / 跳变 / 递减。"""
    if win.height == 0:
        return

    nonpositive = win.filter(pl.col("adjfactor") <= 0)
    if nonpositive.height:
        _add(
            issues,
            "error",
            "adjfactor_nonpositive",
            f"{nonpositive.height} 行 adjfactor <= 0：{_examples(nonpositive)}",
            nonpositive.height,
        )

    missing = win.filter(pl.col("adjfactor").is_null())
    if missing.height:
        _add(
            issues,
            "warning",
            "adjfactor_null",
            f"{missing.height} 行 adjfactor 缺失：{_examples(missing)}",
            missing.height,
        )

    comparable = win.filter(
        pl.col("adjfactor").is_not_null()
        & (pl.col("adjfactor") > 0)
        & pl.col("_prev_adj").is_not_null()
        & (pl.col("_prev_adj") > 0)
    ).with_columns((pl.col("adjfactor") / pl.col("_prev_adj")).alias("_adj_ratio"))
    jumps = comparable.filter(
        (pl.col("_adj_ratio") - 1.0).abs() > ADJ_TOLERANCE
    )
    if jumps.height == 0:
        return

    if actions.height:
        unexplained = jumps.join(
            actions.select("instrument", "date"), on=["instrument", "date"], how="anti"
        )
    else:
        unexplained = jumps
    if unexplained.height:
        _add(
            issues,
            "warning",
            "adjfactor_jump",
            f"{unexplained.height} 行 adjfactor 跳变无法由公司行为解释"
            f"（疑似混用不同来源因子基准）：{_examples(unexplained)}",
            unexplained.height,
        )

    decrease = jumps.filter(pl.col("_adj_ratio") < 1.0 - ADJ_TOLERANCE)
    if decrease.height:
        _add(
            issues,
            "warning",
            "adjfactor_decrease",
            f"{decrease.height} 行后复权因子递减（后复权因子应非递减）："
            f"{_examples(decrease)}",
            decrease.height,
        )


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def validate(
    data_dir: Path,
    *,
    end: date | None = None,
    lookback_days: int | None = None,
) -> ValidationReport:
    """对 ``data_dir`` 缓存做体检。

    - ``end``：只体检该日（含）之前的数据，缺省取行情最大日期（最近窗口模式取今天）。
    - ``lookback_days``：只体检最近这么多个开市日；为 None 时全量体检（留给首抓后）。
    """
    data_dir = Path(data_dir)
    issues: list[ValidationIssue] = []
    if lookback_days is not None and lookback_days <= 0:
        raise ValueError("lookback_days 必须为正整数")

    if not data_dir.exists():
        _add(issues, "error", "data_dir_missing", f"数据目录不存在：{data_dir}", 0)
        return ValidationReport(issues=issues)

    calendar = _read_table(
        data_dir, cache.CALENDAR_FILE, TRADE_CALENDAR, "trade_calendar", issues
    )
    instruments = _read_table(
        data_dir, cache.INSTRUMENTS_FILE, INSTRUMENT_INFO, "instruments", issues
    )
    actions = _read_table(
        data_dir,
        cache.CORPORATE_ACTIONS_FILE,
        CORPORATE_ACTIONS,
        "corporate_actions",
        issues,
        required=False,
    )
    industry = _read_table(
        data_dir, cache.INDUSTRY_FILE, INDUSTRY, "industry", issues
    )
    _check_industry_coverage(issues, instruments, industry)

    has_calendar = calendar.height > 0
    if lookback_days is None:
        if end is not None:
            bars = _read_bars(data_dir, issues, end=end)
            data_end = end
        else:
            bars = _read_bars(data_dir, issues)
            data_end = bars["date"].max() if bars.height else date.today()
        open_days = _open_days(calendar, data_end) if has_calendar else calendar
        if open_days.height and bars.height:
            window_dates = (
                open_days.filter(pl.col("date") >= bars["date"].min())["date"].to_list()
            )
        else:
            window_dates = []
    else:
        data_end = end or date.today()
        open_days = _open_days(calendar, data_end) if has_calendar else calendar
        window_dates = (
            open_days["date"].to_list()[-lookback_days:] if open_days.height else []
        )
        load_start = (
            min(window_dates) - timedelta(days=LOOKBACK_PAD_DAYS)
            if window_dates
            else data_end
        )
        bars = _read_bars(data_dir, issues, start=load_start, end=data_end)

    annotated = _annotate(bars, open_days if has_calendar else None)
    if window_dates:
        win = annotated.filter(pl.col("date").is_in(window_dates))
    elif not has_calendar:
        win = annotated.filter(pl.col("date") <= data_end)
    else:
        win = annotated.head(0)

    if win.height == 0:
        _add(
            issues,
            "warning",
            "no_rows_in_window",
            "体检窗口内没有任何行情行，检查日历与缓存是否对齐",
            0,
        )
        return ValidationReport(
            issues=issues,
            checked_rows=0,
            checked_instruments=0,
        )

    if has_calendar and window_dates:
        window = pl.DataFrame({"date": window_dates}).cast({"date": pl.Date})
        _check_completeness(issues, window, win, instruments)
        _check_gaps(issues, win)

    _check_prices(issues, win)
    _check_volume_amount(issues, win)
    _check_limit_move(issues, win, instruments)
    _check_adjustments(issues, win, actions)

    return ValidationReport(
        issues=issues,
        checked_rows=win.height,
        checked_instruments=win["instrument"].n_unique(),
    )


__all__ = [
    "ADJ_TOLERANCE",
    "COVERAGE_MIN_RATIO",
    "GAP_OPEN_DAYS",
    "INDUSTRY_COVERAGE_MIN_RATIO",
    "LIMIT_TOLERANCE",
    "VWAP_TOLERANCE",
    "ValidationIssue",
    "ValidationReport",
    "validate",
]
