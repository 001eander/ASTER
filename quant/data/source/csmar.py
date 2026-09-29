"""基于 CSMAR 离线导出包的 ``DataSource`` 实现。

CSMAR 导出包是一组离线 CSV，不触网、不重试，建模与回测的行情主源即此：

- ``TRD_Dalyr*.csv``（分片）：A 股日线，``Markettype`` 只保留 1/4/16/32/64，
  即纯 A 股。价格为未复权，成交量为股，成交额为元。停牌日无行，与
  ``quant.data.schema`` 的停牌约定一致。
- ``TRD_AdjustFactor.csv``：事件级后复权因子，``CumulateBwardFactor`` 为上市
  至今累乘的后复权因子（``后复权价 = 未复权价 × CumulateBwardFactor``）。
- ``TRD_Cale.csv``：各市场交易日历，``State == "O"`` 表示开市。
- ``TRD_Co.csv``：仅覆盖 2021 年后状态变动的约 1600 只票，撑不起全市场证券
  信息，只用于补充与校验 ``instruments.parquet``。

设计取舍
--------
- 日线首次调用时整包加载进内存（约 500MB，实测 631 万行），之后每次调用都在
  内存表上过滤。建模与建库都按全量使用，一次性加载换取后续零 IO。
- 逐日复权因子用事件表 + ``join_asof(strategy="backward")`` 前向填充。
  因子表从 2021-01-06 起，早于日线窗口，因此窗口内每个交易日都能对上事件。
  首个事件晚于 ``HISTORY_START`` 的票补一行基线事件（取首行的
  ``CumulateBwardFactor / BwardFactor`` 反推窗口前因子），完全无事件的票取 1.0。
- CSMAR 代码列必须按字符串读取（``infer_schema_length=0``），否则前导零丢失。

偏离说明：指令给的基线事件日期为 ``HISTORY_START``，若对所有票都插入该行，
首事件早于 ``HISTORY_START`` 的票会在窗口首日命中基线而不是真实事件
（同日取值依赖 tie 的未定义行为）。本实现只对「首事件晚于 ``HISTORY_START``」
的票插入基线行，与「基线只对该类票生效」的约定一致。
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import polars as pl

from quant.data.schema import (
    CORPORATE_ACTIONS,
    DAILY_BARS,
    HISTORY_START,
    INSTRUMENT_INFO,
    TRADE_CALENDAR,
    check_daily_bars,
    check_schema,
    normalize_instrument,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

#: 日线分片文件名模式。
DALYR_GLOB: str = "TRD_Dalyr*.csv"
#: 后复权因子事件表文件名。
ADJUST_FILE: str = "TRD_AdjustFactor.csv"
#: 交易日历文件名。
CALENDAR_FILE: str = "TRD_Cale.csv"
#: 公司信息文件名。
COMPANY_FILE: str = "TRD_Co.csv"

#: 保留的 Markettype：1/32 沪市、4/16 深市、64 北交所，即纯 A 股。
KEPT_MARKETTYPES: tuple[int, ...] = (1, 4, 16, 32, 64)
#: Markettype -> 交易所后缀，作为代码段的权威口径。
MARKETTYPE_EXCHANGE: dict[int, str] = {1: "SH", 32: "SH", 4: "SZ", 16: "SZ", 64: "BJ"}

#: 代码段判定：沪市。
SH_HEADS: tuple[str, ...] = ("600", "601", "603", "605", "688", "689")
#: 代码段判定：深市。
SZ_HEADS: tuple[str, ...] = ("000", "001", "002", "003", "300", "301", "302")
#: 板块判定：科创板。
KCB_HEADS: tuple[str, ...] = ("688", "689")
#: 板块判定：创业板。
CYB_HEADS: tuple[str, ...] = ("300", "301", "302")


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------


def _empty(schema: pl.Schema) -> pl.DataFrame:
    return pl.DataFrame(schema=schema)


def _scan_all_strings(path: Path | str) -> pl.LazyFrame:
    """按全字符串惰性读取 CSV。

    CSMAR 的代码列（``Stkcd`` / ``Symbol``）带前导零，按整数推断会丢零；个别表
    的字符串列混有非数字值（如 ``PROVINCECODE`` 里的 ``"CHE"``）。统一
    ``infer_schema_length=0`` 规避推断，后续显式 cast。
    """
    return pl.scan_csv(path, encoding="utf8", infer_schema_length=0)


def _exchange_expr(digits: pl.Expr) -> pl.Expr:
    """按六位代码段向量化判定交易所，与 ``schema.exchange_of_digits`` 同规则。"""
    head = digits.str.slice(0, 3)
    return (
        pl.when(head.is_in(SH_HEADS))
        .then(pl.lit("SH"))
        .when(head.is_in(SZ_HEADS))
        .then(pl.lit("SZ"))
        .when(head.str.starts_with("4") | head.str.starts_with("8") | (head == "920"))
        .then(pl.lit("BJ"))
        .otherwise(None)
    )


def _board_expr(digits: pl.Expr, exchange: pl.Expr) -> pl.Expr:
    """按代码段与交易所向量化判定板块，与 ``schema.board_of`` 同规则。"""
    head = digits.str.slice(0, 3)
    return (
        pl.when(exchange == "BJ")
        .then(pl.lit("bj"))
        .when(head.is_in(KCB_HEADS))
        .then(pl.lit("kcb"))
        .when(head.is_in(CYB_HEADS))
        .then(pl.lit("cyb"))
        .otherwise(pl.lit("main"))
    )


def _instrument_expr(digits: pl.Expr, exchange: pl.Expr) -> pl.Expr:
    return digits + pl.lit(".") + exchange


def _to_date_expr(text: pl.Expr) -> pl.Expr:
    """把 ``YYYY-MM-DD`` 文本转 Date，空串与非法值置 null。"""
    return (
        text.str.strip_chars()
        .replace("", None)
        .str.to_date("%Y-%m-%d", strict=False)
    )


# ---------------------------------------------------------------------------
# 数据源
# ---------------------------------------------------------------------------


class CsmarSource:
    """CSMAR 离线数据源，满足 ``quant.data.source.base.DataSource`` 协议。

    ``csmar_dir`` 指向解压后的 CSV 目录（``TRD_Dalyr*.csv`` 会被 glob 全部分片）。
    ``base_info`` 给定时作为 ``instrument_info`` 的基础表（akshare 口径的
    ``instruments.parquet``），CSMAR 的 ``TRD_Co`` 只补充与校验；不给时
    ``instrument_info`` 只返回 ``TRD_Co`` 子集，覆盖不了全市场。
    ``factor_scale`` 为可选的 ``{instrument: 系数}``，加载时把 ``adjfactor``
    乘以该系数，供复权基期归一化使用（默认恒等）。
    """

    def __init__(
        self,
        csmar_dir: str | Path,
        *,
        base_info: str | Path | None = None,
        factor_scale: dict[str, float] | None = None,
    ) -> None:
        self._dir = Path(csmar_dir)
        self._base_info = Path(base_info) if base_info is not None else None
        self._factor_scale = dict(factor_scale) if factor_scale else {}
        self._raw: pl.DataFrame | None = None

    # ------------------------------------------------------------------
    # 加载
    # ------------------------------------------------------------------

    def _dalyr_files(self) -> list[Path]:
        files = sorted(self._dir.glob(DALYR_GLOB))
        if not files:
            raise FileNotFoundError(
                f"{self._dir} 下未找到日线分片（模式 {DALYR_GLOB}）"
            )
        return files

    def _ensure_loaded(self) -> pl.DataFrame:
        """首次调用时整包加载日线并算好复权因子，之后直接复用内存表。"""
        if self._raw is not None:
            return self._raw
        raw = self._load_daily()
        raw = self._attach_adjfactor(raw)
        self._raw = raw
        return raw

    def _load_daily(self) -> pl.DataFrame:
        lf = pl.concat([_scan_all_strings(path) for path in self._dalyr_files()])
        frame = (
            lf.select(
                pl.col("Stkcd").str.zfill(6).alias("digits"),
                _to_date_expr(pl.col("Trddt")).alias("date"),
                pl.col("Opnprc").cast(pl.Float64, strict=False).alias("open"),
                pl.col("Hiprc").cast(pl.Float64, strict=False).alias("high"),
                pl.col("Loprc").cast(pl.Float64, strict=False).alias("low"),
                pl.col("Clsprc").cast(pl.Float64, strict=False).alias("close"),
                pl.col("Dnshrtrd").cast(pl.Float64, strict=False).alias("volume"),
                pl.col("Dnvaltrd").cast(pl.Float64, strict=False).alias("amount"),
                pl.col("Markettype").cast(pl.Int64, strict=False).alias("markettype"),
                pl.col("Trdsta").cast(pl.Int64, strict=False).alias("trdsta"),
                pl.col("LimitUp").cast(pl.Float64, strict=False).alias("ref_limit_up"),
                pl.col("LimitDown")
                .cast(pl.Float64, strict=False)
                .alias("ref_limit_down"),
                pl.col("PreClosePrice")
                .cast(pl.Float64, strict=False)
                .alias("ref_pre_close"),
            )
            .filter(pl.col("markettype").is_in(KEPT_MARKETTYPES))
            # 防御性剔除零成交行：schema 约定停牌日无行。
            .filter((pl.col("volume") > 0) & (pl.col("amount") > 0))
            .with_columns(
                pl.col("markettype").replace_strict(
                    MARKETTYPE_EXCHANGE, return_dtype=pl.String
                ).alias("exchange_raw"),
                _exchange_expr(pl.col("digits")).alias("exchange_digits"),
            )
        )
        frame = frame.with_columns(
            _instrument_expr(pl.col("digits"), pl.col("exchange_raw")).alias(
                "instrument"
            ),
            (
                pl.col("exchange_digits").is_not_null()
                & (pl.col("exchange_digits") != pl.col("exchange_raw"))
            ).alias("_exchange_mismatch"),
        ).collect()

        mismatch = frame.filter(pl.col("_exchange_mismatch"))
        if mismatch.height:
            logger.warning(
                "CSMAR 交易所交叉校验：%d 行代码段与 Markettype 不一致，以 Markettype 为准（样例 %s）",
                mismatch.height,
                mismatch.select("digits", "exchange_digits", "exchange_raw")
                .head(5)
                .rows(),
            )
        return frame.drop(
            "_exchange_mismatch", "exchange_digits", "exchange_raw", "digits",
            "markettype",
        )

    def _load_events(self) -> pl.DataFrame:
        """读取后复权因子事件表，字段为 (instrument, event_date, cum, bward)。"""
        path = self._dir / ADJUST_FILE
        df = pl.read_csv(path, encoding="utf8", infer_schema_length=0)
        digits = pl.col("Symbol").str.zfill(6)
        return (
            df.select(
                digits.alias("digits"),
                _to_date_expr(pl.col("TradingDate")).alias("event_date"),
                pl.col("CumulateBwardFactor")
                .cast(pl.Float64, strict=False)
                .alias("cum"),
                pl.col("BwardFactor").cast(pl.Float64, strict=False).alias("bward"),
            )
            .with_columns(_exchange_expr(pl.col("digits")).alias("exchange"))
            .filter(pl.col("exchange").is_not_null() & pl.col("event_date").is_not_null())
            .with_columns(
                _instrument_expr(pl.col("digits"), pl.col("exchange")).alias(
                    "instrument"
                )
            )
            .select(["instrument", "event_date", "cum", "bward"])
        )

    def _attach_adjfactor(self, bars: pl.DataFrame) -> pl.DataFrame:
        """按事件表前向填充逐日 ``adjfactor``，并应用 ``factor_scale``。"""
        events = self._load_events().sort(["instrument", "event_date"])
        # 每票首事件（按事件日期最早）。基线只在首事件晚于 HISTORY_START 时补，
        # 否则窗口内每个交易日都能命中真实事件。
        first_events = events.unique(subset=["instrument"], keep="first")
        baseline = first_events.filter(pl.col("event_date") > HISTORY_START).select(
            pl.col("instrument"),
            pl.lit(HISTORY_START, dtype=pl.Date).alias("event_date"),
            (pl.col("cum") / pl.col("bward")).alias("cum"),
        )
        padded = (
            pl.concat(
                [events.select(["instrument", "event_date", "cum"]), baseline],
                how="vertical",
            )
            .sort(["instrument", "event_date"])
        )

        out = (
            bars.sort(["instrument", "date"])
            .join_asof(
                padded,
                left_on="date",
                right_on="event_date",
                by="instrument",
                strategy="backward",
                check_sortedness=False,
            )
            .with_columns(pl.col("cum").fill_null(1.0).alias("adjfactor"))
            .drop("cum", "event_date")
        )
        if self._factor_scale:
            scale = pl.col("instrument").replace_strict(
                self._factor_scale, default=1.0, return_dtype=pl.Float64
            )
            out = out.with_columns((pl.col("adjfactor") * scale).alias("adjfactor"))
        return out.with_columns(
            pl.when(pl.col("volume") > 0)
            .then(pl.col("amount") / pl.col("volume"))
            .otherwise(None)
            .cast(pl.Float64)
            .alias("vwap"),
            pl.lit(None, dtype=pl.Float64).alias("limit_up"),
            pl.lit(None, dtype=pl.Float64).alias("limit_down"),
        ).sort(["instrument", "date"])

    # ------------------------------------------------------------------
    # 日线
    # ------------------------------------------------------------------

    def daily_bars(
        self, instruments: list[str], start: date, end: date
    ) -> pl.DataFrame:
        if not instruments:
            return _empty(DAILY_BARS)
        wanted = [normalize_instrument(item) for item in instruments]
        raw = self._ensure_loaded()
        out = (
            raw.filter(
                pl.col("instrument").is_in(wanted)
                & (pl.col("date") >= start)
                & (pl.col("date") <= end)
            )
            .select(list(DAILY_BARS.keys()))
            .cast(DAILY_BARS)
            .sort(["instrument", "date"])
        )
        check_daily_bars(out)
        return out

    def max_date(self) -> date | None:
        """首次加载后日线的最大日期，无数据时为 ``None``。"""
        raw = self._ensure_loaded()
        if raw.height == 0:
            return None
        return raw["date"].max()

    def limit_reference(self) -> pl.DataFrame:
        """涨跌停预计算参考表（issue #5），不落 ``DAILY_BARS``。

        列：``date``、``instrument``、``trdsta``（Int64）、``ref_limit_up``、
        ``ref_limit_down``、``ref_pre_close``，直接来自 ``TRD_Dalyr`` 的
        ``Trdsta`` / ``LimitUp`` / ``LimitDown`` / ``PreClosePrice``。
        """
        raw = self._ensure_loaded()
        return raw.select(
            "date", "instrument", "trdsta", "ref_limit_up", "ref_limit_down",
            "ref_pre_close",
        )

    # ------------------------------------------------------------------
    # 交易日历
    # ------------------------------------------------------------------

    def trade_calendar(self, start: date, end: date) -> pl.DataFrame:
        path = self._dir / CALENDAR_FILE
        df = pl.read_csv(path, encoding="utf8", infer_schema_length=0)
        if df.height == 0:
            return _empty(TRADE_CALENDAR)
        df = df.select(
            _to_date_expr(pl.col("Clddt")).alias("date"),
            pl.col("Markettype").cast(pl.Int64, strict=False).alias("markettype"),
            pl.col("State").cast(pl.Utf8, strict=False).alias("state"),
        ).filter(pl.col("markettype").is_in(KEPT_MARKETTYPES))
        if df.height == 0:
            return _empty(TRADE_CALENDAR)
        out = (
            df.group_by("date")
            .agg((pl.col("state") == "O").any().alias("is_open"))
            .filter(pl.col("date") >= HISTORY_START)
            .filter((pl.col("date") >= start) & (pl.col("date") <= end))
            .sort("date")
            .cast(TRADE_CALENDAR)
        )
        check_schema(out, TRADE_CALENDAR, name="trade_calendar")
        return out

    # ------------------------------------------------------------------
    # 公司行为
    # ------------------------------------------------------------------

    def corporate_actions(
        self, instruments: list[str], start: date, end: date
    ) -> pl.DataFrame:
        """CSMAR 导出包无分红表，返回空表；公司行为仍由 akshare 路径负责。"""
        return _empty(CORPORATE_ACTIONS)

    # ------------------------------------------------------------------
    # 证券信息
    # ------------------------------------------------------------------

    def _load_company(self) -> pl.DataFrame:
        """读取 ``TRD_Co`` 并归一化成 ``INSTRUMENT_INFO`` 子集。"""
        path = self._dir / COMPANY_FILE
        df = pl.read_csv(path, encoding="utf8", infer_schema_length=0)
        digits = pl.col("Stkcd").str.zfill(6)
        df = (
            df.select(
                digits.alias("digits"),
                pl.col("Stknme").cast(pl.Utf8, strict=False).alias("name"),
                _to_date_expr(pl.col("Listdt")).alias("list_date"),
                pl.col("Statco").cast(pl.Utf8, strict=False).alias("statco"),
                _to_date_expr(pl.col("Statdt")).alias("statdt"),
            )
            .with_columns(_exchange_expr(pl.col("digits")).alias("exchange"))
            .filter(pl.col("exchange").is_not_null())
            .with_columns(
                _instrument_expr(pl.col("digits"), pl.col("exchange")).alias(
                    "instrument"
                ),
                _board_expr(pl.col("digits"), pl.col("exchange")).alias("board"),
            )
        )
        out = df.select(
            "instrument",
            "name",
            "board",
            "list_date",
            pl.when(
                (pl.col("statco") == "D") & pl.col("statdt").is_not_null()
            )
            .then(pl.col("statdt"))
            .otherwise(None)
            .cast(pl.Date)
            .alias("delist_date"),
        )
        return out.unique(subset=["instrument"], keep="first")

    def instrument_info(self) -> pl.DataFrame:
        """证券基本信息：以 ``base_info`` 基础表为准，``TRD_Co`` 补充与校验。

        ``base_info`` 未给定时只返回 ``TRD_Co`` 子集，仅约 1600 只 2021 年后
        状态变动的票，覆盖不了全市场。
        """
        csmar = self._load_company()
        if self._base_info is None:
            out = csmar.sort("instrument").cast(INSTRUMENT_INFO)
            check_schema(out, INSTRUMENT_INFO, name="instrument_info")
            return out

        base = (
            pl.read_parquet(self._base_info)
            .select(list(INSTRUMENT_INFO.keys()))
            .cast(INSTRUMENT_INFO)
        )
        joined = base.join(
            csmar.select(
                "instrument",
                pl.col("list_date").alias("_cs_list"),
                pl.col("delist_date").alias("_cs_delist"),
            ),
            on="instrument",
            how="inner",
        )
        list_conflict = joined.filter(
            pl.col("_cs_list").is_not_null()
            & (pl.col("_cs_list") != pl.col("list_date"))
        )
        delist_conflict = joined.filter(
            pl.col("_cs_delist").is_not_null()
            & (pl.col("_cs_delist") != pl.col("delist_date"))
        )
        if list_conflict.height or delist_conflict.height:
            logger.warning(
                "TRD_Co 与 %s 冲突：list_date %d 处、delist_date %d 处，保留基础表（样例 %s）",
                self._base_info,
                list_conflict.height,
                delist_conflict.height,
                list_conflict.select("instrument", "list_date", "_cs_list")
                .head(5)
                .rows(),
            )

        new_rows = csmar.join(base.select("instrument"), on="instrument", how="anti")
        out = (
            pl.concat(
                [
                    base.select(list(INSTRUMENT_INFO.keys())),
                    new_rows.select(list(INSTRUMENT_INFO.keys())),
                ],
                how="vertical",
            )
            .sort("instrument")
            .cast(INSTRUMENT_INFO)
        )
        check_schema(out, INSTRUMENT_INFO, name="instrument_info")
        return out


__all__ = [
    "HISTORY_START",
    "KEPT_MARKETTYPES",
    "CsmarSource",
]
