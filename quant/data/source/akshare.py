"""基于 akshare 的 ``DataSource`` 实现（日线 / 日历 / 公司行为 / 证券信息）。

设计
----
- 行情优先取东方财富 ``stock_zh_a_hist``（覆盖沪深北），失败时按交易所回退：
  沪市 / 深市用新浪 ``stock_zh_a_daily``，北交所用腾讯 ``stock_zh_a_hist_tx``。
  回退只为在网络或接口波动时保持可用，字段口径统一在本模块内归一化。
- 全部网络访问都经由 akshare，返回的 ``pandas.DataFrame`` 立刻转成 polars（环境无
  pyarrow，用 :func:`_from_pandas` 按列构造），后续处理不碰 pandas。
- 每个 akshare 调用都带有限重试与调用间隔，避免全市场抓取被限流。

单位实测结论（2026-09-28，样本 600519 / 300750 / 920001，与公开行情逐日核对）
--------------------------------------------------------------------------
- 东方财富 ``stock_zh_a_hist``：``成交量`` 单位为「手」，本模块 ×100 转「股」；
  ``成交额`` 单位为「元」；价格为未复权。后复权因子取 ``adjust="hfq"`` 的收盘价
  除以未复权收盘价。*本条依据 akshare 文档与东财字段口径；实测当天
  ``push2his.eastmoney.com`` 被所在网络阻断（TCP 连接被 RST），未能直连复验，
  若你所在网络可达请用 ``scripts/smoke_akshare.py`` 复跑确认。*
- 新浪 ``stock_zh_a_daily``：``volume`` 已是「股」（600519 某日 2,533,166 股，
  与 ``amount / close`` 吻合），``amount`` 为「元」；后复权因子由
  ``adjust="hfq-factor"`` 直接给出（``后复权价 = 未复权价 × hfq_factor``）。
- 腾讯 ``stock_zh_a_hist_tx``：函数据说明已将 ``volume`` 折算为「股」、``amount``
  为「元」（920001 某日 1,246,400 股 / 14,346,700 元，与价格吻合）。
- 三种来源都用「``vwap = amount / volume`` 应落在当日 [low, high] 之间」做内部
  一致性校验，可捕捉成交量单位错位 100 倍的情况。

接口坑
------
- akshare 1.18.97 的 ``stock_zh_a_hist_tx`` 在 ``get_tx_start_year`` 回退分支取
  ``["day"]``，而北交所 / 前复权请求实际返回 ``qfqday``，会抛 ``KeyError``。
  本模块在调用腾讯接口前用一个上下文管理器临时替换该函数绕过。
- 新浪 ``stock_zh_a_daily`` 对北交所股票会在股本接口拿到 ``(null)`` 而报
  ``JSONDecodeError``，因此北交所不走新浪。
- ``stock_fhps_detail_em``（东财分红）对北交所返回 ``TypeError``，故北交所公司
  行为走新浪 ``stock_history_dividend_detail``。

已知风险
--------
- 退市股清单来自交易所接口（``stock_info_sh_delist`` / ``stock_info_sz_delist``），
  历史覆盖不全；北交所退市股无独立接口，缺口更大。能取多少算多少。
- 北交所证券列表接口（``stock_info_bj_name_code``）只含 920 段现役代码，老代码
  （430/830/870 段）已迁移到 920 段，查旧代码取不到数据（诺思兰德 430047.BJ ->
  920047.BJ）。
- 腾讯对个别北交所老股只出到 2020 年，数据陈旧，`daily_bars` 会返回空表而不报错。
"""
from __future__ import annotations

import contextlib
import importlib
import logging
import re
import time
from collections.abc import Callable, Iterator
from datetime import date
from typing import TypeVar

import akshare as ak
import polars as pl

from quant.data.schema import (
    CORPORATE_ACTIONS,
    DAILY_BARS,
    INSTRUMENT_INFO,
    TRADE_CALENDAR,
    board_of,
    check_daily_bars,
    check_schema,
    normalize_instrument,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

#: 每次 akshare 调用前的基础间隔（秒），降低全市场抓取被限流的概率。
REQUEST_INTERVAL_SECONDS: float = 0.3
#: 单次调用的最大尝试次数。
RETRY_TIMES: int = 3
#: 重试的指数退避基数（秒），第 n 次重试等待 RETRY_BASE_DELAY_SECONDS * 2 ** (n - 1)。
RETRY_BASE_DELAY_SECONDS: float = 1.0
#: 东方财富成交量单位「手」换算为「股」的系数。
EM_VOLUME_LOT_SIZE: float = 100.0
#: akshare 日期参数格式。
DATE_FORMAT: str = "%Y%m%d"
#: 腾讯接口起始年份的兜底值，用于绕过 akshare 的 ``qfqday`` bug。
TX_START_YEAR_PLACEHOLDER: str = "1990-01-01"

#: 「派X元」形式的现金分红文本（前导「10」可能与其后的送转共用，故不强制匹配）。
_PER10_CASH_RE = r"派\s*([0-9]+(?:\.[0-9]+)?)"
#: 「送X股」「转X股」「转增X股」形式的送转文本。
_PER10_SHARE_RE = r"(?:送|转增|转)\s*([0-9]+(?:\.[0-9]+)?)"

T = TypeVar("T")


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------


def _call(fn: Callable[..., T], *args: object, **kwargs: object) -> T:
    """带指数退避重试的 akshare 调用。"""
    last_exc: Exception | None = None
    for attempt in range(RETRY_TIMES):
        if attempt:
            time.sleep(RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1)))
        time.sleep(REQUEST_INTERVAL_SECONDS)
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - akshare 抛出的异常类型不稳定
            last_exc = exc
            logger.warning(
                "akshare %s 调用失败（%d/%d）：%s",
                getattr(fn, "__name__", fn),
                attempt + 1,
                RETRY_TIMES,
                exc,
            )
    raise RuntimeError(f"akshare 调用重试 {RETRY_TIMES} 次仍失败") from last_exc


def _fmt(day: date) -> str:
    return day.strftime(DATE_FORMAT)


def _empty(schema: pl.Schema) -> pl.DataFrame:
    return pl.DataFrame(schema=schema)


def _from_pandas(df: object) -> pl.DataFrame:
    """把 akshare 返回的 ``pandas.DataFrame`` 转成 polars。

    环境未安装 pyarrow（akshare 不依赖它），``pl.from_pandas`` 对 object 列会失败，
    因此按列取 Python 对象再构造。NaN / NaT 统一成 None，避免 mixed 列表让 polars
    推断类型时报错（akshare 的日期列常混有 NaT）。
    """
    columns = getattr(df, "columns")

    def _cell(value: object) -> object:
        if value is None:
            return None
        try:
            if value != value:  # NaN 与 pandas.NaT 都满足 x != x
                return None
        except Exception:  # noqa: BLE001 - 少数对象不支持比较
            pass
        return value

    return pl.DataFrame(
        {str(col): [_cell(value) for value in df[col].tolist()] for col in columns}  # type: ignore[index]
    )


def _price_frame(
    df: pl.DataFrame,
    *,
    date_col: str,
    open_col: str,
    high_col: str,
    low_col: str,
    close_col: str,
    volume_col: str,
    amount_col: str,
    volume_scale: float = 1.0,
) -> pl.DataFrame:
    """把不同数据源的原始价格列统一成内部列名，成交量换算为「股」。"""
    return df.select(
        pl.col(date_col).cast(pl.Date).alias("date"),
        pl.col(open_col).cast(pl.Float64, strict=False).alias("open"),
        pl.col(high_col).cast(pl.Float64, strict=False).alias("high"),
        pl.col(low_col).cast(pl.Float64, strict=False).alias("low"),
        pl.col(close_col).cast(pl.Float64, strict=False).alias("close"),
        (pl.col(volume_col).cast(pl.Float64, strict=False) * volume_scale).alias("volume"),
        pl.col(amount_col).cast(pl.Float64, strict=False).alias("amount"),
    )


def _with_adjfactor_by_ratio(df: pl.DataFrame, hfq: pl.DataFrame) -> pl.DataFrame:
    """adjfactor = 后复权收盘 / 未复权收盘，按日期对齐。"""
    hfq_col = hfq.columns[1]
    factor = hfq.select(
        pl.col(hfq.columns[0]).cast(pl.Date).alias("date"),
        pl.col(hfq_col).cast(pl.Float64, strict=False).alias("_hfq_close"),
    )
    return (
        df.join(factor, on="date", how="left")
        .with_columns(
            pl.when(pl.col("close") > 0)
            .then(pl.col("_hfq_close") / pl.col("close"))
            .otherwise(None)
            .cast(pl.Float64)
            .alias("adjfactor")
        )
        .drop("_hfq_close")
        .drop_nulls("adjfactor")
    )


@contextlib.contextmanager
def _tx_start_year_workaround() -> Iterator[None]:
    """绕过 akshare 1.18.97 ``get_tx_start_year`` 对北交所的 ``qfqday`` bug。

    ``stock_hist_tx`` 用 ``from ... import get_tx_start_year`` 绑定函数，
    因此替换该模块命名空间里的同名属性即可，退出时还原。
    """
    module = importlib.import_module("akshare.stock_feature.stock_hist_tx")
    original = module.get_tx_start_year  # type: ignore[attr-defined]
    module.get_tx_start_year = lambda symbol: TX_START_YEAR_PLACEHOLDER  # type: ignore[attr-defined]
    try:
        yield
    finally:
        module.get_tx_start_year = original  # type: ignore[attr-defined]


def parse_dividend_text(text: str) -> tuple[float | None, float | None]:
    """解析「10转1.00派6.00元」这类分红说明，返回（每股派息元, 每股送转股）。

    文本里的数字是「每 10 股」，这里除以 10 得到每股口径。送股与转增可能同时出现
    （如「10送2转3」），按每 10 股累加；缺失的那一项返回 None。
    """
    cash_match = re.search(_PER10_CASH_RE, text)
    share_matches = re.findall(_PER10_SHARE_RE, text)
    cash = float(cash_match.group(1)) / 10.0 if cash_match else None
    share = sum(float(value) for value in share_matches) / 10.0 if share_matches else None
    return cash, share


def _normalize_silent(code: str) -> str | None:
    try:
        return normalize_instrument(code)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# 数据源
# ---------------------------------------------------------------------------


class AkshareSource:
    """akshare 数据源，满足 ``quant.data.source.base.DataSource`` 协议。"""

    # ------------------------------------------------------------------
    # 日线
    # ------------------------------------------------------------------

    def daily_bars(
        self, instruments: list[str], start: date, end: date
    ) -> pl.DataFrame:
        frames: list[pl.DataFrame] = []
        for raw_instrument in instruments:
            instrument = normalize_instrument(raw_instrument)
            try:
                frame = self._daily_bars_one(instrument, start, end)
            except Exception as exc:  # noqa: BLE001 - 单票失败不影响整批
                logger.warning("抓取 %s 日线失败：%s", instrument, exc)
                continue
            if frame is not None and frame.height:
                frames.append(frame)
        if not frames:
            return _empty(DAILY_BARS)
        out = pl.concat(frames).sort(["instrument", "date"]).cast(DAILY_BARS)
        check_daily_bars(out)
        return out

    def _daily_bars_one(
        self, instrument: str, start: date, end: date
    ) -> pl.DataFrame | None:
        digits, exchange = instrument.split(".")
        fetchers: list[Callable[[str, str, date, date], pl.DataFrame | None]] = [
            self._fetch_daily_em
        ]
        if exchange in ("SH", "SZ"):
            fetchers.append(self._fetch_daily_sina)
        else:
            fetchers.append(self._fetch_daily_tx)
        for fetch in fetchers:
            try:
                frame = fetch(instrument, digits, start, end)
            except Exception as exc:  # noqa: BLE001
                logger.warning("%s 抓取 %s 失败：%s", fetch.__name__, instrument, exc)
                continue
            if frame is not None and frame.height:
                return frame
        return None

    def _fetch_daily_em(
        self, instrument: str, digits: str, start: date, end: date
    ) -> pl.DataFrame | None:
        raw = _call(
            ak.stock_zh_a_hist,
            symbol=digits,
            period="daily",
            start_date=_fmt(start),
            end_date=_fmt(end),
            adjust="",
        )
        if raw is None or raw.empty:
            return None
        hfq = _call(
            ak.stock_zh_a_hist,
            symbol=digits,
            period="daily",
            start_date=_fmt(start),
            end_date=_fmt(end),
            adjust="hfq",
        )
        df = _price_frame(
            _from_pandas(raw),
            date_col="日期",
            open_col="开盘",
            high_col="最高",
            low_col="最低",
            close_col="收盘",
            volume_col="成交量",
            amount_col="成交额",
            volume_scale=EM_VOLUME_LOT_SIZE,
        )
        df = _with_adjfactor_by_ratio(
            df, _from_pandas(hfq).select(pl.col("日期"), pl.col("收盘"))
        )
        return self._finalize_bars(instrument, df)

    def _fetch_daily_sina(
        self, instrument: str, digits: str, start: date, end: date
    ) -> pl.DataFrame | None:
        symbol = f"{instrument.split('.')[1].lower()}{digits}"
        raw = _call(
            ak.stock_zh_a_daily,
            symbol=symbol,
            start_date=_fmt(start),
            end_date=_fmt(end),
            adjust="",
        )
        if raw is None or raw.empty:
            return None
        factor = _call(ak.stock_zh_a_daily, symbol=symbol, adjust="hfq-factor")
        df = _price_frame(
            _from_pandas(raw),
            date_col="date",
            open_col="open",
            high_col="high",
            low_col="low",
            close_col="close",
            volume_col="volume",
            amount_col="amount",
        )
        factor_df = _from_pandas(factor).select(
            pl.col("date").cast(pl.Date).alias("_event_date"),
            pl.col("hfq_factor").cast(pl.Float64, strict=False).alias("adjfactor"),
        )
        df = (
            df.sort("date")
            .join_asof(
                factor_df.sort("_event_date"),
                left_on="date",
                right_on="_event_date",
                strategy="backward",
            )
            .drop("_event_date")
            .drop_nulls("adjfactor")
        )
        return self._finalize_bars(instrument, df)

    def _fetch_daily_tx(
        self, instrument: str, digits: str, start: date, end: date
    ) -> pl.DataFrame | None:
        symbol = f"{instrument.split('.')[1].lower()}{digits}"
        with _tx_start_year_workaround():
            raw = _call(
                ak.stock_zh_a_hist_tx,
                symbol=symbol,
                start_date=_fmt(start),
                end_date=_fmt(end),
                adjust="",
            )
            if raw is None or raw.empty:
                return None
            hfq = _call(
                ak.stock_zh_a_hist_tx,
                symbol=symbol,
                start_date=_fmt(start),
                end_date=_fmt(end),
                adjust="hfq",
            )
        df = _price_frame(
            _from_pandas(raw),
            date_col="date",
            open_col="open",
            high_col="high",
            low_col="low",
            close_col="close",
            volume_col="volume",
            amount_col="amount",
        )
        df = _with_adjfactor_by_ratio(
            df, _from_pandas(hfq).select(pl.col("date"), pl.col("close"))
        )
        return self._finalize_bars(instrument, df)

    def _finalize_bars(self, instrument: str, df: pl.DataFrame) -> pl.DataFrame:
        out = (
            df.with_columns(
                pl.lit(instrument).alias("instrument"),
                pl.when(pl.col("volume") > 0)
                .then(pl.col("amount") / pl.col("volume"))
                .otherwise(None)
                .cast(pl.Float64)
                .alias("vwap"),
                pl.lit(None, dtype=pl.Float64).alias("limit_up"),
                pl.lit(None, dtype=pl.Float64).alias("limit_down"),
            )
            .select(list(DAILY_BARS.keys()))
            .cast(DAILY_BARS)
        )
        return out

    # ------------------------------------------------------------------
    # 交易日历
    # ------------------------------------------------------------------

    def trade_calendar(self, start: date, end: date) -> pl.DataFrame:
        raw = _call(ak.tool_trade_date_hist_sina)
        open_dates = _from_pandas(raw).select(
            pl.col("trade_date").cast(pl.Date).alias("date")
        )
        calendar = pl.DataFrame(
            {"date": pl.date_range(start, end, interval="1d", eager=True)}
        )
        calendar = (
            calendar.join(
                open_dates.with_columns(pl.lit(True).alias("is_open")),
                on="date",
                how="left",
            )
            .with_columns(pl.col("is_open").fill_null(False))
            .cast(TRADE_CALENDAR)
        )
        check_schema(calendar, TRADE_CALENDAR, name="trade_calendar")
        return calendar

    # ------------------------------------------------------------------
    # 公司行为（分红送转）
    # ------------------------------------------------------------------

    def corporate_actions(
        self, instruments: list[str], start: date, end: date
    ) -> pl.DataFrame:
        frames: list[pl.DataFrame] = []
        for raw_instrument in instruments:
            instrument = normalize_instrument(raw_instrument)
            digits = instrument.split(".")[0]
            try:
                frame = self._corporate_actions_one(instrument, digits)
            except Exception as exc:  # noqa: BLE001
                logger.warning("抓取 %s 公司行为失败：%s", instrument, exc)
                continue
            if frame is not None and frame.height:
                frames.append(frame)
        if not frames:
            return _empty(CORPORATE_ACTIONS)
        out = (
            pl.concat(frames)
            .filter((pl.col("date") >= start) & (pl.col("date") <= end))
            .group_by(["date", "instrument"])
            .agg(
                pl.col("cash_per_share").sum(),
                pl.col("share_per_share").sum(),
            )
            .sort(["instrument", "date"])
            .cast(CORPORATE_ACTIONS)
        )
        check_schema(out, CORPORATE_ACTIONS, name="corporate_actions")
        return out

    def _corporate_actions_one(
        self, instrument: str, digits: str
    ) -> pl.DataFrame | None:
        try:
            return self._corporate_actions_sina(instrument, digits)
        except Exception as exc:  # noqa: BLE001 - 回退到东财接口
            logger.warning("新浪分红接口抓取 %s 失败：%s，改用东财接口", instrument, exc)
            return self._corporate_actions_em(instrument, digits)

    def _corporate_actions_sina(
        self, instrument: str, digits: str
    ) -> pl.DataFrame | None:
        raw = _call(ak.stock_history_dividend_detail, symbol=digits, indicator="分红")
        if raw is None or raw.empty or "除权除息日" not in raw.columns:
            return None
        df = _from_pandas(raw)
        df = df.filter(
            (pl.col("进度").cast(pl.Utf8, strict=False) == "实施")
            & pl.col("除权除息日").is_not_null()
        )
        if df.height == 0:
            return None
        return df.select(
            pl.col("除权除息日").cast(pl.Date).alias("date"),
            pl.lit(instrument).alias("instrument"),
            (
                pl.col("派息").cast(pl.Float64, strict=False).fill_null(0.0) / 10.0
            ).alias("cash_per_share"),
            (
                (
                    pl.col("送股").cast(pl.Float64, strict=False).fill_null(0.0)
                    + pl.col("转增").cast(pl.Float64, strict=False).fill_null(0.0)
                )
                / 10.0
            ).alias("share_per_share"),
        )

    def _corporate_actions_em(
        self, instrument: str, digits: str
    ) -> pl.DataFrame | None:
        raw = _call(ak.stock_fhps_detail_em, symbol=digits)
        if raw is None or raw.empty or "除权除息日" not in raw.columns:
            return None
        df = _from_pandas(raw)
        df = df.filter(
            pl.col("方案进度").cast(pl.Utf8, strict=False).str.contains("实施")
            & pl.col("除权除息日").is_not_null()
        )
        if df.height == 0:
            return None
        cash_numeric = pl.col("现金分红-现金分红比例").cast(
            pl.Float64, strict=False
        )
        share_numeric = pl.col("送转股份-送股比例").cast(
            pl.Float64, strict=False
        ).fill_null(0.0) + pl.col("送转股份-转股比例").cast(
            pl.Float64, strict=False
        ).fill_null(0.0)
        # 文本列在个别 akshare 版本里会缺失或改名，缺失时退化为 null，以数值列为准。
        description = "现金分红-现金分红比例描述"
        if description in df.columns:
            text = pl.col(description).cast(pl.Utf8, strict=False)
            cash_from_text = (
                text.str.extract(_PER10_CASH_RE, group_index=1).cast(
                    pl.Float64, strict=False
                )
            )
            share_from_text = (
                text.str.extract(_PER10_SHARE_RE, group_index=1).cast(
                    pl.Float64, strict=False
                )
            )
        else:
            cash_from_text = pl.lit(None, dtype=pl.Float64)
            share_from_text = pl.lit(None, dtype=pl.Float64)
        return df.select(
            pl.col("除权除息日").cast(pl.Date).alias("date"),
            pl.lit(instrument).alias("instrument"),
            (cash_numeric.fill_null(cash_from_text).fill_null(0.0) / 10.0).alias(
                "cash_per_share"
            ),
            (share_numeric.fill_null(share_from_text).fill_null(0.0) / 10.0).alias(
                "share_per_share"
            ),
        )

    # ------------------------------------------------------------------
    # 证券信息
    # ------------------------------------------------------------------

    def instrument_info(self) -> pl.DataFrame:
        frames: list[pl.DataFrame] = [
            self._info_sh_active(),
            self._info_sz_active(),
            self._info_bj(),
            self._info_sh_delist(),
            self._info_sz_delist(),
        ]
        out = (
            pl.concat(frames)
            .unique(subset=["instrument"], keep="first")
            .sort("instrument")
            .cast(INSTRUMENT_INFO)
        )
        check_schema(out, INSTRUMENT_INFO, name="instrument_info")
        return out

    def _info_frame(
        self,
        raw: object,
        *,
        code_col: str,
        name_col: str,
        list_col: str,
        delist_col: str | None,
    ) -> pl.DataFrame:
        df = _from_pandas(raw)  # type: ignore[arg-type]
        if df.height == 0:
            return _empty(INSTRUMENT_INFO)
        df = df.select(
            pl.col(code_col).cast(pl.Utf8, strict=False).alias("code"),
            pl.col(name_col).cast(pl.Utf8, strict=False).alias("name"),
            pl.col(list_col)
            .cast(pl.Utf8, strict=False)
            .str.replace_all("/", "-")
            .str.to_date("%Y-%m-%d", strict=False)
            .alias("list_date"),
            (
                pl.col(delist_col)
                .cast(pl.Utf8, strict=False)
                .str.replace_all("/", "-")
                .str.to_date("%Y-%m-%d", strict=False)
                .alias("delist_date")
                if delist_col
                else pl.lit(None, dtype=pl.Date).alias("delist_date")
            ),
        )
        df = df.with_columns(
            pl.col("code")
            .map_elements(_normalize_silent, return_dtype=pl.String)
            .alias("instrument")
        )
        skipped = df.filter(pl.col("instrument").is_null())
        if skipped.height:
            logger.info(
                "instrument_info 跳过 %d 个无法归一化的代码（如 %s）",
                skipped.height,
                skipped["code"].head(5).to_list(),
            )
        df = df.filter(pl.col("instrument").is_not_null())
        df = df.with_columns(
            pl.col("instrument")
            .map_elements(board_of, return_dtype=pl.String)
            .alias("board")
        )
        return df.select(list(INSTRUMENT_INFO.keys()))

    def _info_sh_active(self) -> pl.DataFrame:
        main = _call(ak.stock_info_sh_name_code, symbol="主板A股")
        kcb = _call(ak.stock_info_sh_name_code, symbol="科创板")
        frames = [
            self._info_frame(
                main,
                code_col="证券代码",
                name_col="证券简称",
                list_col="上市日期",
                delist_col=None,
            ),
            self._info_frame(
                kcb,
                code_col="证券代码",
                name_col="证券简称",
                list_col="上市日期",
                delist_col=None,
            ),
        ]
        return pl.concat(frames)

    def _info_sz_active(self) -> pl.DataFrame:
        raw = _call(ak.stock_info_sz_name_code, symbol="A股列表")
        return self._info_frame(
            raw,
            code_col="A股代码",
            name_col="A股简称",
            list_col="A股上市日期",
            delist_col=None,
        )

    def _info_bj(self) -> pl.DataFrame:
        raw = _call(ak.stock_info_bj_name_code)
        return self._info_frame(
            raw,
            code_col="证券代码",
            name_col="证券简称",
            list_col="上市日期",
            delist_col=None,
        )

    def _info_sh_delist(self) -> pl.DataFrame:
        raw = _call(ak.stock_info_sh_delist)
        # 上交所退市接口只给「暂停上市日期」，作为退市日的近似。
        return self._info_frame(
            raw,
            code_col="公司代码",
            name_col="公司简称",
            list_col="上市日期",
            delist_col="暂停上市日期",
        )

    def _info_sz_delist(self) -> pl.DataFrame:
        raw = _call(ak.stock_info_sz_delist, symbol="终止上市公司")
        return self._info_frame(
            raw,
            code_col="证券代码",
            name_col="证券简称",
            list_col="上市日期",
            delist_col="终止上市日期",
        )


__all__ = ["AkshareSource", "parse_dividend_text"]
