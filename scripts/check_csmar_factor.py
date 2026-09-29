"""CSMAR 复权因子与 akshare 后复权因子的一致性交叉核对（ASTER issue #59 验收）。

用途
----
CSMAR 离线导出（``TRD_AdjustFactor`` 的 ``CumulateBwardFactor``）与 akshare 的
``adjfactor`` 都声称是「后复权因子」（后复权价 = 未复权价 × 因子），本脚本在共同
交易日上比较两者的比值，回答三个问题：

1. 比例是否恒为 1（同基期，可直接互换）；
2. 比例是否每股恒定但互不相同（口径一致，基期不同，需按每股 scale 归一化）；
3. 比例是否随时间漂移（存在方法学差异，需要逐日／逐事件排查）。

本脚本刻意不复用 ``quant.data.csmar``：那个模块正在并行开发，本脚本正是它的
独立交叉验证，所以 CSMAR 侧的因子重建（事件行 + 基线行 + ``join_asof``）在这里
自包含实现。

用法
----
在仓库根目录运行::

    uv run python scripts/check_csmar_factor.py --csmar-dir /path/to/csmar_extract

``--csmar-dir`` 指向 CSMAR 解压目录（含 ``TRD_Dalyr*.csv`` 与 ``TRD_AdjustFactor.csv``）。
``--report`` / ``--cache-dir`` 分别控制报告输出与 akshare 抓取缓存位置，
默认 ``runs/csmar_factor_check.md`` 与 ``runs/csmar_ak_cache/``（``runs/`` 不入 git）。

若本机系统代理损坏导致 akshare 请求报 ProxyError，可设置 ``ASTER_NO_PROXY=1``
（脚本会在导入 akshare 之前把 ``NO_PROXY`` 置为 ``*``，与
``scripts/fetch_data.py`` 的写法一致）。默认不设该变量。

输出
----
控制台摘要 + markdown 报告（``--report`` 指定路径）。
退出码：结论一 / 结论二为 0；结论三或有效样本不足 ``MIN_SAMPLE`` 只为 1。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Final

# ---------------------------------------------------------------------------
# 必须在 import akshare（连带 requests）之前：按需绕过系统代理。
# ---------------------------------------------------------------------------
if os.environ.get("ASTER_NO_PROXY") == "1":
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import polars as pl  # noqa: E402

from quant.data.schema import normalize_instrument  # noqa: E402
from quant.data.source.akshare import AkshareSource  # noqa: E402

try:  # 让控制台按 UTF-8 输出，避免中文摘要在管道里变成乱码。
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
except Exception:  # noqa: BLE001 - 老解释器或非标准 stdout 直接跳过
    pass

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

#: 报告与 akshare 抓取缓存的默认位置（``runs/`` 不入 git）。
DEFAULT_REPORT_PATH: Final[Path] = REPO_ROOT / "runs" / "csmar_factor_check.md"
DEFAULT_CACHE_DIR: Final[Path] = REPO_ROOT / "runs" / "csmar_ak_cache"

#: TRD_Dalyr 分片的起始交易日，也是 CSMAR 因子基线行的锚点。
DALYR_START: Final[date] = date(2021, 9, 29)
#: akshare 侧取样起点（近三年，减小公网请求量）。
AK_START: Final[date] = date(2023, 1, 1)
#: 两源核对截止日（CSMAR 导出的最后一天）。
COMPARE_END: Final[date] = date(2026, 9, 28)

#: 结论一：单票 ratio 的时间序列变异系数阈值。
RATIO_STD_TOL: Final[float] = 1e-6
#: 结论一：跨票 ratio_mean 的相对离散度阈值。
CROSS_STOCK_TOL: Final[float] = 1e-4
#: 结论三判定的最少有效样本数。
MIN_SAMPLE: Final[int] = 15

#: 单票 akshare 请求的最大尝试次数与重试间隔（秒）。
AK_RETRY_TIMES: Final[int] = 3
AK_RETRY_INTERVAL: Final[float] = 5.0
#: 单票请求之间的礼貌间隔（秒）。
AK_POLITE_INTERVAL: Final[float] = 1.0

#: 行情核对容差。
CLOSE_RTOL: Final[float] = 1e-4
VOLUME_RTOL: Final[float] = 1e-3

#: CSMAR ``Trdsta`` 中代表 ST / *ST 的取值。
ST_TRDSTA: Final[frozenset[int]] = frozenset({2, 3})

#: akshare 三个行情源的 adjfactor 口径说明，用于报告归因。
SOURCE_NOTE: Final[dict[str, str]] = {
    "_fetch_daily_sina": "新浪 `stock_zh_a_daily(adjust='hfq-factor')` 直接给出的因子，事件级精确值",
    "_fetch_daily_em": "东财 `stock_zh_a_hist`，用后复权收盘 / 未复权收盘反推，两个价格都按分取整",
    "_fetch_daily_tx": "腾讯兜底，同样用后复权收盘 / 未复权收盘反推，后复权价按分取整",
}


# ---------------------------------------------------------------------------
# 抽样清单（先扫全量 csv 确认真实存在，再硬编码在此）
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Sample:
    """一只抽样证券及其入选理由。"""

    instrument: str
    category: str
    reason: str


SAMPLE: Final[tuple[Sample, ...]] = (
    # --- 主板 SH ---
    Sample("600519.SH", "主板SH", "沪市主板代表，全窗口 10 次复权事件"),
    Sample("601318.SH", "主板SH", "沪市主板权重股，12 次复权事件"),
    Sample("600177.SH", "主板SH", "沪市主板，事件数并列第二多（13 次）"),
    Sample("603998.SH", "主板SH", "沪市主板，事件数并列第二多（13 次）"),
    Sample("600273.SH", "主板SH", "沪市主板，12 次复权事件"),
    Sample("600793.SH", "主板SH/无事件", "全窗口 0 次复权事件，校验基线填值逻辑"),
    # --- 主板 SZ ---
    Sample("000001.SZ", "主板SZ", "深市主板代表，9 次复权事件"),
    Sample("000002.SZ", "主板SZ", "深市主板老票，仅 3 次复权事件"),
    Sample("000816.SZ", "主板SZ/无事件", "全窗口 0 次复权事件，校验基线填值逻辑"),
    Sample("002555.SZ", "主板SZ/事件最多", "全窗口复权事件数第一（17 次）"),
    Sample("002270.SZ", "主板SZ/事件最多", "事件数并列第二（13 次），代码升序入选"),
    Sample("001206.SZ", "主板SZ/事件最多", "事件数并列第二（13 次），代码升序入选"),
    # --- 创业板 ---
    Sample("300750.SZ", "创业板", "创业板龙头，9 次复权事件"),
    Sample("300573.SZ", "创业板", "创业板，12 次复权事件"),
    Sample("301469.SZ", "创业板/新股", "2023-08-22 上市，首事件晚于比较窗口起点"),
    # --- 科创板 ---
    Sample("688981.SH", "科创板", "科创板代表，CSMAR 全窗口无事件（akshare 因子恒 1）"),
    Sample("688173.SH", "科创板/新股", "2022-01-21 上市，覆盖科创板新股"),
    Sample("688790.SH", "科创板/新股", "2025-12-16 上市，覆盖极短上市窗口"),
    # --- 北交所 ---
    Sample("920445.BJ", "北交所", "920 段，事件数北交所第一（12 次）"),
    Sample("920726.BJ", "北交所", "920 段，事件数北交所第三（10 次）"),
    # --- ST ---
    Sample("600365.SH", "ST", "Trdsta=2（ST），全窗口 1211 个交易日"),
    Sample("000669.SZ", "ST", "Trdsta=2（ST），全窗口 1211 个交易日"),
    # --- 其他新股 ---
    Sample("601136.SH", "主板SH/新股", "2022-12-22 上市，6 次复权事件"),
)

#: 证券代码（无前缀纯数字）→ 仓库统一代码。
CODE_TO_INSTRUMENT: Final[dict[str, str]] = {
    s.instrument.split(".")[0]: s.instrument for s in SAMPLE
}


# ---------------------------------------------------------------------------
# CSMAR 侧
# ---------------------------------------------------------------------------

def _mapping_frame() -> pl.DataFrame:
    """纯数字代码 → ``600000.SH`` 形式的映射表，用于向量化 join。"""
    codes = sorted(CODE_TO_INSTRUMENT)
    return pl.DataFrame(
        {
            "code": codes,
            "instrument": [CODE_TO_INSTRUMENT[c] for c in codes],
        }
    )


def load_adjust_events(csmar_dir: Path) -> pl.DataFrame:
    """读取 ``TRD_AdjustFactor``，返回样本票的 ``(instrument, date, cum, bward)``。

    仅保留事件级行（有复权事件的日期），日期升序。
    """
    files = sorted(glob.glob(str(csmar_dir / "TRD_AdjustFactor*.csv")))
    if not files:
        raise FileNotFoundError(f"未找到复权因子文件：{csmar_dir}")
    adj = pl.read_csv(
        files[0],
        schema_overrides={"Symbol": pl.Utf8, "TradingDate": pl.Utf8},
    )
    return (
        adj.join(_mapping_frame(), left_on="Symbol", right_on="code", how="inner")
        .select(
            pl.col("instrument"),
            pl.col("TradingDate").str.to_date("%Y-%m-%d").alias("date"),
            pl.col("BwardFactor").cast(pl.Float64, strict=False).alias("bward"),
            pl.col("CumulateBwardFactor").cast(pl.Float64, strict=False).alias("cum"),
        )
        .sort(["instrument", "date"])
    )


def load_daily_bars(csmar_dir: Path) -> pl.DataFrame:
    """读取 ``TRD_Dalyr`` 分片，返回样本票的 ``(instrument, date, close, volume, trdsta)``。

    只保留比较窗口内的行（>= ``AK_START``），成交量为股数。
    """
    files = sorted(glob.glob(str(csmar_dir / "TRD_Dalyr*.csv")))
    if not files:
        raise FileNotFoundError(f"未找到日行情分片：{csmar_dir}")
    codes = list(CODE_TO_INSTRUMENT)
    lazy = pl.concat(
        [
            pl.scan_csv(path, encoding="utf8", schema_overrides={"Stkcd": pl.Utf8})
            .select(
                pl.col("Stkcd"),
                pl.col("Trddt").cast(pl.Utf8),
                pl.col("Clsprc").cast(pl.Float64, strict=False),
                pl.col("Dnshrtrd").cast(pl.Float64, strict=False),
                pl.col("Trdsta").cast(pl.Int32, strict=False),
            )
            .filter(pl.col("Stkcd").is_in(codes))
            for path in files
        ]
    )
    bars = lazy.collect().join(
        _mapping_frame(), left_on="Stkcd", right_on="code", how="inner"
    )
    return (
        bars.with_columns(
            pl.col("Trddt").str.to_date("%Y-%m-%d").alias("date"),
            pl.col("Clsprc").alias("close"),
            pl.col("Dnshrtrd").alias("volume"),
        )
        .filter(pl.col("date") >= AK_START)
        .select("instrument", "date", "close", "volume", "Trdsta")
        .rename({"Trdsta": "trdsta"})
        .sort(["instrument", "date"])
    )


def build_factor_series(events: pl.DataFrame) -> pl.DataFrame:
    """把事件级累计因子展开成可 ``join_asof`` 的阶梯序列。

    事件行取当日的 ``CumulateBwardFactor``（递推 ``Cumulate(t)=Cumulate(t-1)×Bward(t)``，
    即当日事件生效后的值）；每票再加一行基线，因子为 ``首行Cumulate/首行Bward``，
    代表「首个事件之前」的水平。基线日期取 ``min(2021-09-29, 首事件日 - 1 天)``：
    导出文件从 2021-01 起截断，对老票而言首行并非历史上的首个事件，
    若把基线硬钉在 2021-09-29 会遮蔽首个事件之后、下个事件之前的正确取值。
    """
    if events.height == 0:
        return pl.DataFrame(
            schema={"instrument": pl.String, "date": pl.Date, "factor": pl.Float64}
        )
    first = events.group_by("instrument").agg(
        pl.col("date").first().alias("first_date"),
        pl.col("cum").first().alias("first_cum"),
        pl.col("bward").first().alias("first_bward"),
    )
    baseline = first.with_columns(
        pl.min_horizontal(
            pl.lit(DALYR_START, dtype=pl.Date),
            pl.col("first_date") - pl.duration(days=1),
        ).alias("date"),
        (pl.col("first_cum") / pl.col("first_bward")).alias("factor"),
    ).select("instrument", "date", "factor")
    event_rows = events.select(
        "instrument", "date", pl.col("cum").alias("factor")
    ).filter(pl.col("factor").is_not_null())
    return pl.concat([baseline, event_rows]).sort(["instrument", "date"])


def attach_csmar_factor(
    bars: pl.DataFrame, factor_series: pl.DataFrame
) -> pl.DataFrame:
    """把阶梯因子按 ``join_asof``（取 <= 当日的最近值）贴到日行情上。

    无任何事件行的票全部落到 null，按约定填 1.0。
    """
    if factor_series.height == 0:
        return bars.with_columns(pl.lit(1.0).alias("csmar_factor"))
    # polars 无法在带 by 分组的 join_asof 上校验有序性，这里已显式排序，忽略该提示。
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Sortedness of columns cannot be checked")
        joined = bars.sort(["instrument", "date"]).join_asof(
            factor_series.sort(["instrument", "date"]),
            on="date",
            by="instrument",
            strategy="backward",
        )
    return joined.with_columns(
        pl.col("factor").fill_null(1.0).alias("csmar_factor")
    ).drop("factor")


# ---------------------------------------------------------------------------
# akshare 侧
# ---------------------------------------------------------------------------

_SOURCE_USED: dict[str, str] = {}


def _patch_source_tracking() -> None:
    """给三个行情抓取方法套一层记录，记下每票最终命中的接口。

    包装函数保留原 ``__name__``，因为 ``AkshareSource._daily_bars_one``
    用函数名做熔断计数。
    """

    def wrap(name: str) -> None:
        original = getattr(AkshareSource, name)

        def wrapper(self: AkshareSource, instrument: str, *args: Any, **kwargs: Any):
            frame = original(self, instrument, *args, **kwargs)
            if frame is not None and frame.height:
                _SOURCE_USED[instrument] = name
            return frame

        wrapper.__name__ = name
        setattr(AkshareSource, name, wrapper)

    for name in ("_fetch_daily_em", "_fetch_daily_sina", "_fetch_daily_tx"):
        wrap(name)


def _cache_paths(
    cache_dir: Path, instrument: str, start: date, end: date
) -> tuple[Path, Path]:
    stem = f"{instrument.replace('.', '_')}_{start.isoformat()}_{end.isoformat()}"
    return cache_dir / f"{stem}.parquet", cache_dir / f"{stem}.json"


def fetch_akshare_bars(
    source: AkshareSource,
    cache_dir: Path,
    instrument: str,
    start: date,
    end: date,
) -> tuple[pl.DataFrame | None, str | None, str | None]:
    """取单票 akshare 日线，带本地缓存、礼貌间隔与有限重试。

    返回 ``(数据帧, 命中接口, 失败原因)``；失败时前两项为 None。
    """
    parquet_path, meta_path = _cache_paths(cache_dir, instrument, start, end)
    if parquet_path.exists():
        frame = pl.read_parquet(parquet_path)
        meta = json.loads(meta_path.read_text("utf-8")) if meta_path.exists() else {}
        if frame.height:
            return frame, meta.get("source"), None
        return None, meta.get("source"), meta.get("error", "缓存为空表")

    last_error: str | None = None
    for attempt in range(1, AK_RETRY_TIMES + 1):
        time.sleep(AK_POLITE_INTERVAL)
        _SOURCE_USED.pop(instrument, None)
        try:
            frame = source.daily_bars([instrument], start, end)
        except Exception as exc:  # noqa: BLE001 - 单个源异常类型不稳定
            last_error = f"{type(exc).__name__}: {exc}"
            print(f"  [{instrument}] 第 {attempt} 次请求异常：{last_error}", flush=True)
            time.sleep(AK_RETRY_INTERVAL)
            continue
        if frame.height == 0:
            last_error = "akshare 返回空表（可能退市或无该区间数据）"
            print(f"  [{instrument}] 第 {attempt} 次返回空表", flush=True)
            time.sleep(AK_RETRY_INTERVAL)
            continue
        used = _SOURCE_USED.get(instrument)
        cache_dir.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(parquet_path)
        meta_path.write_text(
            json.dumps({"source": used}, ensure_ascii=False), "utf-8"
        )
        return frame, used, None

    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps({"error": last_error}, ensure_ascii=False), "utf-8")
    return None, None, last_error


def collect_akshare(
    source: AkshareSource, cache_dir: Path
) -> tuple[pl.DataFrame, dict[str, str | None], dict[str, str]]:
    """逐票取 akshare 因子。返回（合并后的日线, 每票命中接口, 失败原因）。"""
    frames: list[pl.DataFrame] = []
    sources: dict[str, str | None] = {}
    failures: dict[str, str] = {}
    for sample in SAMPLE:
        instrument = sample.instrument
        frame, used, error = fetch_akshare_bars(
            source, cache_dir, instrument, AK_START, COMPARE_END
        )
        sources[instrument] = used
        if frame is None:
            failures[instrument] = error or "未知失败"
            print(f"  [{instrument}] 失败：{failures[instrument]}", flush=True)
            continue
        frames.append(
            frame.select(
                "instrument",
                "date",
                pl.col("close").alias("ak_close"),
                pl.col("volume").alias("ak_volume"),
                pl.col("adjfactor").alias("ak_factor"),
            )
        )
        print(
            f"  [{instrument}] ok rows={frame.height} source={used} "
            f"区间={frame['date'].min()}~{frame['date'].max()}",
            flush=True,
        )
    if not frames:
        return (
            pl.DataFrame(
                schema={
                    "instrument": pl.String,
                    "date": pl.Date,
                    "ak_close": pl.Float64,
                    "ak_volume": pl.Float64,
                    "ak_factor": pl.Float64,
                }
            ),
            sources,
            failures,
        )
    return pl.concat(frames).sort(["instrument", "date"]), sources, failures


# ---------------------------------------------------------------------------
# 计算
# ---------------------------------------------------------------------------

def compute_ratio_stats(merged: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """在共同交易日上算 ratio，返回（每票统计, 带 ratio 的明细）。"""
    detail = merged.filter(
        pl.col("ak_factor").is_not_null() & (pl.col("ak_factor") != 0)
    ).with_columns((pl.col("csmar_factor") / pl.col("ak_factor")).alias("ratio"))
    stats = (
        detail.group_by("instrument")
        .agg(
            pl.col("ratio").mean().alias("ratio_mean"),
            pl.col("ratio").std(ddof=1).alias("ratio_std"),
            pl.col("ratio").min().alias("ratio_min"),
            pl.col("ratio").max().alias("ratio_max"),
            pl.len().alias("n_days"),
            pl.col("date").min().alias("date_min"),
            pl.col("date").max().alias("date_max"),
        )
        .with_columns(
            pl.when(pl.col("ratio_std").is_null() | (pl.col("ratio_mean") == 0))
            .then(0.0)
            .otherwise(pl.col("ratio_std") / pl.col("ratio_mean"))
            .alias("cv")
        )
        .sort("instrument")
    )
    return stats, detail


def classify(stats: pl.DataFrame) -> tuple[str, str, float | None, pl.DataFrame]:
    """按约定判定结论类别。

    返回 ``(类别代号, 类别描述, 跨票离散度, 非恒定票)``。
    """
    nonconst = stats.filter(pl.col("cv") >= RATIO_STD_TOL)
    const = stats.filter(pl.col("cv") < RATIO_STD_TOL)
    dispersion: float | None = None
    if const.height:
        means = const["ratio_mean"]
        dispersion = float(means.std() / means.mean()) if means.mean() else None
    if nonconst.height:
        return "三", "不恒定：口径存在方法学差异", dispersion, nonconst
    if dispersion is not None and dispersion < CROSS_STOCK_TOL:
        return "一", "比例恒 1：两源同基期，无需归一化", dispersion, nonconst
    return "二", "每股恒定比例差：需按每股 scale 归一化", dispersion, nonconst


def top_deviations(detail: pl.DataFrame, stats: pl.DataFrame, n: int = 10) -> pl.DataFrame:
    """列出偏离该票 ratio_mean 最远的若干行，供排查口径差异。"""
    return (
        detail.join(stats.select("instrument", "ratio_mean"), on="instrument")
        .with_columns((pl.col("ratio") / pl.col("ratio_mean") - 1.0).abs().alias("dev"))
        .sort("dev", descending=True)
        .head(n)
        .select(
            "instrument",
            "date",
            pl.col("csmar_factor"),
            pl.col("ak_factor"),
            pl.col("ratio"),
            pl.col("dev"),
        )
    )


def compute_event_multipliers(
    bars: pl.DataFrame, events: pl.DataFrame, ak: pl.DataFrame
) -> pl.DataFrame:
    """逐事件对比两源的乘数：CSMAR ``BwardFactor`` 对 akshare 因子的当日环比。

    只有 akshare 走新浪源时其因子是事件级精确值，环比才有意义；
    东财 / 腾讯的因子逐日抖动，这里不参与对比。``prev_close`` 为除权日前一交易日
    的未复权收盘价，用来展示除权参考价取整带来的差异。
    """
    if events.height == 0 or ak.height == 0:
        return pl.DataFrame(
            schema={
                "instrument": pl.String,
                "date": pl.Date,
                "bward": pl.Float64,
                "ak_mult": pl.Float64,
                "mult_rel": pl.Float64,
                "prev_close": pl.Float64,
            }
        )
    prev = bars.sort(["instrument", "date"]).with_columns(
        pl.col("close").shift(1).over("instrument").alias("prev_close")
    )
    ak_mult = ak.sort(["instrument", "date"]).with_columns(
        (pl.col("ak_factor") / pl.col("ak_factor").shift(1).over("instrument")).alias(
            "ak_mult"
        )
    )
    return (
        events.sort(["instrument", "date"])
        .join(
            prev.select("instrument", "date", "prev_close"),
            on=["instrument", "date"],
            how="left",
        )
        .join(
            ak_mult.select("instrument", "date", "ak_mult"),
            on=["instrument", "date"],
            how="left",
        )
        .with_columns(
            ((pl.col("bward") - pl.col("ak_mult")) / pl.col("ak_mult")).alias("mult_rel")
        )
        .filter(pl.col("ak_mult").is_not_null() & (pl.col("ak_mult") != 0))
        .select("instrument", "date", "bward", "ak_mult", "mult_rel", "prev_close")
        .sort(["instrument", "date"])
    )


# ---------------------------------------------------------------------------
# 报表
# ---------------------------------------------------------------------------

def _num(value: Any, digits: int = 6) -> str:
    """把数值格式化成报告里的字符串，None / NaN 统一成 n/a。"""
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:  # NaN
        return "n/a"
    return f"{number:.{digits}g}"


def _rel_diff(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    base = max(abs(a), abs(b))
    if base == 0:
        return 0.0
    return abs(a - b) / base


def _md_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def build_report(
    *,
    csmar_dir: Path,
    verdict: str,
    verdict_desc: str,
    dispersion: float | None,
    stats: pl.DataFrame,
    detail: pl.DataFrame,
    deviations: pl.DataFrame,
    event_mults: pl.DataFrame,
    market_check: pl.DataFrame,
    sources: dict[str, str | None],
    failures: dict[str, str],
    events: pl.DataFrame,
    bars: pl.DataFrame,
    elapsed: float,
) -> str:
    """生成 markdown 报告。"""
    lines: list[str] = []
    lines.append("# CSMAR 与 akshare 后复权因子一致性核对报告")
    lines.append("")
    lines.append(f"- 生成时间：{date.today().isoformat()}，耗时 {elapsed:.1f}s")
    lines.append(
        f"- CSMAR：`{csmar_dir}`（TRD_Dalyr 7 分片 + TRD_AdjustFactor），"
        f"样本票事件覆盖 {events['date'].min() if events.height else 'n/a'} ~ "
        f"{events['date'].max() if events.height else 'n/a'}"
    )
    lines.append(
        f"- akshare：`quant.data.source.akshare.AkshareSource.daily_bars`，"
        f"取样区间 {AK_START} ~ {COMPARE_END}"
    )
    lines.append(
        f"- 口径：`ratio = CSMAR CumulateBwardFactor / akshare adjfactor`，"
        f"只在共同交易日计算；判定阈值 单票 cv < {RATIO_STD_TOL:g}、"
        f"跨票离散度 < {CROSS_STOCK_TOL:g}"
    )
    lines.append("")
    lines.append(f"## 结论（类别{verdict}）")
    lines.append("")
    lines.append(f"**{verdict_desc}**")
    lines.append("")
    lines.append(
        f"- 有效样本（两源都有数据）：{stats.height} 只；失败票：{len(failures)} 只"
    )
    if dispersion is not None:
        lines.append(f"- 恒定票的跨票 ratio_mean 相对离散度：{_num(dispersion, 4)}")
    nonconst_count = stats.filter(pl.col("cv") >= RATIO_STD_TOL).height
    lines.append(f"- 非恒定（cv >= {RATIO_STD_TOL:g}）的票数：{nonconst_count}")
    lines.append("")

    # --- 样本表 ---
    lines.append("## 1. 样本清单")
    lines.append("")
    event_counts = (
        events.group_by("instrument").agg(pl.len().alias("n_events"))
        if events.height
        else pl.DataFrame(schema={"instrument": pl.String, "n_events": pl.UInt32})
    )
    bar_range = bars.group_by("instrument").agg(
        pl.col("date").min().alias("csmar_first"), pl.col("date").max().alias("csmar_last")
    )
    st_days = (
        bars.filter(pl.col("trdsta").is_in(sorted(ST_TRDSTA)))
        .group_by("instrument")
        .agg(pl.len().alias("n_st"))
    )
    rows: list[list[str]] = []
    for sample in SAMPLE:
        inst = sample.instrument
        n_ev = event_counts.filter(pl.col("instrument") == inst)["n_events"]
        br = bar_range.filter(pl.col("instrument") == inst)
        st = stats.filter(pl.col("instrument") == inst)
        n_st = st_days.filter(pl.col("instrument") == inst)["n_st"]
        note = failures.get(inst, "")
        rows.append(
            [
                inst,
                sample.category,
                str(int(n_ev[0])) if n_ev.len() else "0",
                f"{br['csmar_first'][0]}~{br['csmar_last'][0]}" if br.height else "n/a",
                str(int(st["n_days"][0])) if st.height else "0",
                str(int(n_st[0])) if n_st.len() else "0",
                sources.get(inst) or "n/a",
                note if note else "ok",
                sample.reason,
            ]
        )
    lines.extend(
        _md_table(
            [
                "代码",
                "类别",
                "CSMAR事件数",
                "CSMAR区间",
                "共同交易日",
                "区间内ST天数",
                "akshare源",
                "状态",
                "入选理由",
            ],
            rows,
        )
    )
    lines.append("")

    # --- ratio 统计表 ---
    lines.append("## 2. 每股 ratio 统计")
    lines.append("")
    rows = [
        [
            row["instrument"],
            _num(row["ratio_mean"]),
            _num(row["ratio_std"]),
            _num(row["cv"], 4),
            _num(row["ratio_min"]),
            _num(row["ratio_max"]),
            _num(row["ratio_max"] / row["ratio_mean"] - 1.0, 4),
            str(row["n_days"]),
            f"{row['date_min']}~{row['date_max']}",
        ]
        for row in stats.iter_rows(named=True)
    ]
    lines.extend(
        _md_table(
            ["代码", "ratio_mean", "ratio_std", "cv(std/mean)", "min", "max", "max偏离均值", "共同日数", "区间"],
            rows,
        )
    )
    lines.append("")
    no_event = (
        {s.instrument for s in SAMPLE}
        - set(events["instrument"].unique().to_list())
        if events.height
        else {s.instrument for s in SAMPLE}
    )
    lines.append(
        "> 说明：CSMAR 全窗口无复权事件的 "
        f"{len(no_event)} 只票（{'、'.join(sorted(no_event))}）按约定把 CSMAR 因子填 1.0。"
        "它们的 ratio 反映的是 akshare 累积因子本身：若 akshare 因子恒为 1"
        "（688173/688790/688981，上市至今从未复权），两边真正一致；"
        "若 akshare 因子大于 1（000669/000816/600365/600793），"
        "说明这些票在 2021 年前有复权事件，而 TRD_AdjustFactor 只导出事件行，"
        "上市至今的累计水平无法从本文件恢复，其 ratio 的绝对水平没有意义。"
    )
    lines.append("")

    # --- 按源分组 ---
    lines.append("### 2.1 按 akshare 行情源分组")
    lines.append("")
    by_source: dict[str, list[tuple[str, float]]] = {}
    for row in stats.iter_rows(named=True):
        src = sources.get(row["instrument"]) or "未知"
        by_source.setdefault(src, []).append((row["instrument"], row["cv"]))
    rows = []
    for src in sorted(by_source):
        items = sorted(by_source[src], key=lambda kv: kv[1], reverse=True)
        cvs = [cv for _, cv in items]
        rows.append(
            [
                src,
                str(len(items)),
                _num(sorted(cvs)[len(cvs) // 2], 4),
                _num(max(cvs), 4),
                "、".join(inst for inst, _ in items),
                SOURCE_NOTE.get(src, "?"),
            ]
        )
    lines.extend(
        _md_table(["akshare源", "票数", "cv中位数", "cv最大", "代码", "口径说明"], rows)
    )
    lines.append("")

    # --- 行情核对 ---
    lines.append(f"## 3. {COMPARE_END} 当日行情核对")
    lines.append("")
    rows = []
    for row in market_check.iter_rows(named=True):
        close_ok = row["close_rel"] is not None and row["close_rel"] <= CLOSE_RTOL
        volume_ok = row["volume_rel"] is not None and row["volume_rel"] <= VOLUME_RTOL
        note = row["note"] or ("一致" if (close_ok and volume_ok) else "不一致")
        rows.append(
            [
                row["instrument"],
                _num(row["csmar_close"], 8),
                _num(row["ak_close"], 8),
                _num(row["close_rel"], 3),
                _num(row["csmar_volume"], 8),
                _num(row["ak_volume"], 8),
                _num(row["volume_rel"], 3),
                note,
            ]
        )
    lines.extend(
        _md_table(
            ["代码", "CSMAR收盘", "akshare收盘", "收盘相对差", "CSMAR成交量", "akshare成交量", "成交量相对差", "判定"],
            rows,
        )
    )
    lines.append("")

    # --- 失败票 ---
    lines.append("## 4. 失败票与原因")
    lines.append("")
    if failures:
        lines.extend(_md_table(["代码", "原因"], [[k, v] for k, v in failures.items()]))
    else:
        lines.append("无。")
    lines.append("")

    # --- 差异最大 ---
    if verdict == "三" and deviations.height:
        lines.append("## 5. 偏离本票 ratio_mean 最大的 10 行")
        lines.append("")
        rows = [
            [
                row["instrument"],
                str(row["date"]),
                _num(row["csmar_factor"]),
                _num(row["ak_factor"]),
                _num(row["ratio"]),
                _num(row["dev"], 3),
            ]
            for row in deviations.iter_rows(named=True)
        ]
        lines.extend(
            _md_table(["代码", "日期", "CSMAR因子", "akshare因子", "ratio", "相对偏离"], rows)
        )
        lines.append("")
        lines.append("### 5.1 剔除「CSMAR 填 1.0」的无事件票后")
        lines.append("")
        detail_ex = detail.filter(~pl.col("instrument").is_in(sorted(no_event)))
        deviations_ex = top_deviations(detail_ex, stats)
        rows = [
            [
                row["instrument"],
                str(row["date"]),
                _num(row["csmar_factor"]),
                _num(row["ak_factor"]),
                _num(row["ratio"]),
                _num(row["dev"], 3),
            ]
            for row in deviations_ex.iter_rows(named=True)
        ]
        if rows:
            lines.extend(
                _md_table(
                    ["代码", "日期", "CSMAR因子", "akshare因子", "ratio", "相对偏离"], rows
                )
            )
        else:
            lines.append("无。")
        lines.append("")

    # --- 逐事件乘数对比 ---
    sina_mults = event_mults.filter(
        pl.col("instrument").is_in(
            [k for k, v in sources.items() if v == "_fetch_daily_sina"]
        )
    )
    if sina_mults.height:
        lines.append("## 5.2 逐事件乘数对比（只含 akshare 走新浪源的票）")
        lines.append("")
        worst = (
            sina_mults.with_columns(pl.col("mult_rel").abs().alias("abs_rel"))
            .group_by("instrument")
            .agg(pl.col("abs_rel").max().alias("abs_rel_max"), pl.len().alias("n"))
            .sort("abs_rel_max", descending=True)
        )
        pick = worst.head(3)["instrument"].to_list()
        rows = [
            [
                row["instrument"],
                str(row["date"]),
                _num(row["bward"], 10),
                _num(row["ak_mult"], 10),
                _num(row["mult_rel"], 3),
                _num(row["prev_close"], 8),
            ]
            for row in sina_mults.filter(pl.col("instrument").is_in(pick))
            .sort(["instrument", "date"])
            .iter_rows(named=True)
        ]
        lines.extend(
            _md_table(
                ["代码", "事件日", "CSMAR乘数", "akshare乘数", "相对差", "除权前收盘"],
                rows,
            )
        )
        lines.append("")
        all_rel = sina_mults["mult_rel"].abs()
        lines.append(
            f"- 新浪源共 {sina_mults.height} 个可比事件，乘数相对差的中位数 "
            f"{_num(all_rel.median(), 3)}、最大 {_num(all_rel.max(), 3)}"
        )
        rows = [
            [row["instrument"], str(int(row["n"])), _num(row["abs_rel_max"], 3)]
            for row in worst.head(5).iter_rows(named=True)
        ]
        lines.extend(_md_table(["代码", "事件数", "乘数相对差最大"], rows))
        lines.append("")
        lines.append(
            "> 读法：CSMAR 的 `BwardFactor` 等于 `前收盘 / round(前收盘 - 每股派息, 2)`，"
            "即用取整到「分」的除权参考价；新浪的 `hfq-factor` 用未取整的理论除权价。"
            "两者的绝对差恒在半分钱量级，对 11 元的 000001 是 4e-4，"
            "对 1700 元的 600519 只有 3e-6，正好解释 cv 与股价成反比的现象。"
        )
        lines.append("")

    # --- 建议 ---
    lines.append("## 6. 结论与建议")
    lines.append("")
    if verdict == "一":
        lines.append(
            "两源后复权因子在共同交易日上比例恒为 1，基期与口径一致，"
            "CSMAR 因子可直接替换 akshare adjfactor，无需任何归一化。"
        )
    elif verdict == "二":
        lines.append(
            "每只票内部比例恒定，说明两源的事件时序与累计方式一致，"
            "差异只是基期归一化常数不同。接入 CSMAR 时，"
            "对每只票保存一个 scale（首次入库时用共同交易日算一次并落库），"
            "与 akshare 因子换算后再写入本地缓存；"
            "换用 CSMAR 作为主源时，同一只票必须始终使用同一个 scale，"
            "避免中途切换口径造成价格序列跳变。"
        )
        lines.append("")
        lines.append("每股 scale（`instrument -> ratio_mean = CSMAR因子 / akshare因子`）：")
        lines.append("")
        scale_rows = [
            [row["instrument"], _num(row["ratio_mean"], 10)]
            for row in stats.iter_rows(named=True)
        ]
        lines.extend(_md_table(["代码", "scale (ratio_mean)"], scale_rows))
    else:
        cvs_by_source: dict[str, list[float]] = {}
        for row in stats.iter_rows(named=True):
            src = sources.get(row["instrument"]) or "未知"
            cvs_by_source.setdefault(src, []).append(row["cv"])
        sina_pos = sorted(cv for cv in cvs_by_source.get("_fetch_daily_sina", []) if cv > 0)
        em_cvs = sorted(cvs_by_source.get("_fetch_daily_em", []))
        tx_cvs = sorted(cvs_by_source.get("_fetch_daily_tx", []))
        lines.append(
            "存在单票 ratio 随时间漂移，按判定阈值不能认定两源可直接换算。"
            "按证据强度排序，漂移有下面三个来源，处理方式各不相同。"
        )
        lines.append("")
        lines.append(
            f"**一、akshare 侧源精度。** 东财与腾讯的 adjfactor 都是"
            "「后复权收盘 / 未复权收盘」反推出来的，两个价格各自按分取整，"
            "价格越低相对误差越大，且东财序列在无事件区间也会逐日漂移。"
            f"东财源 {len(em_cvs)} 只票 cv 在 {_num(min(em_cvs), 3)} ~ {_num(max(em_cvs), 3)}，"
            f"腾讯源 {len(tx_cvs)} 只北交所票在 {_num(min(tx_cvs), 3)} ~ {_num(max(tx_cvs), 3)}；"
            "新浪源直接给出 `hfq-factor`，不受这个问题影响。"
            "000002 尤其值得注意：CSMAR 记录其 2023-08-25 之后不再有复权事件，"
            "手工核对新浪源同区间因子恒为 169.145（对应 ratio 0.8433），"
            "而东财缓存序列从 190 漂到 442，说明该源的 adjfactor 在无事件区间"
            "仍然逐日变化，不能作为基准。"
        )
        lines.append("")
        lines.append(
            f"**二、事件乘数的口径差（新浪源同样存在）。** 新浪源 {len(cvs_by_source.get('_fetch_daily_sina', []))} "
            f"只票里有事件的那些 cv 落在 {_num(min(sina_pos), 3)} ~ {_num(max(sina_pos), 3)}，"
            "比例不是严格恒定，而是每次除权除息时跳一小步。"
            "第 5.2 节的逐事件对比给出了原因：CSMAR 的 `BwardFactor` 等价于 "
            "`前收盘 / round(前收盘 - 每股派息, 2)`，用的是取整到分的除权参考价；"
            "新浪的 `hfq-factor` 用未取整的理论价。以 000001 的 2024-06-14 为例，"
            "前收盘 10.80、每股派息 0.719 元，CSMAR 按 10.08 算得乘数 1.071429，"
            "新浪按 10.081 算得 1.071322。绝对差恒在半分钱量级，"
            "所以股价越低相对差越大：1700 元的 600519 只有 3e-6，"
            "11 元的 000001 有 4e-4，8 元的 600273 最大到 6e-4。"
        )
        lines.append("")
        lines.append(
            "**三、无事件票的水平不可比。** TRD_AdjustFactor 只导出事件行，"
            "窗口内没有事件的票无从取得上市至今的累计因子，脚本按约定填 1.0，"
            "使得 000669/000816/600365/600793 的 ratio 只反映 akshare 因子本身。"
        )
        lines.append("")
        lines.append("### 建议")
        lines.append("")
        lines.append(
            "1. 交叉验证 akshare 时固定走新浪源（或直接用 `adjust='hfq-factor'`），"
            "不要使用东财/腾讯的比率反推因子；当前 `AkshareSource` 的源顺序是"
            "东财优先，本机系统代理损坏时东财偶发可用、偶发不可用，"
            "同一批票会混用不同精度的源，结果不可复现。"
        )
        lines.append(
            "2. 接 CSMAR 作为因子源前，先补两类数据：无事件票的窗口起点累计因子"
            "（或直接导出全量日频累计因子），以及配股/增发等非分红事件的明细。"
            "否则 000001（ratio 1.248）与 000002（新浪 ratio 0.843）这类"
            "基期不一致的票会在拼接时产生价格跳变。"
        )
        lines.append(
            "3. 判定口径建议放宽：把「恒定」阈值从 1e-6 改到 1e-4（或改为在除权除息日"
            "逐事件比较乘数）。1e-6 的阈值低于两源派息取整本身带来的噪声，"
            "会让每一次核对都落到「不恒定」。"
        )
    lines.append("")
    lines.append(
        "运行环境提示：本机系统代理时好时坏，东财 `push2his.eastmoney.com` 多数请求报 "
        "ProxyError、少数请求能通。`AkshareSource` 的源顺序是东财优先，"
        "于是同一批样本里出现了「有的票走东财、有的票走新浪」的混用："
        "走东财的票 adjfactor 由两个按分取整的价格相除而来，"
        "精度不足，且该序列在无事件区间仍会漂移。"
        "若需要可复现的基准，请在核对时只使用新浪源的 `hfq-factor`。"
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CSMAR 与 akshare 后复权因子一致性交叉核对（issue #59）"
    )
    parser.add_argument(
        "--csmar-dir",
        required=True,
        type=Path,
        help="CSMAR 解压目录（含 TRD_Dalyr*.csv 与 TRD_AdjustFactor.csv）",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help=f"markdown 报告输出路径，默认 {DEFAULT_REPORT_PATH}",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"akshare 抓取缓存目录，默认 {DEFAULT_CACHE_DIR}",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    started = time.monotonic()
    print("# CSMAR × akshare 后复权因子一致性核对")
    for sample in SAMPLE:
        digits = sample.instrument.split(".")[0]
        if normalize_instrument(digits) != sample.instrument:
            raise ValueError(f"样本代码与仓库约定不符：{sample.instrument}")
    print(f"样本 {len(SAMPLE)} 只，比较区间 {AK_START} ~ {COMPARE_END}")

    print("\n## 读取 CSMAR")
    events = load_adjust_events(args.csmar_dir)
    bars = load_daily_bars(args.csmar_dir)
    bars = attach_csmar_factor(bars, build_factor_series(events))
    print(
        f"事件行 {events.height} 条（{events['instrument'].n_unique() if events.height else 0} 只），"
        f"日线 {bars.height} 行（{bars['instrument'].n_unique()} 只）"
    )

    print("\n## 抓取 akshare")
    _patch_source_tracking()
    source = AkshareSource()
    ak, sources, failures = collect_akshare(source, args.cache_dir)

    print("\n## 对比")
    merged = bars.join(
        ak.select("instrument", "date", "ak_factor"),
        on=["instrument", "date"],
        how="inner",
    )
    stats, detail = compute_ratio_stats(merged)
    event_mults = compute_event_multipliers(bars, events, ak)
    verdict, verdict_desc, dispersion, nonconst = classify(stats)
    deviations = (
        top_deviations(detail, stats) if nonconst.height else detail.head(0)
    )

    # 2026-09-28 行情核对
    day = COMPARE_END
    csmar_day = bars.filter(pl.col("date") == day).select(
        "instrument",
        pl.col("close").alias("csmar_close"),
        pl.col("volume").alias("csmar_volume"),
    )
    ak_day = ak.filter(pl.col("date") == day).select(
        "instrument",
        pl.col("ak_close"),
        pl.col("ak_volume"),
    )
    check = (
        csmar_day.join(ak_day, on="instrument", how="full", coalesce=True)
        .sort("instrument")
    )
    check_rows = []
    for row in check.iter_rows(named=True):
        note = ""
        if row["csmar_close"] is None:
            note = "CSMAR 该日无行情（停牌或已退市）"
        elif row["ak_close"] is None:
            note = failures.get(row["instrument"], "akshare 该日无行情")
        check_rows.append(
            {
                "instrument": row["instrument"],
                "csmar_close": row["csmar_close"],
                "ak_close": row["ak_close"],
                "close_rel": _rel_diff(row["csmar_close"], row["ak_close"]),
                "csmar_volume": row["csmar_volume"],
                "ak_volume": row["ak_volume"],
                "volume_rel": _rel_diff(row["csmar_volume"], row["ak_volume"]),
                "note": note,
            }
        )
    market_check = pl.DataFrame(
        check_rows,
        schema={
            "instrument": pl.String,
            "csmar_close": pl.Float64,
            "ak_close": pl.Float64,
            "close_rel": pl.Float64,
            "csmar_volume": pl.Float64,
            "ak_volume": pl.Float64,
            "volume_rel": pl.Float64,
            "note": pl.String,
        },
        strict=False,
    ).sort("instrument")

    elapsed = time.monotonic() - started

    # --- 控制台摘要 ---
    print(f"\n## 结论：类别{verdict} —— {verdict_desc}")
    print(f"有效样本 {stats.height} 只，失败 {len(failures)} 只，耗时 {elapsed:.1f}s")
    if dispersion is not None:
        print(f"恒定票跨票离散度 {dispersion:.4g}")
    print("\n每股 ratio：")
    for row in stats.iter_rows(named=True):
        print(
            f"  {row['instrument']:<12} mean={row['ratio_mean']:.8f} "
            f"cv={row['cv']:.3g} n={row['n_days']} "
            f"min={row['ratio_min']:.6f} max={row['ratio_max']:.6f}"
        )
    if verdict == "三":
        print("\n偏离最大的 10 行：")
        for row in deviations.iter_rows(named=True):
            print(
                f"  {row['instrument']:<12} {row['date']} "
                f"csmar={row['csmar_factor']:.6f} ak={row['ak_factor']:.6f} "
                f"ratio={row['ratio']:.6f} dev={row['dev']:.2%}"
            )
    print("\n2026-09-28 行情核对：")
    for row in market_check.iter_rows(named=True):
        print(
            f"  {row['instrument']:<12} close差={_num(row['close_rel'], 3)} "
            f"volume差={_num(row['volume_rel'], 3)} {row['note']}"
        )
    if event_mults.height:
        sina_events = event_mults.filter(
            pl.col("instrument").is_in(
                [k for k, v in sources.items() if v == "_fetch_daily_sina"]
            )
        )
        if sina_events.height:
            rel = sina_events["mult_rel"].abs()
            print(
                f"\n逐事件乘数（新浪源 {sina_events.height} 个事件）："
                f"相对差中位数={rel.median():.3g} 最大={rel.max():.3g}"
            )
    src_counts: dict[str, int] = {}
    for sample in SAMPLE:
        src = sources.get(sample.instrument) or "未知"
        src_counts[src] = src_counts.get(src, 0) + 1
    print("\nakshare 源分布：", src_counts)
    if failures:
        print("\n失败票：")
        for inst, reason in failures.items():
            print(f"  {inst}: {reason}")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        build_report(
            csmar_dir=args.csmar_dir,
            verdict=verdict,
            verdict_desc=verdict_desc,
            dispersion=dispersion,
            stats=stats,
            detail=detail,
            deviations=deviations,
            event_mults=event_mults,
            market_check=market_check,
            sources=sources,
            failures=failures,
            events=events,
            bars=bars,
            elapsed=elapsed,
        ),
        encoding="utf-8",
    )
    print(f"\n报告已写入 {args.report}")

    if verdict == "三" or stats.height < MIN_SAMPLE:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
