"""涨跌停上下限预计算 CLI。

用法::

    uv run python scripts/precompute_limits.py --data-dir data
    uv run python scripts/precompute_limits.py --instruments 600519.SH,300750.SZ
    uv run python scripts/precompute_limits.py --refresh-st

读取 ``data/bars/*.parquet`` 与 ``data/instruments.parquet``，构建 / 读取 ST 区间
（缓存到 ``data/st_intervals.parquet``），调用 :func:`quant.data.limit.compute_limits`
填充 ``limit_up`` / ``limit_down``，按自然年写回原文件（写回前过 ``check_daily_bars``）。

``--instruments`` 只限制 ST 区间构建范围，用于局部跑批；未列出的证券按非 ST 处理。
网络访问经由 ``AkshareSource`` 与曾用名接口，建议设置 ``NO_PROXY=*`` 绕过系统代理。
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl  # noqa: E402

from quant.data.limit import (  # noqa: E402
    ST_INTERVALS,
    build_st_intervals,
    compute_limits,
)
from quant.data.schema import (  # noqa: E402
    DAILY_BARS,
    INSTRUMENT_INFO,
    check_daily_bars,
    normalize_instrument,
)
from quant.data.source.akshare import AkshareSource  # noqa: E402

#: 默认数据目录。
DEFAULT_DATA_DIR: str = "data"
#: 年线文件所在子目录。
BARS_DIRNAME: str = "bars"
#: 证券信息文件名。
INSTRUMENTS_FILENAME: str = "instruments.parquet"
#: ST 区间缓存文件名。
ST_INTERVALS_FILENAME: str = "st_intervals.parquet"

logger = logging.getLogger("precompute_limits")


def load_bars(bars_dir: Path) -> pl.DataFrame:
    """读取并按 (instrument, date) 排序全部年线文件。"""
    files = sorted(bars_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"{bars_dir} 下没有 parquet 文件")
    frames = [pl.read_parquet(path) for path in files]
    out = pl.concat(frames).sort(["instrument", "date"]).cast(DAILY_BARS)
    logger.info("读取 %d 个年线文件，共 %d 行", len(files), out.height)
    return out


def load_instruments(data_dir: Path) -> pl.DataFrame:
    path = data_dir / INSTRUMENTS_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"缺少证券信息文件 {path}")
    return pl.read_parquet(path).cast(INSTRUMENT_INFO)


def load_or_build_st(
    data_dir: Path,
    instruments: list[str],
    *,
    refresh: bool,
    source: AkshareSource,
) -> pl.DataFrame:
    """读取 ST 区间缓存，缺失或 ``refresh`` 时重建并写回。"""
    path = data_dir / ST_INTERVALS_FILENAME
    if path.exists() and not refresh:
        out = pl.read_parquet(path).cast(ST_INTERVALS)
        logger.info("读取 ST 区间缓存 %s（%d 行）", path, out.height)
        return out
    out = build_st_intervals(source, instruments)
    out.write_parquet(path)
    logger.info("构建 ST 区间（%d 行）并缓存到 %s", out.height, path)
    return out


def write_limits_by_year(bars: pl.DataFrame, bars_dir: Path) -> dict[int, int]:
    """按自然年拆分写回，写回前逐文件校验，返回 {年份: 行数}。"""
    yearly = bars.with_columns(pl.col("date").dt.year().alias("_year"))
    counts: dict[int, int] = {}
    for year in sorted(yearly["_year"].unique().to_list()):
        frame = (
            yearly.filter(pl.col("_year") == year)
            .drop("_year")
            .sort(["instrument", "date"])
            .cast(DAILY_BARS)
        )
        check_daily_bars(frame, name=f"bars/{year}")
        frame.write_parquet(bars_dir / f"{year}.parquet")
        counts[year] = frame.height
        logger.info("写回 %s（%d 行）", bars_dir / f"{year}.parquet", frame.height)
    return counts


def parse_instruments(raw: list[str] | None) -> list[str]:
    """支持 ``--instruments A B`` 与 ``--instruments A,B`` 两种写法。"""
    if not raw:
        return []
    tokens = [token for chunk in raw for token in chunk.split(",")]
    return [normalize_instrument(token) for token in tokens if token.strip()]


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        pass

    parser = argparse.ArgumentParser(description="涨跌停上下限预计算")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="数据目录（默认 data）")
    parser.add_argument(
        "--instruments",
        nargs="*",
        default=None,
        help="仅预计算这些证券的 ST 区间（逗号或空格分隔），其余按非 ST 处理",
    )
    parser.add_argument("--refresh-st", action="store_true", help="忽略 ST 缓存，重新拉取")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    data_dir = Path(args.data_dir)
    bars_dir = data_dir / BARS_DIRNAME
    bars = load_bars(bars_dir)
    instruments = load_instruments(data_dir)

    wanted = parse_instruments(args.instruments)
    if not wanted:
        wanted = sorted(bars["instrument"].unique().to_list())
    logger.info("ST 区间覆盖 %d 只证券", len(wanted))

    st = load_or_build_st(
        data_dir, wanted, refresh=args.refresh_st, source=AkshareSource()
    )
    out = compute_limits(bars, instruments, st)
    filled = out.filter(pl.col("limit_up").is_not_null()).height
    logger.info("预计算完成：%d/%d 行填出涨跌停", filled, out.height)
    write_limits_by_year(out, bars_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
