"""个股流通市值日频表（issue #64 权重锚口径）。

数据源
------
CSMAR ``TRD_Dalyr*.csv`` 的 ``Dsmvosd``（日个股流通市值，单位千元，
``352214443.72`` 千元 ≈ 3522 亿元），从 ``HISTORY_START`` 起覆盖。落
``data/float_mv.parquet``，供指数权重锚（:mod:`quant.data.index_history`）与
后续组合模块按日截面取用。

口径
----
- ``Markettype`` 只保留 ``quant.data.source.csmar.KEPT_MARKETTYPES``（纯 A 股），
  交易所后缀按代码段判定（与 :mod:`quant.data.source.csmar` 同规则）。
- 只保留 ``float_mv > 0`` 的行；停牌日无行、市值为空的行剔除。
- CSV 统一 ``infer_schema_length=0`` 按字符串读取，规避 CSMAR 前导零丢失。

日更不在本模块范围：``Dsmvosd`` 目前只来自 CSMAR 离线导出，日常增量需要
CSMAR 定期导出包或 akshare 日截面补齐（见 issue #64 后续方案）。
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import polars as pl

from quant.data import cache
from quant.data.schema import (
    FLOAT_MV,
    HISTORY_START,
    check_schema,
)
from quant.data.source.csmar import (
    KEPT_MARKETTYPES,
    _exchange_expr,
    _instrument_expr,
    _to_date_expr,
)

logger = logging.getLogger(__name__)

#: CSMAR 日线文件名模式（分片为 ``TRD_Dalyr1.csv`` 等）。
DALYR_GLOB: str = "TRD_Dalyr*.csv"

#: 落盘文件名。
FLOAT_MV_FILE: str = "float_mv.parquet"


def csmar_dalyr_files(csmar_dir: Path | str) -> list[Path]:
    """列出 ``csmar_dir`` 下的 CSMAR 日线分片。"""
    return sorted(Path(csmar_dir).glob(DALYR_GLOB))


def read_float_mv(data_dir: Path) -> pl.DataFrame:
    """读取 ``float_mv.parquet``，缺失返回空表（``FLOAT_MV`` schema）。"""
    path = Path(data_dir) / FLOAT_MV_FILE
    if not path.exists():
        return pl.DataFrame(schema=FLOAT_MV)
    return (
        pl.read_parquet(path)
        .select(list(FLOAT_MV.keys()))
        .cast(FLOAT_MV)
        .sort(["instrument", "date"])
    )


def write_float_mv(data_dir: Path, frame: pl.DataFrame) -> Path:
    """把流通市值表原子写入 ``data_dir/float_mv.parquet``。"""
    out = (
        frame.select(list(FLOAT_MV.keys()))
        .cast(FLOAT_MV)
        .unique(subset=["date", "instrument"], keep="last")
        .sort(["instrument", "date"])
    )
    check_schema(out, FLOAT_MV, name="float_mv")
    path = Path(data_dir) / FLOAT_MV_FILE
    cache._atomic_write_parquet(path, out)
    return path


def build_float_mv(
    csmar_dir: Path | str,
    *,
    start: date = HISTORY_START,
    end: date | None = None,
) -> pl.DataFrame:
    """从 CSMAR 日线分片提取 ``(date, instrument, float_mv)``。

    逐分片惰性读取并投影 ``Stkcd`` / ``Trddt`` / ``Dsmvosd`` / ``Markettype``，
    过滤后 collect，避免整包载入内存。返回 ``FLOAT_MV`` schema，按
    ``(instrument, date)`` 排序。
    """
    files = csmar_dalyr_files(csmar_dir)
    if not files:
        raise FileNotFoundError(f"{csmar_dir} 下未找到 CSMAR 日线分片（模式 {DALYR_GLOB}）")

    frames: list[pl.DataFrame] = []
    for path in files:
        digits = pl.col("Stkcd").str.zfill(6)
        frame = (
            pl.scan_csv(path, encoding="utf8", infer_schema_length=0)
            .select("Stkcd", "Trddt", "Dsmvosd", "Markettype")
            .with_columns(
                digits.alias("digits"),
                _to_date_expr(pl.col("Trddt")).alias("date"),
                pl.col("Dsmvosd").cast(pl.Float64, strict=False).alias("float_mv"),
                pl.col("Markettype").cast(pl.Int64, strict=False).alias("markettype"),
            )
            .with_columns(_exchange_expr(pl.col("digits")).alias("exchange"))
            .filter(
                pl.col("markettype").is_in(KEPT_MARKETTYPES)
                & pl.col("exchange").is_not_null()
                & pl.col("date").is_not_null()
                & (pl.col("float_mv") > 0)
            )
            .with_columns(
                _instrument_expr(pl.col("digits"), pl.col("exchange")).alias(
                    "instrument"
                )
            )
            .select("date", "instrument", "float_mv")
            .collect()
        )
        frames.append(frame)

    out = pl.concat(frames, how="vertical_relaxed").filter(pl.col("date") >= start)
    if end is not None:
        out = out.filter(pl.col("date") <= end)
    out = (
        out.select(list(FLOAT_MV.keys()))
        .cast(FLOAT_MV)
        .unique(subset=["date", "instrument"], keep="last")
        .sort(["instrument", "date"])
    )
    check_schema(out, FLOAT_MV, name="float_mv")
    return out


__all__ = [
    "DALYR_GLOB",
    "FLOAT_MV",
    "FLOAT_MV_FILE",
    "build_float_mv",
    "csmar_dalyr_files",
    "read_float_mv",
    "write_float_mv",
]
