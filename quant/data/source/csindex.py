"""中证指数官网（csindex）成分名单与月末权重数据源（issue #64）。

接口
----
封装 akshare 的两个中证官方接口，输出本项目 schema：

- ``index_stock_cons_csindex(symbol)``：最新一期成分名单，字段含
  ``日期``（名单 as-of 交易日）、``成分券代码``。
- ``index_stock_cons_weight_csindex(symbol)``：最新一期月末样本权重，
  字段含 ``日期``（月末交易日）、``成分券代码``、``权重``（百分比）。

两个接口都只返回**最新一期**，没有历史。中证官网对历史月度权重无公开归档，
因此本数据源只服务于「官方锚的最新一份」：历史锚需外部离线导出（CSMAR 等）
后由 :mod:`quant.data.index_members` 的统一入口合并。

口径
----
- 成分快照 ``snapshot_date`` 取名单的 as-of 交易日；权重 ``weight`` 由百分比
  换算为小数（``权重 / 100``）。
- 成分券代码经 :func:`quant.data.schema.normalize_instrument` 归一化为
  ``600000.SH`` 形式；无法识别的代码跳过并记录。
"""
from __future__ import annotations

import logging

import akshare as ak
import polars as pl

from quant.data.schema import (
    INDEX_MEMBER_SNAPSHOTS,
    INDEX_WEIGHTS,
    check_schema,
    normalize_instrument,
)
from quant.data.source.akshare import _call, _from_pandas

logger = logging.getLogger(__name__)

#: 中证权重文件为百分比口径，换算为小数的系数。
WEIGHT_PERCENT_SCALE: float = 100.0


def _normalize_silent(code: str) -> str | None:
    try:
        return normalize_instrument(code)
    except ValueError:
        return None


class CsindexSource:
    """中证指数官网最新成分与权重数据源，满足 ``AnchorSource`` 协议。"""

    def member_snapshot(self, index_code: str) -> pl.DataFrame:
        """最新一期成分名单，返回 ``INDEX_MEMBER_SNAPSHOTS``。"""
        raw = _call(ak.index_stock_cons_csindex, symbol=index_code)
        df = _from_pandas(raw)
        if df.height == 0 or "成分券代码" not in df.columns:
            return pl.DataFrame(schema=INDEX_MEMBER_SNAPSHOTS)
        snapshot_date = df.select(pl.col("日期").cast(pl.Date).max()).item()
        if snapshot_date is None:
            return pl.DataFrame(schema=INDEX_MEMBER_SNAPSHOTS)
        codes = (
            df.select(pl.col("成分券代码").cast(pl.Utf8, strict=False).alias("code"))
            .with_columns(
                pl.col("code")
                .map_elements(_normalize_silent, return_dtype=pl.String)
                .alias("instrument")
            )
            .drop_nulls("instrument")
        )
        skipped = df.height - codes.height
        if skipped:
            logger.info("%s 成分名单跳过 %d 个无法归一化的代码", index_code, skipped)
        out = (
            codes.select(
                pl.lit(snapshot_date).alias("snapshot_date"),
                "instrument",
                pl.lit(index_code).alias("index_code"),
            )
            .unique()
            .sort(["index_code", "snapshot_date", "instrument"])
            .cast(INDEX_MEMBER_SNAPSHOTS)
        )
        check_schema(out, INDEX_MEMBER_SNAPSHOTS, name="index_member_snapshots")
        return out

    def weight_anchor(self, index_code: str) -> pl.DataFrame:
        """最新一期月末权重，返回 ``INDEX_WEIGHTS``（小数口径）。"""
        raw = _call(ak.index_stock_cons_weight_csindex, symbol=index_code)
        df = _from_pandas(raw)
        if df.height == 0 or "权重" not in df.columns:
            return pl.DataFrame(schema=INDEX_WEIGHTS)
        anchor_date = df.select(pl.col("日期").cast(pl.Date).max()).item()
        if anchor_date is None:
            return pl.DataFrame(schema=INDEX_WEIGHTS)
        codes = (
            df.select(
                pl.col("成分券代码").cast(pl.Utf8, strict=False).alias("code"),
                (pl.col("权重").cast(pl.Float64, strict=False) / WEIGHT_PERCENT_SCALE).alias(
                    "weight"
                ),
            )
            .with_columns(
                pl.col("code")
                .map_elements(_normalize_silent, return_dtype=pl.String)
                .alias("instrument")
            )
            .drop_nulls(["instrument", "weight"])
        )
        out = (
            codes.select(
                pl.lit(anchor_date).alias("date"),
                "instrument",
                pl.lit(index_code).alias("index_code"),
                "weight",
            )
            .unique(subset=["date", "instrument", "index_code"], keep="last")
            .sort(["index_code", "date", "instrument"])
            .cast(INDEX_WEIGHTS)
        )
        check_schema(out, INDEX_WEIGHTS, name="index_weights")
        return out


__all__ = ["CsindexSource", "WEIGHT_PERCENT_SCALE"]
