"""用 CSMAR 逐日交易状态（Trdsta）离线构建 ST 区间表（issue #59 配套，服务 #5）。

``precompute_limits.py`` 的缺省 ST 来源是 akshare 曾用名历史（全市场逐票公网
请求，慢且易限流）。CSMAR 导出包自带的 ``TRD_Dalyr.Trdsta`` 是交易所口径的逐日
交易状态，质量更高且完全离线。本脚本把建库时导出的
``data/csmar_limit_reference.parquet`` 转成 ``data/st_intervals.parquet``
（schema 见 :data:`quant.data.limit.ST_INTERVALS`），``precompute_limits.py``
检测到该文件存在即直接使用，不再触网。

口径
----
- ST 判定只对主板票有意义：创业板 / 科创板 / 北交所的 ST 票与非 ST 票涨跌幅
  相同（20% / 20% / 30%），``limit_ratio`` 不看这些板块的 is_st，区间表只保留主板。
- 主板 ST 判定以**交易所实际执行幅度**为准：当日 ``ref_limit_up / ref_pre_close - 1
  < 0.08`` 即按 ST（5% 带）处理。CSMAR 的 ``Trdsta`` 标注与交易所执行存在脱节
  （实测 603822.SH 2026-07-06 摘帽恢复 10%，Trdsta 到导出日仍标 2；全市场约
  8400 行 Trdsta 标 ST 但按 10% 执行），broker 关心的是实际可成交边界，
  故以执行幅度为准。``ref_limit_up`` / ``ref_pre_close`` 缺失的日子回退
  ``Trdsta ∈ {2, 3, 5, 6}``。
- 2026-07-06 起主板 ST 涨跌幅并入 10%（交易所规则变更，见
  :data:`quant.data.limit.MAIN_ST_UNIFY_DATE`），此后主板不再存在 5% 带，
  强制判非 ST：区间在切换日前自然终止，也避免 ref 缺失日回退 Trdsta 造成误判。
- 连续段按交易日行序划分：两段 ST 之间隔一个非 ST 交易日即分为两个区间，
  横跨周末 / 假期（无行情行）不影响连续性。
- 区间结束日恰好是参考数据最后一天（即当前仍 ST）时 ``end_date`` 置 null
  （「至今」），与 ``build_st_intervals`` 的约定一致；窗口首日即 ST 的票
  起点取窗口首日（窗口外历史不可知，对窗口内的涨跌停计算无影响）。

用法::

    uv run python scripts/build_st_intervals_from_csmar.py --data-dir data
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl  # noqa: E402

from quant.data.limit import (  # noqa: E402
    MAIN_ST_UNIFY_DATE,
    ST_INTERVALS,
)
from quant.data.schema import board_of  # noqa: E402

#: CSMAR 建库时导出的涨跌停 / 交易状态参考表（scripts/build_csmar.py 产物）。
LIMIT_REFERENCE_FILE: str = "csmar_limit_reference.parquet"
#: 输出文件名（与 precompute_limits.py / daily_update.py 的约定一致）。
ST_INTERVALS_FILENAME: str = "st_intervals.parquet"

#: Trdsta 中代表 ST 的取值（ST / *ST / SST / S*ST），仅作无参照数据时的回退。
ST_TRDSTA: tuple[int, ...] = (2, 3, 5, 6)
#: 主板涨停幅度低于此值即判定当日按 ST 的 5% 带执行（非 ST 为 10%，裕量充足）。
ST_BAND_MAX: float = 0.08


def build_st_intervals_from_trdsta(reference: pl.DataFrame) -> pl.DataFrame:
    """把逐日交易状态表折叠成 ST 区间表（仅主板，判定以交易所执行幅度为准）。

    输入列：``date`` / ``instrument`` / ``trdsta`` / ``ref_limit_up`` /
    ``ref_pre_close``。输出 :data:`quant.data.limit.ST_INTERVALS`，按
    ``(instrument, start_date)`` 排序。
    """
    main_only = reference.filter(
        pl.col("instrument").map_elements(board_of, return_dtype=pl.String) == "main"
    )
    marked = (
        main_only.select(
            "instrument", "date", "trdsta", "ref_limit_up", "ref_pre_close"
        )
        .sort(["instrument", "date"])
        .with_columns(
            pl.when(
                pl.col("ref_limit_up").is_not_null()
                & pl.col("ref_pre_close").is_not_null()
                & (pl.col("ref_pre_close") > 0)
            )
            # 无限制日的哨兵 99999.99 幅度远超 ST 带，自然落到非 ST。
            .then(pl.col("ref_limit_up") / pl.col("ref_pre_close") - 1 < ST_BAND_MAX)
            .otherwise(pl.col("trdsta").is_in(ST_TRDSTA))
            .alias("_is_st_raw")
        )
        .with_columns(
            # 2026-07-06 起主板 ST 并入 10% 带，ST 标记对涨跌停失去意义。
            (pl.col("_is_st_raw") & (pl.col("date") < MAIN_ST_UNIFY_DATE)).alias(
                "_is_st"
            )
        )
        .with_columns(
            (
                pl.col("_is_st")
                != pl.col("_is_st").shift(1).over("instrument", order_by="date")
            )
            .fill_null(True)
            .cast(pl.UInt32)
            .cum_sum()
            .over("instrument", order_by="date")
            .alias("_seg")
        )
    )
    segments = (
        marked.filter(pl.col("_is_st"))
        .group_by("instrument", "_seg")
        .agg(
            pl.col("date").min().alias("start_date"),
            pl.col("date").max().alias("end_date"),
        )
        .sort(["instrument", "start_date"])
    )
    if segments.height == 0:
        return pl.DataFrame(schema=ST_INTERVALS)
    last_date = reference["date"].max()
    out = segments.with_columns(
        pl.when(pl.col("end_date") == last_date)
        .then(pl.lit(None, dtype=pl.Date))
        .otherwise(pl.col("end_date"))
        .alias("end_date")
    ).select("instrument", "start_date", "end_date")
    return out.cast(ST_INTERVALS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="用 CSMAR Trdsta 离线构建 st_intervals.parquet（issue #59 配套）"
    )
    parser.add_argument("--data-dir", default="data", help="缓存目录，默认 data/")
    args = parser.parse_args(argv)
    data_dir = Path(args.data_dir)

    ref_path = data_dir / LIMIT_REFERENCE_FILE
    if not ref_path.exists():
        print(f"缺少 {ref_path}，先运行 scripts/build_csmar.py", file=sys.stderr)
        return 1

    reference = pl.read_parquet(ref_path)
    intervals = build_st_intervals_from_trdsta(reference)

    out_path = data_dir / ST_INTERVALS_FILENAME
    intervals.write_parquet(out_path)

    n_instruments = intervals["instrument"].n_unique()
    n_open = intervals.filter(pl.col("end_date").is_null()).height
    print("# ST 区间构建完成")
    print(f"参考数据 {reference.height} 行（{reference['instrument'].n_unique()} 只）")
    print(f"ST 区间 {intervals.height} 段，涉及 {n_instruments} 只")
    print(f"其中当前仍 ST（end_date=null）：{n_open} 段")
    print(f"已写入 {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
