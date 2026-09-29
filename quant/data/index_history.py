"""CSMAR 指数取样变更表 → 历史成分快照与等效权重锚（issue #64 历史回填）。

背景
----
中证官网只公开**最新一期**成分名单与月末权重，2021-09 起的历史段无公开归档
（调研结论见 issue #64）。历史成分从 CSMAR 指数研究库的取样变更表重建：变更表
记录每次调样的「变更日期 + 成分证券代码 + 变动方式」，以官方最新成分名单为
基准集，按变更日期**从新到旧逆放**（新增→删除、剔除→加回）即可得到每个调样
生效日的完整名单。

口径与取舍
----------
- **逆放重建**：以官方最新快照为基准，只信变更流水。逆放路径与正放路径互为
  逆运算，:func:`check_reconstruction_invariant` 以正放回推做自洽校验。
- **生效日**：CSMAR ``Chgsmp01`` 变更日期即调样生效日；与
  :func:`quant.data.index_members.rebalance_effective_dates` 推定的定期窗口互验，
  不一致的记录属临时调样 / 剔除停牌票，快照照样落地。
- **权重降级**：变更表无权重字段。锚日权重用**流通市值加权**近似——调样生效日
  成分股按当日 CSMAR ``Dsmvosd``（日流通市值，千元）横截面归一。无市值数据的票
  剔除后重新归一，剔除清单单独返回。该口径与官方自由流通市值权重的偏差量级由
  :func:`compare_weights_to_official` 在官方锚日披露。
- **市值表**：流通市值由 :mod:`quant.data.float_mv` 从 CSMAR ``TRD_Dalyr`` 提取并
  落 ``data/float_mv.parquet``，本模块只消费该表。
- CSMAR 代码列带前导零，CSV 统一 ``infer_schema_length=0`` 按字符串读取（与
  :mod:`quant.data.source.csmar` 同口径）。
"""
from __future__ import annotations

import bisect
import logging
from datetime import date
from pathlib import Path
from typing import Sequence

import polars as pl

from quant.data.schema import (
    FLOAT_MV,
    HISTORY_START,
    INDEX_CODES,
    INDEX_MEMBER_CHANGES,
    INDEX_MEMBER_SNAPSHOTS,
    INDEX_WEIGHTS,
    check_schema,
    normalize_instrument,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

#: CSMAR 取样变更表文件名模式（导出包内可能分片为 ``IDX_Chgsmp1.csv`` 等）。
CSMAR_CHANGE_GLOB: str = "IDX_Chgsmp*.csv"

#: CSMAR 变更表列名。
CSMAR_INDEX_COL: str = "Indexcd"
CSMAR_DATE_COL: str = "Chgsmp01"
CSMAR_CODE_COL: str = "Chgsmp02"
CSMAR_MODE_COL: str = "Chgsmp04"
CSMAR_TYPE_COL: str = "Chgsmp05"

#: ``Chgsmp05`` 保留值：1=股票类（其余为基金 / 债券 / 期货等）。
CSMAR_STOCK_TYPE: str = "1"
#: ``Chgsmp04`` 变动方式：1=新增。
CHANGE_ADD: int = 1
#: ``Chgsmp04`` 变动方式：2=剔除。
CHANGE_DROP: int = 2

#: 等效权重表的剔除清单列。
_DROPPED_SCHEMA = pl.Schema(
    {"date": pl.Date, "index_code": pl.String, "instrument": pl.String}
)


def csmar_change_files(root: Path | str) -> list[Path]:
    """递归列出 ``root`` 下所有 CSMAR 取样变更表分片。"""
    return sorted(Path(root).rglob(CSMAR_CHANGE_GLOB))


def _normalize_silent(code: str) -> str | None:
    try:
        return normalize_instrument(code)
    except ValueError:
        return None


def _to_date_expr(text: pl.Expr) -> pl.Expr:
    return text.str.strip_chars().replace("", None).str.to_date("%Y-%m-%d", strict=False)


def read_csmar_changes(
    paths: Sequence[Path | str],
    *,
    index_codes: tuple[str, ...] = INDEX_CODES,
) -> pl.DataFrame:
    """读取并合并 CSMAR 取样变更表分片，输出 ``INDEX_MEMBER_CHANGES``。

    过滤 ``Chgsmp05 == 1``（股票类）与目标指数，按字符串读入规避前导零丢失，
    证券代码经 :func:`quant.data.schema.normalize_instrument` 归一化。
    多次导出重叠的分片按 ``(index_code, effective_date, instrument, change_type)`` 去重。
    """
    frames: list[pl.DataFrame] = []
    for path in paths:
        lf = pl.scan_csv(
            path,
            encoding="utf8-lossy",
            infer_schema_length=0,
        )
        # 先投影需要的列并按类型与指数过滤再 collect，避免把整包分片读进内存。
        frames.append(
            lf.select(
                CSMAR_INDEX_COL,
                CSMAR_DATE_COL,
                CSMAR_CODE_COL,
                CSMAR_MODE_COL,
                CSMAR_TYPE_COL,
            )
            .filter(
                (pl.col(CSMAR_TYPE_COL) == CSMAR_STOCK_TYPE)
                & pl.col(CSMAR_INDEX_COL).is_in(list(index_codes))
            )
            .collect()
        )
    if not frames:
        return pl.DataFrame(schema=INDEX_MEMBER_CHANGES)

    raw = pl.concat(frames, how="vertical")
    parsed = raw.select(
        pl.col(CSMAR_INDEX_COL).alias("index_code"),
        _to_date_expr(pl.col(CSMAR_DATE_COL)).alias("effective_date"),
        pl.col(CSMAR_CODE_COL)
        .str.zfill(6)
        .map_elements(_normalize_silent, return_dtype=pl.String)
        .alias("instrument"),
        pl.col(CSMAR_MODE_COL).cast(pl.Int8, strict=False).alias("change_type"),
    )
    skipped = parsed.filter(pl.col("instrument").is_null()).height
    if skipped:
        logger.warning("CSMAR 取样变更表跳过 %d 行无法归一化的证券代码", skipped)
    out = (
        parsed.drop_nulls(["effective_date", "instrument", "change_type"])
        .unique(subset=["index_code", "effective_date", "instrument", "change_type"])
        .sort(["index_code", "effective_date", "instrument"])
        .cast(INDEX_MEMBER_CHANGES)
    )
    check_schema(out, INDEX_MEMBER_CHANGES, name="index_member_changes")
    return out


def reconstruct_snapshots(
    changes: pl.DataFrame,
    base_snapshots: pl.DataFrame,
    *,
    coverage_start: date | None = HISTORY_START,
    index_codes: tuple[str, ...] = INDEX_CODES,
) -> pl.DataFrame:
    """以官方最新成分为基准逆放变更流水，重建历史成分快照。

    ``base_snapshots`` 为官方成分快照表，每个指数取 ``snapshot_date`` 最大的一份
    作为基准集。按变更日期从新到旧：先记录当日名单，再对当日变更逆放
    （新增→删除、剔除→加回）。``coverage_start`` 只保留「最近一次 ``<= coverage_start``
    的变更日」及其之后的快照，避免重建出早于建库窗口的无用名单；为 ``None`` 时全留。
    指数在首条变更之前的状态不生成快照（如 932000 在 2023-08-11 之前）。
    """
    if changes.height == 0 or base_snapshots.height == 0:
        return pl.DataFrame(schema=INDEX_MEMBER_SNAPSHOTS)

    changes = changes.select(list(INDEX_MEMBER_CHANGES.keys())).cast(
        INDEX_MEMBER_CHANGES
    )
    base_snapshots = base_snapshots.select(list(INDEX_MEMBER_SNAPSHOTS.keys())).cast(
        INDEX_MEMBER_SNAPSHOTS
    )
    rows: list[dict[str, object]] = []

    for code in index_codes:
        base = base_snapshots.filter(pl.col("index_code") == code)
        if base.height == 0:
            continue
        base_date = base["snapshot_date"].max()
        current = set(
            base.filter(pl.col("snapshot_date") == base_date)["instrument"].to_list()
        )
        sub = changes.filter(
            (pl.col("index_code") == code) & (pl.col("effective_date") <= base_date)
        )
        if sub.height == 0:
            continue

        dates = sorted(set(sub["effective_date"].to_list()), reverse=True)
        keep_from: date | None = None
        if coverage_start is not None:
            prior = [day for day in dates if day <= coverage_start]
            keep_from = prior[0] if prior else None  # dates 降序，prior[0] 是最近的 <= start

        for day in dates:
            if keep_from is None or day >= keep_from:
                rows.extend(
                    {
                        "snapshot_date": day,
                        "instrument": instrument,
                        "index_code": code,
                    }
                    for instrument in current
                )
            batch = sub.filter(pl.col("effective_date") == day)
            for row in batch.iter_rows(named=True):
                if row["change_type"] == CHANGE_ADD:
                    current.discard(row["instrument"])
                elif row["change_type"] == CHANGE_DROP:
                    current.add(row["instrument"])

    out = (
        pl.DataFrame(rows, schema=INDEX_MEMBER_SNAPSHOTS)
        if rows
        else pl.DataFrame(schema=INDEX_MEMBER_SNAPSHOTS)
    )
    out = (
        out.unique(subset=["index_code", "snapshot_date", "instrument"])
        .sort(["index_code", "snapshot_date", "instrument"])
        .cast(INDEX_MEMBER_SNAPSHOTS)
    )
    check_schema(out, INDEX_MEMBER_SNAPSHOTS, name="index_member_snapshots")
    return out


def check_reconstruction_invariant(
    changes: pl.DataFrame,
    snapshots: pl.DataFrame,
) -> None:
    """自洽校验：从最早快照正放变更，逐日集合应与逆放路径完全一致。

    不一致即抛 :class:`ValueError`。正放只在有快照的变更日比较；完整重建的
    快照集（:func:`reconstruct_snapshots` 的输出）应恒等通过。
    """
    if changes.height == 0 or snapshots.height == 0:
        return
    changes = changes.select(list(INDEX_MEMBER_CHANGES.keys())).cast(
        INDEX_MEMBER_CHANGES
    )
    snapshots = snapshots.select(list(INDEX_MEMBER_SNAPSHOTS.keys())).cast(
        INDEX_MEMBER_SNAPSHOTS
    )

    for code in snapshots["index_code"].unique().sort().to_list():
        sub_snaps = snapshots.filter(pl.col("index_code") == code)
        sub_changes = changes.filter(pl.col("index_code") == code)
        dates = sorted(set(sub_snaps["snapshot_date"].to_list()))
        if not dates:
            continue
        first = dates[0]
        current = set(
            sub_snaps.filter(pl.col("snapshot_date") == first)["instrument"].to_list()
        )
        for day in dates[1:]:
            batch = sub_changes.filter(pl.col("effective_date") == day)
            for row in batch.iter_rows(named=True):
                if row["change_type"] == CHANGE_ADD:
                    current.add(row["instrument"])
                elif row["change_type"] == CHANGE_DROP:
                    current.discard(row["instrument"])
            recorded = set(
                sub_snaps.filter(pl.col("snapshot_date") == day)["instrument"].to_list()
            )
            if current != recorded:
                missing = sorted(recorded - current)[:5]
                extra = sorted(current - recorded)[:5]
                raise ValueError(
                    f"成分重建不变式失败：{code}@{day} 正放 {len(current)} 票 / "
                    f"逆放 {len(recorded)} 票；缺失 {missing}，多余 {extra}"
                )


def float_mv_weights_on(
    members: pl.DataFrame,
    float_mv: pl.DataFrame,
    day: date,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """单指数单日流通市值权重：当日 ``Dsmvosd`` 横截面归一。

    ``members`` 为含 ``instrument`` 列的当日成分名单，``float_mv`` 为 ``FLOAT_MV``
    表。缺当日市值（停牌 / 未上市 / 市值非正）的票剔除后重新归一。返回
    ``(instrument, weight)`` 与剔除清单（``instrument`` 单列）。全部票无有效市值时
    返回空权重表。
    """
    empty_w = pl.DataFrame({"instrument": pl.String, "weight": pl.Float64})
    members = members.select("instrument").unique()
    mv = float_mv.filter(pl.col("date") == day).select("instrument", "float_mv")
    joined = members.join(mv, on="instrument", how="left")
    valid = joined.filter(pl.col("float_mv").is_not_null() & (pl.col("float_mv") > 0))
    dropped = (
        joined.filter(pl.col("float_mv").is_null() | (pl.col("float_mv") <= 0))
        .select("instrument")
        .sort("instrument")
    )
    if valid.height == 0:
        return empty_w, dropped
    total = valid["float_mv"].sum()
    weights = valid.select("instrument", (pl.col("float_mv") / total).alias("weight"))
    return weights, dropped


def build_float_mv_anchors(
    snapshots: pl.DataFrame,
    float_mv: pl.DataFrame,
    open_days: Sequence[date],
    *,
    index_codes: tuple[str, ...] = INDEX_CODES,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """由历史成分快照构造每个调样生效日的流通市值权重锚。

    ``open_days`` 为升序开市日（应已截到建库窗口）；快照日若不是开市日，锚日顺延到
    其后的首个开市日，同日取该指数较新的一份名单。返回 ``(INDEX_WEIGHTS, 剔除清单)``。
    """
    days = sorted(set(open_days))
    if snapshots.height == 0 or float_mv.height == 0 or not days:
        return pl.DataFrame(schema=INDEX_WEIGHTS), pl.DataFrame(schema=_DROPPED_SCHEMA)

    snapshots = snapshots.select(list(INDEX_MEMBER_SNAPSHOTS.keys())).cast(
        INDEX_MEMBER_SNAPSHOTS
    )
    float_mv = float_mv.select(list(FLOAT_MV.keys())).cast(FLOAT_MV)
    anchor_rows: list[dict[str, object]] = []
    dropped_rows: list[dict[str, object]] = []

    for code in index_codes:
        sub = snapshots.filter(pl.col("index_code") == code)
        if sub.height == 0:
            continue
        # 快照日顺延到首个开市日；同日多份名单时较新的一份生效。
        chosen: dict[date, list[str]] = {}
        for snap_date in sorted(set(sub["snapshot_date"].to_list())):
            pos = bisect.bisect_left(days, snap_date)
            if pos >= len(days):
                continue
            chosen[days[pos]] = sub.filter(pl.col("snapshot_date") == snap_date)[
                "instrument"
            ].to_list()
        for anchor_day in sorted(chosen):
            members = pl.DataFrame({"instrument": chosen[anchor_day]})
            weights, dropped = float_mv_weights_on(members, float_mv, anchor_day)
            anchor_rows.extend(
                {
                    "date": anchor_day,
                    "instrument": row["instrument"],
                    "index_code": code,
                    "weight": row["weight"],
                }
                for row in weights.iter_rows(named=True)
            )
            dropped_rows.extend(
                {"date": anchor_day, "index_code": code, "instrument": instrument}
                for instrument in dropped["instrument"].to_list()
            )

    anchors = (
        pl.DataFrame(anchor_rows, schema=INDEX_WEIGHTS)
        if anchor_rows
        else pl.DataFrame(schema=INDEX_WEIGHTS)
    )
    anchors = (
        anchors.unique(subset=["date", "instrument", "index_code"])
        .sort(["index_code", "date", "instrument"])
        .cast(INDEX_WEIGHTS)
    )
    check_schema(anchors, INDEX_WEIGHTS, name="index_weights")
    dropped_frame = (
        pl.DataFrame(dropped_rows, schema=_DROPPED_SCHEMA)
        if dropped_rows
        else pl.DataFrame(schema=_DROPPED_SCHEMA)
    )
    return anchors, dropped_frame


def compare_weights_to_official(
    official_anchors: pl.DataFrame,
    float_mv: pl.DataFrame,
) -> pl.DataFrame:
    """用同一流通市值口径重算官方锚日的权重，返回与官方值的逐票偏差明细。

    官方锚日成分名单取自 ``official_anchors``，权重用当日 ``Dsmvosd`` 构造。
    输出列 ``(anchor_date, index_code, instrument, float_mv_weight,
    official_weight, abs_deviation)``，用于披露近似口径的误差量级。
    """
    out_schema = pl.Schema(
        {
            "anchor_date": pl.Date,
            "index_code": pl.String,
            "instrument": pl.String,
            "float_mv_weight": pl.Float64,
            "official_weight": pl.Float64,
            "abs_deviation": pl.Float64,
        }
    )
    official_anchors = official_anchors.select(list(INDEX_WEIGHTS.keys())).cast(
        INDEX_WEIGHTS
    )
    rows: list[dict[str, object]] = []
    if official_anchors.height == 0:
        return pl.DataFrame(schema=out_schema)
    for code in official_anchors["index_code"].unique().sort().to_list():
        sub = official_anchors.filter(pl.col("index_code") == code)
        for day in sorted(set(sub["date"].to_list())):
            official = sub.filter(pl.col("date") == day)
            weights, _ = float_mv_weights_on(
                official.select("instrument"), float_mv, day
            )
            joined = weights.rename({"weight": "float_mv_weight"}).join(
                official.select(
                    "instrument", pl.col("weight").alias("official_weight")
                ),
                on="instrument",
                how="inner",
            )
            rows.extend(
                {
                    "anchor_date": day,
                    "index_code": code,
                    "instrument": row["instrument"],
                    "float_mv_weight": row["float_mv_weight"],
                    "official_weight": row["official_weight"],
                    "abs_deviation": abs(
                        row["float_mv_weight"] - row["official_weight"]
                    ),
                }
                for row in joined.iter_rows(named=True)
            )
    return pl.DataFrame(rows, schema=out_schema)


__all__ = [
    "CHANGE_ADD",
    "CHANGE_DROP",
    "CSMAR_CHANGE_GLOB",
    "build_float_mv_anchors",
    "check_reconstruction_invariant",
    "compare_weights_to_official",
    "csmar_change_files",
    "float_mv_weights_on",
    "read_csmar_changes",
    "reconstruct_snapshots",
]
