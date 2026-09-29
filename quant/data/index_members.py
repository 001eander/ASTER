"""指数成分与权重的日频 PIT 落地（issue #64）。

数据链路
--------
1. **成分快照**（``index_member_snapshots.parquet``）：官方成分名单的原始落盘，
   每只指数在一个生效日对应一份完整名单。
2. **权重锚**（``index_weights.parquet`` 的锚行）：中证官方月末权重，小数口径。
3. **派生日频表**：
   - ``index_members.parquet``：两个调样生效日之间成分名单保持不变，
     按生效日展开到每个开市日；
   - ``index_weights.parquet``：锚日取官方权重，锚间用成分股的后复权日收益
     漂移（``w_i(t) ∝ w_i(t0) × ∏(1 + r_i)``），每日归一；
   - ``index_weight_drift.parquet``：下一个官方锚到达时，漂移值与官方值的对拍明细。

口径与取舍
----------
- **无前视**：成分切换只发生在快照生效日；权重只从**更早**的锚向前漂移，
  绝不用未来锚回填过去。
- **漂移公式**：``r_i(t) = close×adjfactor`` 的后复权日收益。停牌日无行，
  用前值前向填充得到 ``r = 0``（权重保持不变）；整个区间无行情的票同理。
  每日对全部成分归一，保证权重和恒为 1。
- **锚间成分变化**：漂移段只用该段起点锚的成分名单。下一个锚新增的成分
  在锚日以官方权重直接进入；对拍只比较两段共有的成分，并如实记录。
- 中证官方只在官网提供**最新一期**月末权重与最新成分名单，历史月度权重无公开
  归档（调研结论见 issue #64）。因此本模块提供一个可增量累加的锚库：历史回填
  需要外部离线导出（如 CSMAR 指数成分/权重表），由 :func:`update_index_anchors`
  写入快照与锚文件后 :func:`build_daily_tables` 即可生成完整日频表。
"""
from __future__ import annotations

import bisect
import logging
import warnings
from datetime import date, timedelta
from pathlib import Path
from typing import Protocol

import polars as pl

from quant.data import cache
from quant.data.schema import (
    INDEX_CODES,
    INDEX_DRIFT_CHECK,
    INDEX_MEMBERS,
    INDEX_MEMBER_SNAPSHOTS,
    INDEX_WEIGHTS,
    check_index_members,
    check_index_weights,
    check_schema,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

#: 指数清单的单一事实源在 ``quant.data.schema.INDEX_CODES``（issue #67 落地），
#: 此处仅附指数中文名。
INDEX_NAMES: dict[str, str] = {
    "000300": "沪深300",
    "000905": "中证500",
    "000852": "中证1000",
    "932000": "中证2000",
}

#: 定期调样月份：每年 6 月、12 月第二个星期五的下一交易日生效。
REBALANCE_MONTHS: tuple[int, ...] = (6, 12)
#: 星期五在 ``date.weekday()`` 中的序号（周一 = 0）。
FRIDAY: int = 4

MEMBER_SNAPSHOTS_FILE: str = "index_member_snapshots.parquet"
WEIGHT_ANCHORS_FILE: str = "index_weight_anchors.parquet"
INDEX_MEMBERS_FILE: str = "index_members.parquet"
INDEX_WEIGHTS_FILE: str = "index_weights.parquet"
INDEX_DRIFT_CHECK_FILE: str = "index_weight_drift.parquet"

#: 单只指数同一日权重和的合理区间（权重为小数）。
WEIGHT_SUM_MIN: float = 0.99
WEIGHT_SUM_MAX: float = 1.01
#: 漂移权重与官方权重绝对偏差的中位数告警阈值。
DRIFT_DEVIATION_WARN: float = 1e-4


class AnchorSource(Protocol):
    """指数官方成分 / 权重数据源协议（供离线导出与在线抓取共用）。"""

    def member_snapshot(self, index_code: str) -> pl.DataFrame:
        """返回 ``INDEX_MEMBER_SNAPSHOTS``：``(snapshot_date, instrument, index_code)``。"""
        ...

    def weight_anchor(self, index_code: str) -> pl.DataFrame:
        """返回 ``INDEX_WEIGHTS`` 的锚行：``(date, instrument, index_code, weight)``。"""
        ...


# ---------------------------------------------------------------------------
# 调样日历
# ---------------------------------------------------------------------------


def second_friday(year: int, month: int) -> date:
    """返回 ``year`` 年 ``month`` 月的第二个星期五。"""
    first = date(year, month, 1)
    offset = (FRIDAY - first.weekday()) % 7
    return first + timedelta(days=offset + 7)


def rebalance_effective_dates(
    start: date, end: date, open_days: list[date]
) -> list[date]:
    """定期调样生效日：每年 6/12 月第二个星期五**收市后**生效，即其后的首个开市日。

    ``open_days`` 为升序去重的开市日列表。返回落在 ``[start, end]`` 内的生效日。
    """
    opens = sorted(set(open_days))
    out: list[date] = []
    for year in range(start.year, end.year + 1):
        for month in REBALANCE_MONTHS:
            friday = second_friday(year, month)
            pos = bisect.bisect_right(opens, friday)
            if pos >= len(opens):
                continue
            effective = opens[pos]
            if start <= effective <= end:
                out.append(effective)
    return sorted(set(out))


# ---------------------------------------------------------------------------
# 成分展开
# ---------------------------------------------------------------------------


def expand_member_snapshots(
    snapshots: pl.DataFrame, open_days: pl.DataFrame
) -> pl.DataFrame:
    """把成分快照按生效日展开到每个开市日，输出 ``INDEX_MEMBERS``。

    每个开市日取 ``snapshot_date <= 当日`` 的最近一份快照；早于首份快照的开市日
    不产生行（此时该指数尚未发布，如 932000 在 2023-08 之前）。
    """
    if snapshots.height == 0 or open_days.height == 0:
        return pl.DataFrame(schema=INDEX_MEMBERS)

    snaps = snapshots.select(list(INDEX_MEMBER_SNAPSHOTS.keys())).cast(
        INDEX_MEMBER_SNAPSHOTS
    )
    days = open_days.select("date").unique().sort("date")
    codes = snaps.select("index_code").unique()
    grid = days.join(codes, how="cross")

    timeline = snaps.select("index_code", "snapshot_date").unique().sort(
        ["index_code", "snapshot_date"]
    )
    grid = grid.sort(["index_code", "date"])
    with warnings.catch_warnings():
        # polars 无法在带 by 分组时校验有序性，这里已显式排序，忽略该提示。
        warnings.filterwarnings(
            "ignore", message="Sortedness of columns cannot be checked"
        )
        grid = grid.join_asof(
            timeline,
            left_on="date",
            right_on="snapshot_date",
            by="index_code",
            strategy="backward",
        )
    grid = grid.filter(pl.col("snapshot_date").is_not_null())
    out = (
        grid.join(snaps, on=["index_code", "snapshot_date"], how="inner")
        .select(list(INDEX_MEMBERS.keys()))
        .unique()
        .sort(["index_code", "date", "instrument"])
        .cast(INDEX_MEMBERS)
    )
    check_index_members(out)
    return out


# ---------------------------------------------------------------------------
# 权重漂移
# ---------------------------------------------------------------------------


def _open_days_in(open_days: list[date], start: date, end: date) -> list[date]:
    """``[start, end]`` 内的开市日；``start`` 本身必含（锚日可能在非开市日）。"""
    days = [day for day in open_days if start < day <= end]
    return [start, *days]


def _drift_segment(
    index_code: str,
    anchors: pl.DataFrame,
    bars: pl.DataFrame,
    open_days: list[date],
    start: date,
    next_anchor: date | None,
    last_day: date,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """单个锚区间的漂移，返回（``[start, next_anchor)`` 的日频权重, 锚日对拍明细）。

    ``next_anchor`` 为下一个锚日（最后一个区间为 None）。漂移一直算到
    ``next_anchor``（或 ``last_day``），锚日按官方权重落盘，对拍用漂移到
    ``next_anchor`` 的值与官方锚比较；最后一个区间没有对拍。
    """
    members = anchors.filter(pl.col("date") == start).select("instrument", "weight")
    seg_end = next_anchor if next_anchor is not None else last_day
    days = _open_days_in(open_days, start, seg_end)
    grid = (
        pl.DataFrame({"date": days})
        .join(members.select("instrument"), how="cross")
        .join(
            bars.select("date", "instrument", "close", "adjfactor"),
            on=["date", "instrument"],
            how="left",
        )
        .with_columns((pl.col("close") * pl.col("adjfactor")).alias("_px"))
        .sort(["instrument", "date"])
        .with_columns(
            pl.col("_px").forward_fill().over("instrument", order_by="date").alias("_px")
        )
        .join(members, on="instrument", how="left")
    )
    growth = (
        (
            1.0
            + (
                pl.col("_px") / pl.col("_px").shift(1).over("instrument", order_by="date")
                - 1.0
            )
        )
        .fill_null(1.0)
        .cum_prod()
        .over("instrument", order_by="date")
    )
    grid = grid.with_columns((pl.col("weight") * growth).alias("_w"))
    grid = grid.with_columns(
        (pl.col("_w") / pl.col("_w").sum().over("date")).alias("weight")
    )

    emitted = grid.filter(pl.col("date") < next_anchor) if next_anchor is not None else grid
    weights = emitted.select(
        pl.col("date"),
        pl.col("instrument"),
        pl.lit(index_code).alias("index_code"),
        pl.col("weight"),
    ).cast(INDEX_WEIGHTS)

    checks = pl.DataFrame(schema=INDEX_DRIFT_CHECK)
    if next_anchor is not None:
        predicted = grid.filter(pl.col("date") == next_anchor).select(
            "instrument", pl.col("weight").alias("drift_weight")
        )
        official = anchors.filter(pl.col("date") == next_anchor).select(
            "instrument", pl.col("weight").alias("official_weight")
        )
        joined = predicted.join(official, on="instrument", how="inner").with_columns(
            (pl.col("drift_weight") - pl.col("official_weight")).abs().alias("abs_deviation")
        )
        checks = joined.select(
            pl.lit(next_anchor).alias("anchor_date"),
            pl.lit(index_code).alias("index_code"),
            "instrument",
            "drift_weight",
            "official_weight",
            "abs_deviation",
        ).cast(INDEX_DRIFT_CHECK)
    return weights, checks


def drift_weights(
    anchors: pl.DataFrame,
    bars: pl.DataFrame,
    open_days: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """按官方锚 + 后复权日收益漂移出日频权重。

    - 锚日取官方权重；锚间每个开市日按成分股后复权收益复利并归一。
    - 返回 ``(INDEX_WEIGHTS, INDEX_DRIFT_CHECK)``：前者为日频权重（锚日官方、
      锚间漂移），后者为每个锚日的漂移-官方对拍明细。
    - 最后一个锚向后的区间一直漂移到 ``open_days`` 的最大开市日。
    """
    if anchors.height == 0 or open_days.height == 0:
        return (
            pl.DataFrame(schema=INDEX_WEIGHTS),
            pl.DataFrame(schema=INDEX_DRIFT_CHECK),
        )

    anchors = anchors.select(list(INDEX_WEIGHTS.keys())).cast(INDEX_WEIGHTS)
    days = sorted(set(open_days["date"].to_list()))
    weight_frames: list[pl.DataFrame] = []
    check_frames: list[pl.DataFrame] = []

    relevant = anchors.select("instrument").unique()
    used_bars = bars.join(relevant, on="instrument", how="semi")

    for index_code in anchors["index_code"].unique().sort():
        sub = anchors.filter(pl.col("index_code") == index_code).sort("date")
        anchor_dates = sorted(set(sub["date"].to_list()))
        for pos, start in enumerate(anchor_dates):
            next_anchor = (
                anchor_dates[pos + 1] if pos + 1 < len(anchor_dates) else None
            )
            last_day = days[-1] if days else start
            if (next_anchor or last_day) <= start:
                continue
            weights, checks = _drift_segment(
                index_code, sub, used_bars, days, start, next_anchor, last_day
            )
            weight_frames.append(weights)
            check_frames.append(checks)

    weights = (
        pl.concat(weight_frames, how="vertical_relaxed")
        if weight_frames
        else pl.DataFrame(schema=INDEX_WEIGHTS)
    )
    checks = (
        pl.concat(check_frames, how="vertical_relaxed")
        if check_frames
        else pl.DataFrame(schema=INDEX_DRIFT_CHECK)
    )
    weights = (
        weights.unique(subset=["date", "instrument", "index_code"], keep="last")
        .sort(["index_code", "date", "instrument"])
        .cast(INDEX_WEIGHTS)
    )
    checks = checks.sort(["index_code", "anchor_date", "instrument"]).cast(INDEX_DRIFT_CHECK)
    check_index_weights(weights)
    return weights, checks


# ---------------------------------------------------------------------------
# 落盘 / 读取
# ---------------------------------------------------------------------------


def _snapshots_path(data_dir: Path) -> Path:
    return Path(data_dir) / MEMBER_SNAPSHOTS_FILE


def _weight_anchors_path(data_dir: Path) -> Path:
    return Path(data_dir) / WEIGHT_ANCHORS_FILE


def _derived_weights_path(data_dir: Path) -> Path:
    return Path(data_dir) / INDEX_WEIGHTS_FILE


def _members_path(data_dir: Path) -> Path:
    return Path(data_dir) / INDEX_MEMBERS_FILE


def _drift_path(data_dir: Path) -> Path:
    return Path(data_dir) / INDEX_DRIFT_CHECK_FILE


def read_snapshots(data_dir: Path) -> pl.DataFrame:
    """读取成分快照，缺失返回空表。"""
    path = _snapshots_path(data_dir)
    if not path.exists():
        return pl.DataFrame(schema=INDEX_MEMBER_SNAPSHOTS)
    return (
        pl.read_parquet(path)
        .select(list(INDEX_MEMBER_SNAPSHOTS.keys()))
        .cast(INDEX_MEMBER_SNAPSHOTS)
        .sort(["index_code", "snapshot_date", "instrument"])
    )


def read_anchor_weights(data_dir: Path) -> pl.DataFrame:
    """读取官方权重锚（``index_weight_anchors.parquet``），缺失返回空表。"""
    path = _weight_anchors_path(data_dir)
    if not path.exists():
        return pl.DataFrame(schema=INDEX_WEIGHTS)
    return (
        pl.read_parquet(path)
        .select(list(INDEX_WEIGHTS.keys()))
        .cast(INDEX_WEIGHTS)
    )


def read_index_members(data_dir: Path) -> pl.DataFrame:
    """读取日频成分表，缺失返回空表。"""
    path = _members_path(data_dir)
    if not path.exists():
        return pl.DataFrame(schema=INDEX_MEMBERS)
    return (
        pl.read_parquet(path)
        .select(list(INDEX_MEMBERS.keys()))
        .cast(INDEX_MEMBERS)
        .sort(["index_code", "date", "instrument"])
    )


def read_index_weights(data_dir: Path) -> pl.DataFrame:
    """读取日频 PIT 权重表（``index_weights.parquet``），缺失返回空表。"""
    path = _derived_weights_path(data_dir)
    if not path.exists():
        return pl.DataFrame(schema=INDEX_WEIGHTS)
    return (
        pl.read_parquet(path)
        .select(list(INDEX_WEIGHTS.keys()))
        .cast(INDEX_WEIGHTS)
        .sort(["index_code", "date", "instrument"])
    )


def read_drift_check(data_dir: Path) -> pl.DataFrame:
    """读取漂移对拍明细，缺失返回空表。"""
    path = _drift_path(data_dir)
    if not path.exists():
        return pl.DataFrame(schema=INDEX_DRIFT_CHECK)
    return (
        pl.read_parquet(path)
        .select(list(INDEX_DRIFT_CHECK.keys()))
        .cast(INDEX_DRIFT_CHECK)
    )


def _merge_snapshots(existing: pl.DataFrame, incoming: pl.DataFrame) -> pl.DataFrame:
    if incoming.height == 0:
        return existing
    incoming = incoming.select(list(INDEX_MEMBER_SNAPSHOTS.keys())).cast(
        INDEX_MEMBER_SNAPSHOTS
    )
    # 同一 (index_code, snapshot_date) 的名单整份覆盖，避免新旧名单混在一起。
    keys = incoming.select("index_code", "snapshot_date").unique()
    kept = (
        existing.join(keys, on=["index_code", "snapshot_date"], how="anti")
        if existing.height
        else existing
    )
    frames = [frame for frame in (kept, incoming) if frame.height]
    out = (
        pl.concat(frames, how="vertical_relaxed")
        .select(list(INDEX_MEMBER_SNAPSHOTS.keys()))
        .cast(INDEX_MEMBER_SNAPSHOTS)
        .unique(subset=["snapshot_date", "instrument", "index_code"], keep="last")
        .sort(["index_code", "snapshot_date", "instrument"])
    )
    check_schema(out, INDEX_MEMBER_SNAPSHOTS, name="index_member_snapshots")
    return out


def _merge_anchors(existing: pl.DataFrame, incoming: pl.DataFrame) -> pl.DataFrame:
    if incoming.height == 0:
        return existing
    incoming = incoming.select(list(INDEX_WEIGHTS.keys())).cast(INDEX_WEIGHTS)
    # 同一 (index_code, date) 的权重整份覆盖。
    keys = incoming.select("index_code", "date").unique()
    kept = (
        existing.join(keys, on=["index_code", "date"], how="anti")
        if existing.height
        else existing
    )
    frames = [frame for frame in (kept, incoming) if frame.height]
    out = (
        pl.concat(frames, how="vertical_relaxed")
        .select(list(INDEX_WEIGHTS.keys()))
        .cast(INDEX_WEIGHTS)
        .unique(subset=["date", "instrument", "index_code"], keep="last")
        .sort(["index_code", "date", "instrument"])
    )
    check_index_weights(out, name="index_weights_anchors")
    return out


def build_daily_tables(data_dir: Path, *, end: date | None = None) -> dict[str, int]:
    """从快照与锚重建日频成分表、日频权重表与对拍明细，返回各表行数。"""
    data_dir = Path(data_dir)
    snapshots = read_snapshots(data_dir)
    anchors = read_anchor_weights(data_dir)

    calendar = cache.load_calendar(data_dir, end=end)
    open_days = calendar.filter(pl.col("is_open")).select("date")
    if end is not None:
        open_days = open_days.filter(pl.col("date") <= end)

    members = expand_member_snapshots(snapshots, open_days)

    if anchors.height == 0 or open_days.height == 0:
        weights = pl.DataFrame(schema=INDEX_WEIGHTS)
        checks = pl.DataFrame(schema=INDEX_DRIFT_CHECK)
    else:
        start = anchors["date"].min()
        instruments = anchors["instrument"].unique().to_list()
        bars = cache.load_bars(data_dir, instruments=instruments, start=start, end=end)
        weights, checks = drift_weights(anchors, bars, open_days)

    cache._atomic_write_parquet(_members_path(data_dir), members)
    cache._atomic_write_parquet(_derived_weights_path(data_dir), weights)
    cache._atomic_write_parquet(_drift_path(data_dir), checks)
    return {
        "members": members.height,
        "weights": weights.height,
        "drift_check": checks.height,
        "snapshots": snapshots.height,
        "anchors": anchors.height,
    }


# ---------------------------------------------------------------------------
# 增量落地
# ---------------------------------------------------------------------------


def _member_set(snapshot: pl.DataFrame) -> set[str]:
    return set(snapshot["instrument"].to_list())


def update_index_anchors(
    source: AnchorSource,
    data_dir: Path,
    *,
    index_codes: tuple[str, ...] = INDEX_CODES,
) -> dict[str, int]:
    """抓取官方最新成分名单与月末权重，变更时增量写入快照与锚文件。

    - 成分名单与已存最新快照完全一致时不落盘（避免每日重复写同一份名单）；
      发生变化时以抓取到的 ``snapshot_date`` 追加一份新快照。
    - 权重锚按 ``(index_code, date)`` 覆盖写入，幂等。

    返回 ``{index_code: 是否更新}`` 计数的汇总字典。
    """
    data_dir = Path(data_dir)
    snapshots = read_snapshots(data_dir)
    anchors = read_anchor_weights(data_dir)
    updated = 0
    checked = 0

    for index_code in index_codes:
        checked += 1
        try:
            snapshot = source.member_snapshot(index_code)
        except Exception as exc:  # noqa: BLE001 - 单指数失败不影响其余指数
            logger.warning("抓取 %s 成分名单失败：%s", index_code, exc)
            snapshot = pl.DataFrame(schema=INDEX_MEMBER_SNAPSHOTS)
        if snapshot.height:
            previous = snapshots.filter(pl.col("index_code") == index_code)
            same = False
            if previous.height:
                latest_date = previous["snapshot_date"].max()
                latest = previous.filter(pl.col("snapshot_date") == latest_date)
                same = _member_set(latest) == _member_set(snapshot)
            if not same:
                snapshots = _merge_snapshots(snapshots, snapshot)
                updated += 1

        try:
            anchor = source.weight_anchor(index_code)
        except Exception as exc:  # noqa: BLE001
            logger.warning("抓取 %s 权重失败：%s", index_code, exc)
            anchor = pl.DataFrame(schema=INDEX_WEIGHTS)
        if anchor.height:
            anchors = _merge_anchors(anchors, anchor)

    cache._atomic_write_parquet(_snapshots_path(data_dir), snapshots)
    cache._atomic_write_parquet(_weight_anchors_path(data_dir), anchors)
    return {"checked": checked, "updated": updated}


def run_index_update(
    source: AnchorSource,
    data_dir: Path,
    *,
    end: date | None = None,
    index_codes: tuple[str, ...] = INDEX_CODES,
) -> dict[str, int]:
    """抓取官方锚并重建日频表，返回 :func:`build_daily_tables` 行数 + 抓取计数。"""
    data_dir = Path(data_dir)
    stats = update_index_anchors(source, data_dir, index_codes=index_codes)
    counts = build_daily_tables(data_dir, end=end)
    counts.update(stats)
    return counts


__all__ = [
    "DRIFT_DEVIATION_WARN",
    "INDEX_CODES",
    "INDEX_NAMES",
    "REBALANCE_MONTHS",
    "WEIGHT_SUM_MAX",
    "WEIGHT_SUM_MIN",
    "AnchorSource",
    "build_daily_tables",
    "drift_weights",
    "expand_member_snapshots",
    "read_anchor_weights",
    "read_drift_check",
    "read_index_members",
    "read_index_weights",
    "read_snapshots",
    "rebalance_effective_dates",
    "run_index_update",
    "second_friday",
    "update_index_anchors",
]
