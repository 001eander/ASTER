"""股票池的 point-in-time 解析（issue #66）。

训练、跑批、回测、因子评估四处都要按股票池取数，本模块提供统一的股票池抽象：

- **命名池**：``hs300`` / ``zz500`` / ``zz1000`` / ``zz2000``，从
  ``index_members.parquet`` 按 ``index_code + date`` 解析。该表由
  ``quant.data.index_members`` 展开落地，成分在调样生效日之间保持不变。
- **自定义池**：传入 parquet / csv 路径。
  - 静态池（只有 ``instrument`` 一列）：任意日返回同一集合；
  - 动态池（``date`` / ``instrument`` 两列）：按 ``day`` 精确匹配当日成员（PIT），
    该日无记录返回空集。

口径与取舍
----------
- **无前视**：``members`` 只返回 ``day`` 当日已生效的成分，调样按生效日切换，
  不用未来名单回填过去。
- **区间展开**：``members_range`` 输出 ``(date, instrument)`` 两列，按
  ``(date, instrument)`` 排序去重。静态池没有自身的时间轴，展开时用
  ``data_dir/calendar.parquet`` 的开市日作为交易日轴。
- **空池**：命名池在 ``index_members.parquet`` 里某日无记录（如 932000 在
  2023-08 之前尚未发布）时返回空集，不报错。
- 自定义池的 ``instrument`` 统一过 :func:`quant.data.schema.normalize_instrument`
  归一化，命名池数据落盘时已是规范格式。
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import cast

import polars as pl

from quant.data import cache
from quant.data.index_members import INDEX_MEMBERS_FILE, read_index_members
from quant.data.schema import normalize_instrument

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

#: 命名池注册表：池名 -> 指数代码。指数代码的单一事实源在
#: ``quant.data.schema.INDEX_CODES``，此处只做别名映射。
NAMED_UNIVERSES: dict[str, str] = {
    "hs300": "000300",
    "zz500": "000905",
    "zz1000": "000852",
    "zz2000": "932000",
}

#: 自定义池支持的文件后缀。
SUPPORTED_SUFFIXES: tuple[str, ...] = (".parquet", ".csv")

#: 自定义池的必需列与可选日期列。
INSTRUMENT_COLUMN: str = "instrument"
DATE_COLUMN: str = "date"

#: ``members_range`` 的输出 schema。
OUTPUT_SCHEMA = pl.Schema({"date": pl.Date, "instrument": pl.String})


class UniverseError(ValueError):
    """股票池无法解析（未知池名、文件缺失或格式不符）。"""


# ---------------------------------------------------------------------------
# 目标解析
# ---------------------------------------------------------------------------


def _available_names() -> str:
    return ", ".join(sorted(NAMED_UNIVERSES))


def _resolve_target(name_or_path: str) -> tuple[str, str | Path]:
    """判定输入是命名池还是自定义池文件。

    名字命中注册表即为命名池；否则按路径处理（存在且后缀为 ``.parquet`` /
    ``.csv``）；都不满足抛 :class:`UniverseError` 并列出可用池名。
    """
    if name_or_path in NAMED_UNIVERSES:
        return "named", NAMED_UNIVERSES[name_or_path]
    path = Path(name_or_path)
    if path.suffix.lower() in SUPPORTED_SUFFIXES and path.is_file():
        return "custom", path
    raise UniverseError(
        f"未知股票池 {name_or_path!r}：既不在命名池注册表（可用：{_available_names()}），"
        f"也不是存在的 {'/'.join(SUPPORTED_SUFFIXES)} 文件"
    )


def _load_custom(path: Path) -> tuple[pl.DataFrame, bool]:
    """读取自定义池，返回 ``(归一化后的表, 是否动态池)``。

    静态池只有 ``instrument`` 列；动态池含 ``date`` 列。``instrument`` 过
    :func:`normalize_instrument`，``date`` 统一为 :class:`datetime.date`。
    """
    suffix = path.suffix.lower()
    try:
        raw = (
            pl.read_parquet(path)
            if suffix == ".parquet"
            else pl.read_csv(path, infer_schema_length=0)
        )
    except (pl.exceptions.PolarsError, OSError) as exc:
        raise UniverseError(f"读取自定义池 {path} 失败：{exc}") from exc

    if INSTRUMENT_COLUMN not in raw.columns:
        raise UniverseError(
            f"自定义池 {path} 缺少 {INSTRUMENT_COLUMN!r} 列，实际列：{raw.columns}"
        )

    dynamic = DATE_COLUMN in raw.columns
    columns = [DATE_COLUMN, INSTRUMENT_COLUMN] if dynamic else [INSTRUMENT_COLUMN]
    frame = raw.select(columns)
    if dynamic and frame.schema[DATE_COLUMN] != pl.Date:
        frame = frame.with_columns(
            pl.col(DATE_COLUMN).cast(pl.String).str.to_date()
        )
    frame = frame.with_columns(
        pl.col(INSTRUMENT_COLUMN).map_elements(
            normalize_instrument, return_dtype=pl.String
        )
    )
    return frame, dynamic


def _open_days(data_dir: Path, start: date, end: date) -> pl.DataFrame:
    """``[start, end]`` 内的开市日，静态池区间展开的交易日轴。"""
    try:
        calendar = cache.load_calendar(Path(data_dir), start=start, end=end)
    except FileNotFoundError as exc:
        raise UniverseError(
            f"静态池的区间展开需要 {Path(data_dir) / cache.CALENDAR_FILE} "
            "提供交易日轴，该文件不存在"
        ) from exc
    return calendar.filter(pl.col("is_open")).select(DATE_COLUMN)


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------


def members(
    name_or_path: str, day: date, *, data_dir: Path = Path("data")
) -> set[str]:
    """返回 ``day`` 当日生效的成分集合。

    ``name_or_path`` 为注册表里的池名或自定义池文件路径。命名池某日在
    ``index_members.parquet`` 无记录、动态池该日无记录时返回空集。
    """
    kind, target = _resolve_target(name_or_path)
    if kind == "named":
        frame = read_index_members(Path(data_dir)).filter(
            (pl.col("index_code") == target) & (pl.col(DATE_COLUMN) == day)
        )
        return set(frame[INSTRUMENT_COLUMN].to_list())
    frame, dynamic = _load_custom(cast(Path, target))
    if dynamic:
        frame = frame.filter(pl.col(DATE_COLUMN) == day)
    return set(frame[INSTRUMENT_COLUMN].to_list())


def members_range(
    name_or_path: str,
    start: date,
    end: date,
    *,
    data_dir: Path = Path("data"),
) -> pl.DataFrame:
    """返回 ``[start, end]`` 区间的成分展开，``(date, instrument)`` 两列。

    输出按 ``(date, instrument)`` 排序并去重。命名池与动态池只输出表内有记录的
    日期；静态池用 ``data_dir/calendar.parquet`` 的开市日展开。
    """
    kind, target = _resolve_target(name_or_path)
    if kind == "named":
        frame = read_index_members(Path(data_dir)).filter(
            (pl.col("index_code") == target)
            & (pl.col(DATE_COLUMN) >= start)
            & (pl.col(DATE_COLUMN) <= end)
        ).select(DATE_COLUMN, INSTRUMENT_COLUMN)
    else:
        path = cast(Path, target)
        custom, dynamic = _load_custom(path)
        if dynamic:
            frame = custom.filter(
                (pl.col(DATE_COLUMN) >= start) & (pl.col(DATE_COLUMN) <= end)
            ).select(DATE_COLUMN, INSTRUMENT_COLUMN)
        else:
            days = _open_days(data_dir, start, end)
            frame = days.join(
                custom.select(INSTRUMENT_COLUMN), how="cross"
            ).select(DATE_COLUMN, INSTRUMENT_COLUMN)
    return frame.unique().sort([DATE_COLUMN, INSTRUMENT_COLUMN]).cast(OUTPUT_SCHEMA)


__all__ = [
    "INDEX_MEMBERS_FILE",
    "NAMED_UNIVERSES",
    "OUTPUT_SCHEMA",
    "SUPPORTED_SUFFIXES",
    "UniverseError",
    "members",
    "members_range",
]
