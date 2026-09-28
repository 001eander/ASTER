"""open-to-open 标签构造：全项目唯一的标签口径。

量化口径
--------
信号 cutoff 是 T 日收盘，成交发生在 T+1 开盘，标签衡量 T+1 开盘到之后的收益。
``horizon=h`` 时::

    label = adj_open(T+1+h) / adj_open(T+1) - 1

其中 ``adj_open = open × adjfactor``（后复权价），``T`` 为信号日（输入的 ``date``），
``T+1`` / ``T+1+h`` 是**同一只证券自身行情序列**里的下一个 / 下 h 个交易行。
``horizon=1`` 即 T+1 开盘 → T+2 开盘。

为什么必须是 T+1 开盘 → T+2 开盘
--------------------------------
若用「T 收盘 → T+1 收盘」会引入前视：T 日收盘后才知道要用哪个信号，却把 T 日
收盘到 T+1 收盘的收益算给该信号，系统性高估 IC。T+1 开盘成交把可实现的建仓价
对齐到信号可得之后，是本项目不可让步的红线。

为什么用后复权价
----------------
除权除息日未复权价格会向下跳空（送转股本、派息），直接用 ``open`` 算收益会在
除权日产生与真实持仓无关的假收益。``adj_open = open × adjfactor`` 把价格还原到
连续口径，标签对除权不敏感。``adjfactor`` 为后复权因子，对历史日期不随新的公司
行为变化。

停牌口径（自身序列 shift + ``delay_days`` 暴露）
------------------------------------------------
「下一交易日」按每只证券自身有行情的记录序列（``shift``）定义，不依赖交易日历。
T+1 停牌时该票当天本就不会成交，标签顺延到它下一个实际有行情的交易日。两种口径
的取舍：

- **交易日历口径**：T+1 停牌时标签为 null，只保留前后都能成交的样本。干净，但需要
  外部日历，且会把停牌前后大量本可实现的标签一并丢掉。
- **自身序列口径（本模块采用）**：标签照常给出，停牌造成的跳跃由 ``delay_days``
  列暴露。默认无停牌，是否过滤交给下游按自己的容忍度决定，避免在标签层过早丢样本。

``delay_days`` 是**信号日到实际建仓日（T+1）之间的日历天数**，无停牌的连续交易日
通常为 1，跨周末为 3，停牌则更大。它只反映建仓端的延迟；持有端（T+1 → T+1+h）的
跳跃已由 ``horizon`` 的交易日语义隐含，不再单独设列。

尾部与空值
----------
序列尾部没有未来行情时 ``label`` 为 null，**不丢行**，由下游决定如何使用。
"""
from __future__ import annotations

import polars as pl

from quant.data.schema import DAILY_BARS

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

#: 默认 horizon：T+1 开盘 → T+2 开盘。
DEFAULT_HORIZON: int = 1

#: 计算标签所需的最小列集合。
REQUIRED_COLUMNS: tuple[str, ...] = ("date", "instrument", "open", "adjfactor")

#: 参数列与中间列名。
_ADJ_OPEN = "_adj_open"
_ENTRY = "_entry"
_EXIT = "_exit"
_ENTRY_DATE = "_entry_date"

_LABEL_SCHEMA = pl.Schema(
    {
        "date": pl.Date,
        "instrument": pl.String,
        "label": pl.Float64,
        "delay_days": pl.Int64,
    }
)


def open_to_open_label(
    bars: pl.DataFrame, horizon: int = DEFAULT_HORIZON
) -> pl.DataFrame:
    """构造 open-to-open 标签长表。

    输入为 ``DAILY_BARS``（至少含 ``date`` / ``instrument`` / ``open`` /
    ``adjfactor`` 列，额外列会被忽略），输出 ``(date, instrument, label,
    delay_days)`` 四列，按 ``(instrument, date)`` 排序。

    - ``label = adj_open(T+1+h) / adj_open(T+1) - 1``，``adj_open = open × adjfactor``。
    - ``delay_days``：信号日到实际建仓日 T+1 的日历天数（见模块 docstring 的停牌口径）。
    - 尾部无未来行情或 ``horizon < 1`` 的非法入参见下。

    ``horizon`` 必须 >= 1，否则抛 :class:`ValueError`；输入缺列同样抛
    :class:`ValueError`。函数内部先按 ``(instrument, date)`` 排序，输入顺序无所谓。
    """
    if not isinstance(horizon, int) or isinstance(horizon, bool):
        raise ValueError(f"horizon 必须为整数，收到 {horizon!r}")
    if horizon < 1:
        raise ValueError(f"horizon 必须 >= 1，收到 {horizon}")
    missing = [col for col in REQUIRED_COLUMNS if col not in bars.columns]
    if missing:
        raise ValueError(f"bars 缺少列: {missing}")

    ordered = bars.select(*REQUIRED_COLUMNS).sort(["instrument", "date"])
    if ordered.height == 0:
        return ordered.select(
            "date",
            "instrument",
            pl.lit(None, dtype=pl.Float64).alias("label"),
            pl.lit(None, dtype=pl.Int64).alias("delay_days"),
        ).cast(_LABEL_SCHEMA)

    shifted = ordered.with_columns(
        (pl.col("open") * pl.col("adjfactor")).alias(_ADJ_OPEN)
    ).with_columns(
        pl.col(_ADJ_OPEN)
        .shift(-1)
        .over("instrument", order_by="date")
        .alias(_ENTRY),
        pl.col(_ADJ_OPEN)
        .shift(-1 - horizon)
        .over("instrument", order_by="date")
        .alias(_EXIT),
        pl.col("date")
        .shift(-1)
        .over("instrument", order_by="date")
        .alias(_ENTRY_DATE),
    )

    labelled = shifted.with_columns(
        pl.when((pl.col(_ENTRY) > 0) & (pl.col(_EXIT).is_not_null()))
        .then(pl.col(_EXIT) / pl.col(_ENTRY) - 1.0)
        .otherwise(None)
        .alias("label"),
        (pl.col(_ENTRY_DATE) - pl.col("date")).dt.total_days().alias("delay_days"),
    )
    return labelled.select(
        "date", "instrument", "label", "delay_days"
    ).cast(_LABEL_SCHEMA)


def attach_label(
    bars: pl.DataFrame, horizon: int = DEFAULT_HORIZON
) -> pl.DataFrame:
    """把 ``label`` 与 ``delay_days`` 附加到 ``DAILY_BARS`` 上，供训练管线直接用。

    保留输入的全部列，按 ``(date, instrument)`` 左连接标签，输出按
    ``(instrument, date)`` 排序。尾部无未来行情的记录 ``label`` 为 null。
    """
    labels = open_to_open_label(bars, horizon=horizon)
    return (
        bars.join(labels, on=["date", "instrument"], how="left")
        .sort(["instrument", "date"])
    )


__all__ = [
    "DEFAULT_HORIZON",
    "REQUIRED_COLUMNS",
    "attach_label",
    "open_to_open_label",
]
