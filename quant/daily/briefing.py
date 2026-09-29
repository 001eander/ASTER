"""每日简报：把一次跑批的 :class:`~quant.daily.pipeline.DailyReport` 渲染成 markdown。

用途
----
人工跟单的参考材料，回答三个问题：

1. 今天要买卖什么（调仓明细，含被预过滤的订单与原因）；
2. 组合现在长什么样（个股 / 行业敞口，指增策略可选叠加风险四表）；
3. 模型依赖的因子最近还好吗（近端 RankIC）。

板块
----
- 头部：信号日、账户、策略 / 池 / 基准、净值与现金 / 持仓敞口、换手；
- 调仓明细：可执行订单与被预过滤订单（原因中文化）；
- 个股敞口：当前持仓，以及目标权重相对当前权重的 top 偏离；
- 行业 / 风险敞口：行业层面持仓分布；指增策略且拿得到基准权重时，可选叠加
  :mod:`quant.eval.risk_report` 的四表（见 :func:`build_briefing_risk_report`，
  数据不足时返回 ``None``，简报里只留扩展点，不硬凑）；
- 因子近期表现：因子库每个因子最近 ``window_days`` 个交易日的 RankIC 汇总。

无前视
------
因子近期表现的标签复用 :func:`quant.labels.open_to_open.open_to_open_label`，信号日 T
衡量 T+1 开盘 → T+2 开盘收益，与训练口径一致。行情与因子的时间范围由调用方保证。
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from quant.automl.dataset import FactorCompute
from quant.daily.strategy import STRATEGY_INDEX_ENHANCED
from quant.data.index_members import read_index_members
from quant.eval.metrics import (
    DATE_COL,
    FACTOR_COL,
    INSTRUMENT_COL,
    LABEL_COL,
    MIN_CROSS_SECTION_COUNT,
    ICSummary,
    ic_series,
    summarize_ic,
)
from quant.eval.risk_report import RiskReport, build_risk_report, frame_to_markdown
from quant.labels.open_to_open import open_to_open_label
from quant.portfolio.enhanced_inputs import (
    benchmark_weights,
    float_mv_frame,
    industry_table,
    style_frame,
)

if TYPE_CHECKING:  # 避免与 quant.daily.pipeline 形成导入环
    from quant.daily.pipeline import DailyReport

# ---------------------------------------------------------------------------
# 配置（默认值集中在此）
# ---------------------------------------------------------------------------

#: 持仓权重列名（与虚拟账户持仓摘要一致）。
WEIGHT_COL: str = "weight"

#: 因子 ``compute`` 输出列名。
VALUE_COL: str = "value"

#: 因子近期表现默认回看的交易日数。
DEFAULT_IC_WINDOW: int = 20

#: 简报默认落盘目录（``runs/`` 不入 git）。
DEFAULT_BRIEFING_DIR: Path = Path("runs/reports")

#: 简报文件名后缀（与同目录 JSON 报告区分）。
BRIEFING_SUFFIX: str = ".md"

#: 个股敞口里列出的最大偏离条数。
TOP_DEVIATIONS: int = 15

#: 板块代码 → 中文名（``board_of`` 的输出）。
BOARD_LABELS: dict[str, str] = {
    "main": "主板",
    "cyb": "创业板",
    "kcb": "科创板",
    "bj": "北交所",
}

#: 交易方向 → 中文名。
SIDE_LABELS: dict[str, str] = {"buy": "买入", "sell": "卖出"}

#: 预过滤原因 → 中文名。
REASON_LABELS: dict[str, str] = {
    "suspended": "停牌",
    "limit_up": "涨停",
    "limit_down": "跌停",
}

#: 行业缺失占位符。
UNKNOWN_INDUSTRY: str = "未知"

#: 判定权重偏离为 0 的数值容差。
DEVIATION_EPSILON: float = 1e-9


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FactorIC:
    """单个因子近端 RankIC 的汇总。

    Attributes
    ----------
    name:
        因子名（因子库文件名）。
    window_days:
        回看的交易日数。
    n_days:
        有效 IC 天数（因子与标签都非空、截面证券数达阈值的天数）。
    mean_ic / std_ic / icir / win_rate:
        与 :class:`~quant.eval.metrics.ICSummary` 同口径，样本不足时为 ``None``。
    """

    name: str
    window_days: int
    n_days: int
    mean_ic: float | None = None
    std_ic: float | None = None
    icir: float | None = None
    win_rate: float | None = None

    @property
    def direction(self) -> str:
        """按近端 RankIC 均值给出的方向：正向 / 反向 / 中性 / 无。"""
        if self.mean_ic is None:
            return "无"
        if self.mean_ic > 0.0:
            return "正向"
        if self.mean_ic < 0.0:
            return "反向"
        return "中性"


# ---------------------------------------------------------------------------
# 因子近期表现
# ---------------------------------------------------------------------------


def _factor_frame(name: str, compute: FactorCompute, bars: pl.DataFrame) -> pl.DataFrame:
    """执行单个因子，规整为 ``(date, instrument, factor)``。"""
    result = compute(bars)
    missing = [
        col for col in (DATE_COL, INSTRUMENT_COL, VALUE_COL) if col not in result.columns
    ]
    if missing:
        raise ValueError(f"因子 {name!r} 的输出缺少列：{missing}")
    return result.select(
        DATE_COL,
        INSTRUMENT_COL,
        pl.col(VALUE_COL).cast(pl.Float64).alias(FACTOR_COL),
    )


def recent_factor_ic(
    bars: pl.DataFrame,
    factors: Mapping[str, FactorCompute],
    end_date: date,
    window_days: int = DEFAULT_IC_WINDOW,
) -> list[FactorIC]:
    """逐因子统计最近 ``window_days`` 个交易日的 RankIC。

    行情窗口取 ``bars`` 中 ``<= end_date`` 的最近 ``window_days`` 个交易日；标签用
    :func:`quant.labels.open_to_open.open_to_open_label`（T+1 开盘 → T+2 开盘），
    指标复用 :func:`quant.eval.metrics.ic_series` 与
    :func:`quant.eval.metrics.summarize_ic`。因子与标签都缺失的日期不参与汇总。

    返回按近端 RankIC 均值降序（``None`` 排最后）的 :class:`FactorIC` 列表，供简报
    判断「最近哪些因子在失效」。
    """
    if window_days < 1:
        raise ValueError(f"window_days 必须 >= 1，收到 {window_days}")

    window = (
        bars.filter(pl.col(DATE_COL) <= end_date)
        .select(DATE_COL)
        .unique()
        .sort(DATE_COL)
        .tail(window_days)
    )
    labels = open_to_open_label(bars).select(DATE_COL, INSTRUMENT_COL, LABEL_COL)

    results: list[FactorIC] = []
    for name, compute in factors.items():
        frame = (
            _factor_frame(name, compute, bars)
            .join(window, on=DATE_COL, how="semi")
            .join(labels, on=[DATE_COL, INSTRUMENT_COL], how="left")
            .drop_nulls([FACTOR_COL, LABEL_COL])
        )
        series = ic_series(
            frame.select(DATE_COL, INSTRUMENT_COL, FACTOR_COL, LABEL_COL),
            method="spearman",
            min_count=MIN_CROSS_SECTION_COUNT,
        )
        summary: ICSummary = summarize_ic(series)
        results.append(
            FactorIC(
                name=name,
                window_days=window_days,
                n_days=summary.n_days,
                mean_ic=summary.mean,
                std_ic=summary.std,
                icir=summary.icir,
                win_rate=summary.ic_win_rate,
            )
        )

    return sorted(
        results,
        key=lambda item: (
            item.mean_ic is None,
            -(item.mean_ic if item.mean_ic is not None else 0.0),
        ),
    )


# ---------------------------------------------------------------------------
# 敞口
# ---------------------------------------------------------------------------


def industry_exposure(
    holdings: pl.DataFrame,
    industry: Mapping[str, str] | None,
) -> pl.DataFrame | None:
    """按行业汇总当前持仓权重，返回 ``(industry_l1, weight, n_positions)``。

    持仓为空或缺 ``weight`` 列、行业映射为空时返回 ``None``。行业缺失的证券归
    :data:`UNKNOWN_INDUSTRY`，结果按权重降序。
    """
    if not industry:
        return None
    if holdings.height == 0 or WEIGHT_COL not in holdings.columns:
        return None

    mapping = pl.DataFrame(
        {
            INSTRUMENT_COL: list(industry.keys()),
            "industry_l1": list(industry.values()),
        },
        schema={INSTRUMENT_COL: pl.String, "industry_l1": pl.String},
    )
    return (
        holdings.select(INSTRUMENT_COL, WEIGHT_COL)
        .join(mapping, on=INSTRUMENT_COL, how="left")
        .with_columns(pl.col("industry_l1").fill_null(UNKNOWN_INDUSTRY))
        .group_by("industry_l1")
        .agg(
            pl.col(WEIGHT_COL).sum().alias(WEIGHT_COL),
            pl.len().cast(pl.Int64).alias("n_positions"),
        )
        .sort(WEIGHT_COL, descending=True)
    )


def _top_deviations(
    target: Mapping[str, float],
    current: Mapping[str, float],
    top: int,
) -> list[tuple[str, float, float, float]]:
    """目标权重与当前权重的偏离 ``(instrument, 当前, 目标, 偏离)``，按 |偏离| 降序。"""
    if not target and not current:
        return []
    schema = {INSTRUMENT_COL: pl.String, "value": pl.Float64}
    target_frame = pl.DataFrame(
        {INSTRUMENT_COL: list(target), "value": list(target.values())}, schema=schema
    ).rename({"value": "target"})
    current_frame = pl.DataFrame(
        {INSTRUMENT_COL: list(current), "value": list(current.values())}, schema=schema
    ).rename({"value": "current"})
    joined = (
        target_frame.join(current_frame, on=INSTRUMENT_COL, how="full", coalesce=True)
        .with_columns(pl.col("target").fill_null(0.0), pl.col("current").fill_null(0.0))
        .with_columns((pl.col("target") - pl.col("current")).alias("deviation"))
        .filter(pl.col("deviation").abs() > DEVIATION_EPSILON)
        .sort(pl.col("deviation").abs(), descending=True)
        .head(top)
    )
    return [
        (
            str(row[INSTRUMENT_COL]),
            float(row["current"]),
            float(row["target"]),
            float(row["deviation"]),
        )
        for row in joined.iter_rows(named=True)
    ]


def _industry_frame_for_report(
    data_dir: str | Path,
    report: DailyReport,
    benchmark: Mapping[str, float] | None = None,
) -> pl.DataFrame:
    """为风险四表取行业截面；持仓与基准成分的行业都覆盖到。"""
    instruments = (
        report.holdings[INSTRUMENT_COL].to_list() if report.holdings.height else []
    )
    covered = sorted(set(instruments) | set(benchmark or {}))
    frame, _ = industry_table(data_dir, report.date, covered)
    return frame


def build_briefing_risk_report(
    data_dir: str | Path,
    bars: pl.DataFrame,
    report: DailyReport,
    *,
    benchmark: str | None = None,
    industry: pl.DataFrame | None = None,
) -> RiskReport | None:
    """指增简报可选接入 :mod:`quant.eval.risk_report` 的四表。

    仅当报告为 ``index_enhanced``、给出基准代码、且账户当前有持仓时构造；
    基准权重、持仓或行业数据不足时返回 ``None``，由调用方在简报里说明，不硬凑。
    指数分布用 :func:`quant.data.index_members.read_index_members` 当日成分，
    市值分布用流通市值（缺失退化为等效市值），行业暴露的基准权重取 PIT 权重。
    """
    if report.strategy != STRATEGY_INDEX_ENHANCED or not benchmark:
        return None
    if report.holdings.height == 0 or WEIGHT_COL not in report.holdings.columns:
        return None

    bench_map = benchmark_weights(data_dir, benchmark, report.date)
    if not bench_map:
        return None

    holdings = (
        report.holdings.select(INSTRUMENT_COL, WEIGHT_COL)
        .with_columns(pl.lit(report.date).alias(DATE_COL))
        .select(DATE_COL, INSTRUMENT_COL, WEIGHT_COL)
    )
    members = read_index_members(Path(data_dir)).filter(pl.col(DATE_COL) == report.date)
    equiv_mv = float_mv_frame(data_dir, bars).rename({"float_mv": "equiv_mv"})
    style_factors = style_frame(bars)
    bench_frame = pl.DataFrame(
        {
            DATE_COL: [report.date] * len(bench_map),
            INSTRUMENT_COL: list(bench_map),
            "index_code": [benchmark] * len(bench_map),
            WEIGHT_COL: list(bench_map.values()),
        },
        schema={
            DATE_COL: pl.Date,
            INSTRUMENT_COL: pl.String,
            "index_code": pl.String,
            WEIGHT_COL: pl.Float64,
        },
    )
    industry_frame = industry
    if industry_frame is None or industry_frame.height == 0:
        industry_frame = _industry_frame_for_report(data_dir, report, bench_map)

    return build_risk_report(
        holdings,
        members=members,
        equiv_mv=equiv_mv,
        style_factors=style_factors,
        bench_weights=bench_frame,
        industry=industry_frame if industry_frame.height else None,
        bench_index=benchmark,
    )


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------


def _fmt_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _fmt_pct(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}%}"


def _fmt_money(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:,.2f}"


def _render_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """渲染一张对齐的 markdown 表。"""
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _render_orders(report: DailyReport) -> list[str]:
    """调仓明细：可执行订单表 + 被预过滤订单表。"""
    lines = ["## 调仓明细", ""]
    if report.orders.height == 0:
        lines.append("今日无调仓。")
    else:
        headers = ["板块", "证券", "方向", "股数", "参考价", "估算金额", "当前股数", "目标股数"]
        rows = [
            [
                BOARD_LABELS.get(str(row["board"]), str(row["board"])),
                str(row["instrument"]),
                SIDE_LABELS.get(str(row["side"]), str(row["side"])),
                str(int(row["volume"])),
                _fmt_float(float(row["ref_price"]), 3),
                _fmt_money(float(row["est_amount"])),
                str(int(row["current_volume"])),
                str(int(row["target_volume"])),
            ]
            for row in report.orders.iter_rows(named=True)
        ]
        lines.append(_render_table(headers, rows))
    lines.append("")

    lines.append("### 被预过滤订单（T 日停牌 / 涨跌停，仅预估）")
    lines.append("")
    if report.filtered.height == 0:
        lines.append("本次没有被预过滤的订单。")
    else:
        headers = ["板块", "证券", "方向", "股数", "参考价", "估算金额", "原因"]
        rows = [
            [
                BOARD_LABELS.get(str(row.get("board")), str(row.get("board"))),
                str(row["instrument"]),
                SIDE_LABELS.get(str(row["side"]), str(row["side"])),
                str(int(row["volume"])),
                _fmt_float(float(row["ref_price"]), 3),
                _fmt_money(float(row["est_amount"])),
                REASON_LABELS.get(str(row["reason"]), str(row["reason"])),
            ]
            for row in report.filtered.iter_rows(named=True)
        ]
        lines.append(_render_table(headers, rows))
    lines.append("")
    return lines


def _render_holdings(report: DailyReport) -> list[str]:
    """个股敞口：持仓表 + 目标 vs 当前权重的 top 偏离。"""
    lines = ["## 个股敞口", "", "### 当前持仓", ""]
    if report.holdings.height == 0:
        lines.append("空仓。")
    else:
        headers = ["证券", "股数", "可卖", "成本", "现价", "市值", "权重"]
        rows = [
            [
                str(row["instrument"]),
                str(int(row["volume"])),
                str(int(row["sellable"])),
                _fmt_float(float(row["avg_cost"]), 3),
                _fmt_float(float(row["price"]), 3),
                _fmt_money(float(row["market_value"])),
                _fmt_pct(float(row["weight"])),
            ]
            for row in report.holdings.iter_rows(named=True)
        ]
        lines.append(_render_table(headers, rows))
    lines.append("")

    lines.append(f"### 目标权重 vs 当前权重（偏离绝对值前 {TOP_DEVIATIONS}）")
    lines.append("")
    deviations = _top_deviations(report.target_weights, report.current_weights, TOP_DEVIATIONS)
    if not deviations:
        lines.append("目标权重与当前权重一致。")
    else:
        headers = ["证券", "当前权重", "目标权重", "偏离"]
        rows = [
            [
                instrument,
                _fmt_pct(current),
                _fmt_pct(target),
                f"{deviation:+.2%}",
            ]
            for instrument, current, target, deviation in deviations
        ]
        lines.append(_render_table(headers, rows))
    lines.append("")
    return lines


def _render_exposures(
    report: DailyReport,
    industry: Mapping[str, str] | None,
    risk_report: RiskReport | None,
) -> list[str]:
    """行业 / 风险敞口：行业层面持仓分布 + 可选的风险四表。"""
    lines = ["## 行业 / 风险敞口", "", "### 行业持仓分布", ""]
    table = industry_exposure(report.holdings, industry)
    if table is None:
        lines.append("无行业数据或空仓，略过。")
    else:
        headers = ["行业", "权重", "持仓只数"]
        rows = [
            [
                str(row["industry_l1"]),
                _fmt_pct(float(row[WEIGHT_COL])),
                str(int(row["n_positions"])),
            ]
            for row in table.iter_rows(named=True)
        ]
        lines.append(_render_table(headers, rows))
    lines.append("")

    if risk_report is None:
        if report.strategy == STRATEGY_INDEX_ENHANCED:
            lines.append(
                "风险四表未接入：缺少基准权重或当前持仓时跳过（扩展点见 "
                "`quant.daily.briefing.build_briefing_risk_report`）。"
            )
            lines.append("")
        return lines

    lines.append("### 风险四表")
    lines.append("")
    for name, frame in risk_report.tables().items():
        lines.append(f"#### {name}")
        lines.append("")
        lines.append(frame_to_markdown(frame))
        lines.append("")
    return lines


def _render_factor_ic(factor_ic: Sequence[FactorIC]) -> list[str]:
    """因子近期表现表。"""
    window = factor_ic[0].window_days if factor_ic else DEFAULT_IC_WINDOW
    lines = [f"## 因子近期表现（近 {window} 个交易日 RankIC）", ""]
    if not factor_ic:
        lines.append("未提供因子 RankIC 数据。")
        lines.append("")
        return lines
    headers = ["因子", "方向", "RankIC 均值", "标准差", "ICIR", "胜率", "有效天数"]
    rows = [
        [
            item.name,
            item.direction,
            _fmt_float(item.mean_ic),
            _fmt_float(item.std_ic),
            _fmt_float(item.icir),
            _fmt_pct(item.win_rate),
            str(item.n_days),
        ]
        for item in factor_ic
    ]
    lines.append(_render_table(headers, rows))
    lines.append("")
    return lines


def render_briefing(
    report: DailyReport,
    *,
    factor_ic: Sequence[FactorIC] = (),
    industry: Mapping[str, str] | None = None,
    risk_report: RiskReport | None = None,
    title: str | None = None,
    notes: Sequence[str] = (),
) -> str:
    """把 :class:`DailyReport` 渲染成一份 markdown 简报。"""
    lines: list[str] = []
    lines.append(f"# {title or f'每日简报 {report.date.isoformat()}'}")
    lines.append("")
    if report.already_ran:
        lines.append(
            f"> 信号日 {report.date.isoformat()} 此前已跑批，本简报未重新生成。"
        )
        lines.append("")

    exposure = report.exposure or {}
    status = "正常"
    if report.relaxed:
        status = "放宽换手约束后求解成功"
    if report.hold_fallback:
        status = "保持现有持仓（本次不调仓）"

    lines.append("## 概览")
    lines.append("")
    lines.append(f"- 信号日：{report.date.isoformat()}")
    lines.append(f"- 账户：{report.account_name}")
    lines.append(
        f"- 策略：{report.strategy}    池：{report.universe or '全市场'}    "
        f"基准：{report.benchmark or '（未指定）'}    调仓频率：{report.rebalance_freq}"
    )
    lines.append(
        f"- 净值：{_fmt_money(report.nav)} = 现金 {_fmt_money(report.cash)} + "
        f"持仓市值 {_fmt_money(report.market_value)}"
    )
    lines.append(
        f"- 敞口：现金 {_fmt_pct(exposure.get('cash_ratio'))}，"
        f"持仓 {_fmt_pct(exposure.get('position_ratio'))}，"
        f"持仓只数 {int(exposure.get('n_positions', 0.0))}"
    )
    lines.append(f"- 换手：{_fmt_float(report.turnover)}    候选数：{report.n_candidates}")
    lines.append(f"- 状态：{status}")
    lines.append("")

    if notes:
        lines.append("## 说明")
        lines.append("")
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")

    lines.extend(_render_orders(report))
    lines.extend(_render_holdings(report))
    lines.extend(_render_exposures(report, industry, risk_report))
    lines.extend(_render_factor_ic(factor_ic))
    return "\n".join(lines) + "\n"


def write_briefing(
    report: DailyReport,
    path: str | Path | None = None,
    *,
    factor_ic: Sequence[FactorIC] = (),
    industry: Mapping[str, str] | None = None,
    risk_report: RiskReport | None = None,
    title: str | None = None,
    notes: Sequence[str] = (),
) -> Path:
    """渲染并落盘简报，返回文件路径。

    ``path`` 缺省为 ``runs/reports/<信号日>.md``。文件内容与
    :func:`render_briefing` 一致。
    """
    target = (
        Path(path)
        if path is not None
        else DEFAULT_BRIEFING_DIR / f"{report.date.isoformat()}{BRIEFING_SUFFIX}"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        render_briefing(
            report,
            factor_ic=factor_ic,
            industry=industry,
            risk_report=risk_report,
            title=title,
            notes=notes,
        ),
        encoding="utf-8",
    )
    return target


__all__ = [
    "BOARD_LABELS",
    "BRIEFING_SUFFIX",
    "DEFAULT_BRIEFING_DIR",
    "DEFAULT_IC_WINDOW",
    "FactorIC",
    "REASON_LABELS",
    "TOP_DEVIATIONS",
    "build_briefing_risk_report",
    "industry_exposure",
    "recent_factor_ic",
    "render_briefing",
    "write_briefing",
]
