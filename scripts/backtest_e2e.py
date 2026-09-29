"""端到端 M1（issue #16）：训练好的模型 → AutoGluon 打分 → 凸优化 → 账户回测。

链路位置
--------
与 ``scripts/train_baseline.py`` 配对，读取它的产物 ``<model-dir>`` 与同级训练配置
JSON，在 ``[start, end]`` 区间跑一次完整的 :class:`~quant.backtest.engine.BacktestEngine`
账户回测（T+1 开盘成交、涨跌停、整手、费用、公司行为）。

两条策略路径（``--strategy``，issue #71）
----------------------------------------
- ``stock_selection``（默认，M1 现状）：打分 top-K ∪ 当前持仓 →
  :class:`~quant.portfolio.optimizer.PortfolioOptimizer`。
- ``index_enhanced``：候选 = 基准成分（PIT 权重）∪ top-K ∪ 当前持仓 →
  :class:`~quant.portfolio.enhanced.EnhancedOptimizer`，非调仓日保持持仓（权重复由
  持仓市值自然漂移）。输入组装（基准权重 / 行业 / 流通市值 / 六风格 / 收益矩阵）
  复用 :mod:`quant.portfolio.enhanced_inputs`，与 ``run_daily`` 指增分派同一套口径。
  该路径额外落 ``enhanced_log.parquet``（逐日调仓状态、放松轮数、覆盖度、个股带
  最大偏离等），并在报告中给出指增核对段。

无前视的说明（关键）
--------------------
回测开始前对**整个加载窗口**一次性 ``predict``，再把打分按信号日 T 查表。这不构成
前视，因为 :func:`~quant.automl.dataset.build_dataset` 里每只票在 T 日的特征只由
``date <= T`` 的行情算出（因子含 rolling / shift，逐日截面 z-score 只用当日截面），
``predict`` 不读 ``label``。等价于「每天收盘后用当日可得数据打分」，只是把逐日调用
合并成一次批量推理。同理，收益面板与收盘价面板都按 ``date <= T`` 切片后再喂给优化器。

股票池的两套行情窗口
--------------------
``universe`` 非空时，因子 / 打分 / 数据集用**逐日 PIT 成员**裁剪的行情（截面 z-score
只在池内算）；引擎撮合与估值另用「窗口内曾属于该池」的并集行情，否则调样后离池的
持仓既卖不掉也补不到价（broker 按当日无行情判停牌）。两套窗口都不含 cutoff 之后的信息。

信号函数（T 日收盘后被引擎调用，只读实时 ``Account``）
-----------------------------------------------------
1. 取 ``date == T`` 的打分，去缺失，按分数降序取 top-k，并上当前持仓作为候选；
2. ``w_prev`` 由实时持仓按 T 日（或最近可得）收盘价折算；
3. 收益矩阵取 T 日及以前最近 ``lookback_days`` 个交易日的候选票后复权日收益；
4. :class:`~quant.portfolio.optimizer.PortfolioOptimizer` 求目标权重；不可行时把
   ``max_turnover`` 放宽后重试，仍不可行则保持现状并记 warning；
5. 目标权重整手取整，与当前持仓 diff 出订单（先卖后买）。

用法::

    uv run python scripts/backtest_e2e.py \
        --data-dir data --model-dir runs/automl/baseline \
        --start 2025-01-02 --end 2026-09-28 \
        --initial-cash 1000000 --top-k 50 --lookback-days 250 \
        --benchmark 000905 --out-dir runs/e2e

    uv run python scripts/backtest_e2e.py --strategy index_enhanced \
        --universe zz1000 --benchmark 000852 --model-dir runs/automl/zz1000 \
        --start 2025-01-02 --end 2026-09-28 --risk-report \
        --out-dir runs/e2e-zz1000-enhanced

落盘（``--out-dir``）:: nav.parquet / report.md / account_states.md / nav.png / holdings.parquet
指定 ``--benchmark`` 时额外落 ``benchmark.parquet``（基准 / 超额净值与日收益序列），
报告「绩效」表追加基准年化 / 超额年化 / 跟踪误差 / 信息比率，净值图叠加基准与超额曲线。
指定 ``--risk-report`` 时额外落 ``risk_report.md``（issue #69 四表，读 ``holdings.parquet``）。
``--strategy index_enhanced`` 时额外落 ``enhanced_log.parquet``（逐日优化日志）。
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # 无显示环境，必须在 pyplot 之前
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from quant.automl.dataset import (  # noqa: E402
    DATE_COL,
    INSTRUMENT_COL,
    LABEL_COL,
    build_dataset,
)
from quant.automl.trainer import SCORE_COL, BaselineTrainer  # noqa: E402
from quant.backtest.broker import Broker, Order  # noqa: E402
from quant.backtest.engine import BacktestEngine, BacktestResult, SignalFn  # noqa: E402
from quant.backtest.fee import FeeModel  # noqa: E402
from quant.daily.pipeline import (  # noqa: E402
    DEFAULT_FACTOR_LIBRARY_DIR,
    diff_orders,
    discover_factors,
)
from quant.daily.strategy import (  # noqa: E402
    DEFAULT_REBALANCE_FREQ,
    DEFAULT_STRATEGY,
    REBALANCE_FREQS,
    STRATEGIES,
    StrategyConfig,
    StrategyConfigError,
    build_enhanced_optimizer,
    build_stock_optimizer,
    is_rebalance_day,
    load_strategy_config,
    parse_strategy_config,
)
from quant.data.cache import (  # noqa: E402
    load_bars,
    load_calendar,
    load_corporate_actions,
    load_index_bars,
)
from quant.eval.metrics import (  # noqa: E402
    BENCHMARK_NAV_COL,
    BenchmarkSummary,
    EXCESS_NAV_COL,
    benchmark_nav_series,
    benchmark_performance,
)
from quant.labels.open_to_open import DEFAULT_HORIZON  # noqa: E402
from quant.portfolio.enhanced import (  # noqa: E402
    FINAL_TIER_FLAG,
    STATUS_HELD,
    STATUS_RELAXED,
    EnhancedOptimizer,
)
from quant.portfolio.enhanced_inputs import (  # noqa: E402
    benchmark_weights,
    float_mv_frame,
    float_mv_map,
    industry_map,
    industry_table,
    style_frame,
    style_map,
)
from quant.portfolio.optimizer import (  # noqa: E402
    InfeasibleError,
    PortfolioError,
    PortfolioOptimizer,
)
from quant.portfolio.roundlot import round_weights_to_lots  # noqa: E402
from quant.universe.members import (  # noqa: E402
    filter_bars_to_universe,
    mean_daily_instruments,
    members_range,
)
from scripts.train_baseline import (  # noqa: E402
    TrainConfig,
    default_train_config_path,
    load_train_config,
)

logger = logging.getLogger("backtest_e2e")

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

#: 默认数据目录。
DEFAULT_DATA_DIR: str = "data"

#: 默认模型目录。
DEFAULT_MODEL_DIR: str = "runs/automl/baseline"

#: 默认产出目录（``runs/`` 不入 git）。
DEFAULT_OUT_DIR: str = "runs/e2e"

#: 默认初始资金（元）。
DEFAULT_INITIAL_CASH: float = 1_000_000.0

#: 默认候选证券数。
DEFAULT_TOP_K: int = 50

#: 默认行情回溯交易日数（协方差与因子窗口）。
DEFAULT_LOOKBACK_DAYS: int = 250

#: 因子预热余量（交易日）：在 ``lookback_days`` 之外多取，供 rolling / shift 预热。
FACTOR_WINDOW_MARGIN: int = 60

#: 年化与夏普使用的年交易日数。
TRADING_DAYS_PER_YEAR: int = 252

#: 优化不可行的放宽倍数与下限。
#:
#: 空仓首次建仓的 ``|w − w_prev|₁`` 恒等于 1，若只按倍数放宽（0.3 × 3 = 0.9）仍会
#: 不可行，整轮回测将永远保持空仓。因此放宽值再取 ``1.0`` 下限，保证首次建仓可行；
#: 长多组合 ``Σw = 1`` 时 ``|w − w_prev|₁`` 的理论上限为 2，下限 1.0 仍在合法范围。
TURNOVER_RELAX_FACTOR: float = 3.0
TURNOVER_RELAX_FLOOR: float = 1.0

#: 归一化目标权重时允许的和上限容差（与 pipeline 一致）。
WEIGHT_SUM_TOL: float = 1e-9

#: 报告里列出的 Top 拒单原因数。
TOP_REJECT_REASONS: int = 5

#: account_states.md 抽样的有成交交易日数。
SAMPLE_DAYS: int = 3

#: 净值图 DPI。
PNG_DPI: int = 120

#: 指增个股带自洽性检查的绝对松弛：求解器容差 + ``_finalize`` 约零归一带来的微小偏移。
BAND_CHECK_ABS_SLACK: float = 1e-4

#: 指增日志保留的最大 warning 条数。
MAX_ENHANCED_NOTES: int = 200

ORDER_SIDE_BUY: str = "buy"
ORDER_SIDE_SELL: str = "sell"


# ---------------------------------------------------------------------------
# 异常与结果类型
# ---------------------------------------------------------------------------


class E2EError(Exception):
    """端到端回测无法继续。"""


@dataclass(frozen=True)
class E2EConfig:
    """一次端到端回测的全部输入配置。"""

    data_dir: Path
    model_dir: Path
    out_dir: Path
    start: date
    end: date
    initial_cash: float = DEFAULT_INITIAL_CASH
    top_k: int = DEFAULT_TOP_K
    lookback_days: int = DEFAULT_LOOKBACK_DAYS
    factor_library_dir: Path = DEFAULT_FACTOR_LIBRARY_DIR
    horizon: int = DEFAULT_HORIZON
    train_config_path: Path | None = None
    #: 基准指数代码（六位，如 ``000905``）；None 表示不计算基准 / 超额绩效。
    benchmark: str | None = None
    #: 是否额外产出持仓风险分析四表（``risk_report.md``，issue #69）。
    risk_report: bool = False
    #: 回测股票池（命名池名或自定义池路径）；None 表示全市场。须与训练配置一致。
    universe: str | None = None
    #: 策略枚举：``stock_selection`` / ``index_enhanced``（issue #71）。
    strategy: str = DEFAULT_STRATEGY
    #: 指增调仓频率：``D`` / ``W`` / ``M``（仅 ``index_enhanced`` 生效）。
    rebalance_freq: str = DEFAULT_REBALANCE_FREQ
    #: 策略配置 JSON 路径；给了就用它的 ``optimize`` 等参数，并要求与上面各项一致。
    strategy_config_path: Path | None = None

    @property
    def resolved_train_config_path(self) -> Path:
        """训练配置路径：显式指定优先，否则按 ``model-dir`` 的同级约定。"""
        return (
            self.train_config_path
            if self.train_config_path is not None
            else default_train_config_path(self.model_dir)
        )

    def resolved_strategy_config(self) -> StrategyConfig:
        """把命令行参数整理成已校验的 :class:`StrategyConfig`。

        给了 ``--strategy-config`` 时以该 JSON 为准，但要求 ``strategy`` / ``universe`` /
        ``benchmark`` / ``top_k`` / ``rebalance_freq`` 与命令行参数逐项一致，避免同一份
        配置里出现两套互相矛盾的口径（与训练配置的 universe 校验同思路）。
        """
        if self.strategy_config_path is None:
            try:
                return parse_strategy_config(
                    {
                        "strategy": self.strategy,
                        "universe": self.universe,
                        "benchmark": self.benchmark,
                        "top_k": self.top_k,
                        "rebalance_freq": self.rebalance_freq,
                        "optimize": {},
                    }
                )
            except StrategyConfigError as exc:
                raise E2EError(f"策略配置不合法：{exc}") from exc
        loaded = load_strategy_config(self.strategy_config_path)
        mismatched = [
            name
            for name, cli_value, file_value in (
                ("strategy", self.strategy, loaded.strategy),
                ("universe", self.universe, loaded.universe),
                ("benchmark", self.benchmark, loaded.benchmark),
                ("top_k", self.top_k, loaded.top_k),
                ("rebalance_freq", self.rebalance_freq, loaded.rebalance_freq),
            )
            if cli_value != file_value
        ]
        if mismatched:
            raise E2EError(
                f"策略配置 {self.strategy_config_path} 与命令行参数不一致：{mismatched}；"
                "请让两边给出同一份口径"
            )
        return loaded


@dataclass
class SignalStats:
    """信号函数运行统计，供报告核对优化调用与降级次数。"""

    calls: int = 0
    optimize_calls: int = 0
    empty_score_days: int = 0
    relaxed_attempts: int = 0
    relaxed_success: int = 0
    hold_fallback: int = 0
    orders: int = 0
    notes: list[str] = field(default_factory=list)


#: 指增逐日优化日志 schema（``enhanced_log.parquet``）。
#:
#: 口径：``cover_rate`` / ``max_band_dev`` 等暴露量都按**归一后的当日持仓**计算
#: （求解日 = 优化目标权重，held 日 = 漂移后持仓的归一权重），因此可以逐日横向比较；
#: ``invested_ratio`` 单独记实际账户在信号日的已投资比例（市值 / nav），用于分辨
#: 「组合偏离基准」与「现金没投出去」两类问题。
ENHANCED_LOG_SCHEMA: pl.Schema = pl.Schema(
    {
        "date": pl.Date,
        "status": pl.String,
        "relax_rounds": pl.Int64,
        "final_turnover_free": pl.Boolean,
        "turnover": pl.Float64,
        "n_bench": pl.Int64,
        "n_bench_candidates": pl.Int64,
        "n_candidates": pl.Int64,
        "invested_ratio": pl.Float64,
        "cover_rate": pl.Float64,
        "max_band_dev": pl.Float64,
        "band_violations": pl.Int64,
        "max_industry_ratio": pl.Float64,
        "zero_bench_weight": pl.Float64,
        "market_value_std": pl.Float64,
        "style_std_max": pl.Float64,
        "thresholds": pl.String,
    }
)


@dataclass
class EnhancedStats:
    """指增路径的核对统计（issue #71）：覆盖度、个股带、放松轮次、行业快照退化。"""

    rebalance_days: int = 0
    non_rebalance_days: int = 0
    held_days: int = 0
    final_tier_days: int = 0
    relax_rounds: Counter[int] = field(default_factory=Counter)
    cover_rate_min: float | None = None
    cover_rate_min_date: date | None = None
    max_band_dev: float = 0.0
    max_band_dev_date: date | None = None
    band_violations: int = 0
    industry_fallback_days: int = 0
    bench_missing_days: int = 0
    records: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def add_note(self, text: str) -> None:
        """记录一条 warning（上限 :data:`MAX_ENHANCED_NOTES`）。"""
        if len(self.notes) < MAX_ENHANCED_NOTES:
            self.notes.append(text)

    @property
    def log_frame(self) -> pl.DataFrame:
        """逐日优化日志表（``records`` 为空时返回零行同 schema 表）。"""
        if not self.records:
            return pl.DataFrame(schema=ENHANCED_LOG_SCHEMA)
        return pl.DataFrame(self.records, schema=ENHANCED_LOG_SCHEMA)


@dataclass
class E2EResult:
    """端到端回测产出。"""

    config: E2EConfig
    train_config: TrainConfig | None
    result: BacktestResult
    metrics: dict[str, Any]
    stats: SignalStats
    nav_path: Path
    report_path: Path
    states_path: Path
    png_path: Path
    #: 基准绩效汇总与对齐序列（``--benchmark`` 未指定时为 None）。
    benchmark: BenchmarkSummary | None = None
    benchmark_series: pl.DataFrame | None = None
    benchmark_path: Path | None = None
    #: 逐日持仓权重落盘路径（``holdings.parquet``，issue #69）。
    holdings_path: Path | None = None
    #: 持仓风险分析 markdown 路径（``--risk-report`` 未开启时为 None）。
    risk_report_path: Path | None = None
    #: 指增逐日优化日志（``--strategy index_enhanced`` 时非 None）。
    enhanced_log: pl.DataFrame | None = None
    enhanced_log_path: Path | None = None
    enhanced_stats: EnhancedStats | None = None


# ---------------------------------------------------------------------------
# 面板构造（收盘价 / 后复权收益）
# ---------------------------------------------------------------------------


def build_close_panel(bars: pl.DataFrame) -> pl.DataFrame:
    """构造 ``date × instrument`` 收盘价面板，并对停牌日前向填充。"""
    if bars.height == 0:
        raise ValueError("行情为空，无法构造收盘价面板")
    panel = bars.pivot(on=INSTRUMENT_COL, index=DATE_COL, values="close").sort(DATE_COL)
    value_columns = [col for col in panel.columns if col != DATE_COL]
    return panel.with_columns([pl.col(col).forward_fill() for col in value_columns])


def build_returns_panel(bars: pl.DataFrame) -> pl.DataFrame:
    """构造 ``date × instrument`` 后复权日收益面板（``close × adjfactor`` 的 pct_change）。"""
    if bars.height == 0:
        raise ValueError("行情为空，无法构造收益面板")
    window = bars.sort([INSTRUMENT_COL, DATE_COL]).with_columns(
        (pl.col("close") * pl.col("adjfactor")).alias("_adj_close")
    )
    window = window.with_columns(
        pl.when(
            (pl.col("_adj_close").shift(1).over(INSTRUMENT_COL) > 0)
            & pl.col("_adj_close").is_finite()
        )
        .then(
            pl.col("_adj_close")
            / pl.col("_adj_close").shift(1).over(INSTRUMENT_COL)
            - 1.0
        )
        .otherwise(None)
        .alias("_ret")
    )
    return window.pivot(
        on=INSTRUMENT_COL, index=DATE_COL, values="_ret"
    ).sort(DATE_COL)


def _close_on(panel: pl.DataFrame, day: date) -> dict[str, float]:
    """取 ``day`` 当日（缺失时取最近不晚于 ``day``）的正收盘价映射。"""
    rows = panel.filter(pl.col(DATE_COL) == day)
    if rows.height == 0:
        rows = panel.filter(pl.col(DATE_COL) <= day)
        if rows.height == 0:
            return {}
        rows = rows.tail(1)
    record = rows.row(0, named=True)
    prices: dict[str, float] = {}
    for key, value in record.items():
        if key == DATE_COL or value is None:
            continue
        number = float(value)
        if math.isfinite(number) and number > 0.0:
            prices[str(key)] = number
    return prices


def select_returns(
    panel: pl.DataFrame,
    instruments: list[str],
    ref_date: date,
    lookback_days: int,
) -> pl.DataFrame:
    """取 ``ref_date`` 及以前最近 ``lookback_days`` 个交易日的候选收益矩阵。

    列顺序与 ``instruments`` 严格一致（优化器会校验）；面板中缺列的候选补 0，
    缺失 / 非有限值填 0，保证矩阵始终有限。
    """
    window = panel.filter(pl.col(DATE_COL) <= ref_date).tail(lookback_days)
    missing = [inst for inst in instruments if inst not in window.columns]
    if missing:
        window = window.with_columns([pl.lit(0.0).alias(inst) for inst in missing])
    if window.height == 0:
        return pl.DataFrame({inst: pl.Series([], dtype=pl.Float64) for inst in instruments})
    clean = window.select(list(instruments)).with_columns(
        [
            pl.when(pl.col(inst).is_finite())
            .then(pl.col(inst))
            .otherwise(0.0)
            .alias(inst)
            for inst in instruments
        ]
    )
    return clean


# ---------------------------------------------------------------------------
# 优化降级
# ---------------------------------------------------------------------------


def relax_turnover(optimizer: PortfolioOptimizer) -> PortfolioOptimizer:
    """放宽 ``max_turnover``：按倍数放大并取 ``TURNOVER_RELAX_FLOOR`` 下限。"""
    relaxed = max(
        float(optimizer.max_turnover) * TURNOVER_RELAX_FACTOR, TURNOVER_RELAX_FLOOR
    )
    return replace(optimizer, max_turnover=relaxed)


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    """把目标权重归一到 ``Σw <= 1``，避免整手取整透支。"""
    total = float(sum(weights.values()))
    if total <= 0.0:
        raise E2EError("优化器给出的目标权重之和非正")
    if total > 1.0 + WEIGHT_SUM_TOL:
        return {inst: value / total for inst, value in weights.items()}
    return dict(weights)


# ---------------------------------------------------------------------------
# 信号函数
# ---------------------------------------------------------------------------


def make_signal_fn(
    score_by_day: dict[date, pl.DataFrame],
    close_panel: pl.DataFrame,
    returns_panel: pl.DataFrame,
    optimizer: PortfolioOptimizer,
    *,
    top_k: int,
    lookback_days: int,
    stats: SignalStats,
) -> SignalFn:
    """构造引擎所需的 ``signal_fn(T, history, account) -> list[Order]``。

    ``history`` 由引擎保证只含 ``date <= T`` 的行；本实现改用预构造的面板并显式按
    ``date <= T`` 切片，口径与 ``history`` 等价（见模块 docstring 的无前视说明）。
    """

    def signal(day: date, _history: pl.DataFrame, account: Any) -> list[Order]:
        stats.calls += 1
        score_frame = score_by_day.get(day)
        if score_frame is None or score_frame.height == 0:
            stats.empty_score_days += 1
            return []
        scores = score_frame.filter(
            pl.col(SCORE_COL).is_not_null() & pl.col(SCORE_COL).is_finite()
        )
        if scores.height == 0:
            stats.empty_score_days += 1
            return []
        scores = scores.sort([SCORE_COL, INSTRUMENT_COL], descending=[True, False])
        score_map = {
            str(row[INSTRUMENT_COL]): float(row[SCORE_COL])
            for row in scores.iter_rows(named=True)
        }
        min_score = min(score_map.values())
        top_instruments = [str(inst) for inst in scores.head(top_k)[INSTRUMENT_COL].to_list()]

        positions = {
            inst: pos
            for inst, pos in account.positions.items()
            if pos.volume > 0
        }
        current_volumes = {inst: pos.volume for inst, pos in positions.items()}

        prices = _close_on(close_panel, day)
        # 缺价持仓用摊薄成本兜底估值（与 pipeline 一致）；仍无价则记 0，只用于估值。
        for inst, pos in positions.items():
            if inst not in prices:
                prices[inst] = pos.avg_cost if pos.avg_cost > 0.0 else 0.0

        candidates = list(dict.fromkeys(top_instruments + list(current_volumes)))
        candidates = [inst for inst in candidates if prices.get(inst, 0.0) > 0.0]
        if not candidates:
            stats.hold_fallback += 1
            stats.notes.append(f"{day.isoformat()}: 候选取价失败，保持现状")
            return []

        nav = account.nav(prices)
        if nav <= 0.0:
            stats.hold_fallback += 1
            stats.notes.append(f"{day.isoformat()}: nav 非正，保持现状")
            return []

        candidate_set = set(candidates)
        current_weights = {
            inst: prices[inst] * volume / nav
            for inst, volume in current_volumes.items()
            if inst in candidate_set
        }
        alpha = {inst: score_map.get(inst, min_score) for inst in candidates}

        returns = select_returns(returns_panel, candidates, day, lookback_days)
        if returns.height < 2:
            stats.hold_fallback += 1
            stats.notes.append(f"{day.isoformat()}: 收益观测不足，保持现状")
            return []

        optimized = _optimize(optimizer, alpha, candidates, returns, current_weights, day, stats)
        if optimized is None:
            return []

        try:
            target_weights = normalize_weights(optimized.weights)
            rounded = round_weights_to_lots(target_weights, prices, nav)
        except (ValueError, E2EError) as exc:
            stats.hold_fallback += 1
            stats.notes.append(f"{day.isoformat()}: 目标权重取整失败，保持现状（{exc}）")
            return []

        target_volumes = {
            str(row["instrument"]): int(row["volume"])
            for row in rounded.orders.iter_rows(named=True)
        }
        diff = diff_orders(target_volumes, current_volumes)
        orders = [
            Order(str(row["instrument"]), str(row["side"]), int(row["volume"]))
            for row in diff.iter_rows(named=True)
        ]
        stats.orders += len(orders)
        return orders

    return signal


def _optimize(
    optimizer: PortfolioOptimizer,
    alpha: dict[str, float],
    candidates: list[str],
    returns: pl.DataFrame,
    w_prev: dict[str, float],
    day: date,
    stats: SignalStats,
) -> Any | None:
    """先按原约束求解，不可行时放宽换手重试；仍不可行返回 None（保持现状）。"""
    stats.optimize_calls += 1
    try:
        return optimizer.optimize(alpha, candidates, returns, w_prev)
    except InfeasibleError:
        stats.relaxed_attempts += 1
    except PortfolioError as exc:
        stats.hold_fallback += 1
        stats.notes.append(f"{day.isoformat()}: 优化求解失败，保持现状（{exc}）")
        return None

    stats.optimize_calls += 1
    relaxed = relax_turnover(optimizer)
    try:
        outcome = relaxed.optimize(alpha, candidates, returns, w_prev)
    except (InfeasibleError, PortfolioError) as exc:
        stats.hold_fallback += 1
        stats.notes.append(f"{day.isoformat()}: 放宽换手后仍不可行，保持现状（{exc}）")
        return None
    stats.relaxed_success += 1
    logger.warning("%s 组合优化不可行，已放宽 max_turnover 至 %.4f 重试成功", day, relaxed.max_turnover)
    return outcome


# ---------------------------------------------------------------------------
# 指增信号函数（issue #71）
# ---------------------------------------------------------------------------


def _band_check(
    weights: dict[str, float],
    bench_weights: dict[str, float],
    band: float,
) -> tuple[float, int]:
    """个股带自洽性检查：返回 ``(|w_i − bench_norm_i| 最大值, 超带证券数)``。

    ``bench_norm`` 为基准权重在候选集内归一后的值（与优化器约束同一口径）；
    判定超带的条件为 ``deviation > band + BAND_CHECK_ABS_SLACK``。
    """
    total = float(sum(bench_weights.values()))
    if total <= 0.0:
        return 0.0, 0
    limit = band + BAND_CHECK_ABS_SLACK
    worst = 0.0
    violations = 0
    for instrument, bench in bench_weights.items():
        deviation = abs(float(weights.get(instrument, 0.0)) - float(bench) / total)
        worst = max(worst, deviation)
        if deviation > limit:
            violations += 1
    return worst, violations


def _exposure_diagnostics(exposures: dict[str, Any]) -> tuple[float, float, float, float]:
    """从 :func:`quant.portfolio.enhanced.compute_exposures` 的结果抽出核对量。

    返回 ``(行业最大相对偏离, 零基准行业上的组合权重, 市值偏离 std 倍数, 风格最大 std 倍数)``。
    """
    industry = exposures.get("industry", {})
    max_ratio = 0.0
    zero_bench = 0.0
    for item in industry.values():
        bench = float(item.get("bench", 0.0))
        portfolio = float(item.get("portfolio", 0.0))
        if bench > 0.0:
            max_ratio = max(max_ratio, abs(portfolio / bench - 1.0))
        else:
            zero_bench = max(zero_bench, portfolio)
    market_value = exposures.get("market_value", {})
    mv_std = abs(float(market_value.get("std", 0.0)))
    style = exposures.get("style", {})
    style_std = max(
        (abs(float(item.get("std", 0.0))) for item in style.values()), default=0.0
    )
    return max_ratio, zero_bench, mv_std, style_std


def make_enhanced_signal_fn(
    *,
    data_dir: Path,
    score_by_day: dict[date, pl.DataFrame],
    close_panel: pl.DataFrame,
    returns_panel: pl.DataFrame,
    bars: pl.DataFrame,
    calendar: pl.DataFrame,
    strategy_config: StrategyConfig,
    optimizer: EnhancedOptimizer,
    lookback_days: int,
    stats: SignalStats,
    enhanced: EnhancedStats,
) -> SignalFn:
    """构造指增路径的 ``signal_fn``，口径与 ``run_daily`` 的指增分派一致。

    候选 = 基准成分（PIT 权重）∪ 打分 top-K ∪ 当前持仓；非调仓日直接保持持仓（引擎
    按市值自然漂移权重）。行业 / 流通市值 / 六风格在各日按 cutoff 截取，风格与市值
    全窗口只算一次，避免逐日重算。

    核对口径：逐日日志的覆盖度 / 个股带偏离按**归一后的当日持仓**计算——求解日取优化
    目标权重，``held`` 日取漂移后持仓的归一权重（``EnhancedOptimizer.optimize_day``
    内部统一）。实际已投资比例单列 ``invested_ratio``，避免把「组合偏离基准」与
    「现金没投出去」混为一谈。
    """
    benchmark = strategy_config.benchmark
    if benchmark is None:
        raise E2EError("index_enhanced 策略缺少 benchmark")
    open_days = calendar.filter(pl.col("is_open"))[DATE_COL].to_list()
    style = style_frame(bars)
    mv_frame = float_mv_frame(data_dir, bars)

    def signal(day: date, _history: pl.DataFrame, account: Any) -> list[Order]:
        stats.calls += 1
        if not is_rebalance_day(open_days, day, strategy_config.rebalance_freq):
            enhanced.non_rebalance_days += 1
            return []
        enhanced.rebalance_days += 1

        score_frame = score_by_day.get(day)
        if score_frame is None or score_frame.height == 0:
            stats.empty_score_days += 1
            return []
        scores = score_frame.filter(
            pl.col(SCORE_COL).is_not_null() & pl.col(SCORE_COL).is_finite()
        )
        if scores.height == 0:
            stats.empty_score_days += 1
            return []
        scores = scores.sort([SCORE_COL, INSTRUMENT_COL], descending=[True, False])
        score_map = {
            str(row[INSTRUMENT_COL]): float(row[SCORE_COL])
            for row in scores.iter_rows(named=True)
        }
        min_score = min(score_map.values())
        top_instruments = [
            str(inst)
            for inst in scores.head(strategy_config.top_k)[INSTRUMENT_COL].to_list()
        ]

        bench = benchmark_weights(data_dir, benchmark, day)
        if not bench:
            enhanced.bench_missing_days += 1
            stats.hold_fallback += 1
            enhanced.add_note(f"{day.isoformat()}: 基准权重缺失，保持现状")
            return []

        positions = {
            inst: pos for inst, pos in account.positions.items() if pos.volume > 0
        }
        current_volumes = {inst: pos.volume for inst, pos in positions.items()}

        prices = _close_on(close_panel, day)
        # 缺价持仓用摊薄成本兜底估值（与 pipeline 一致），仍无价则记 0，只用于估值。
        for inst, pos in positions.items():
            if inst not in prices:
                prices[inst] = pos.avg_cost if pos.avg_cost > 0.0 else 0.0

        bench_candidates = sorted(
            inst for inst in bench if prices.get(inst, 0.0) > 0.0
        )
        ordered = list(
            dict.fromkeys([*bench_candidates, *top_instruments, *current_volumes])
        )
        candidates = [inst for inst in ordered if prices.get(inst, 0.0) > 0.0]
        if not candidates:
            stats.hold_fallback += 1
            enhanced.add_note(f"{day.isoformat()}: 候选取价失败，保持现状")
            return []

        nav = account.nav(prices)
        if nav <= 0.0:
            stats.hold_fallback += 1
            enhanced.add_note(f"{day.isoformat()}: nav 非正，保持现状")
            return []
        invested_ratio = account.market_value(prices) / nav

        candidate_set = set(candidates)
        current_weights = {
            inst: prices[inst] * volume / nav
            for inst, volume in current_volumes.items()
            if inst in candidate_set
        }
        alpha = {inst: score_map.get(inst, min_score) for inst in candidates}
        returns = select_returns(returns_panel, candidates, day, lookback_days)
        if returns.height < 2:
            stats.hold_fallback += 1
            enhanced.add_note(f"{day.isoformat()}: 收益观测不足，保持现状")
            return []

        bench_cands = {inst: bench[inst] for inst in bench_candidates}
        covered = sorted(set(candidates) | set(bench_cands))
        industry_table_frame, fallback = industry_table(data_dir, day, covered)
        if fallback:
            enhanced.industry_fallback_days += 1
        industry = industry_map(industry_table_frame)

        stats.optimize_calls += 1
        try:
            outcome = optimizer.optimize_day(
                date=day,
                instruments=candidates,
                alpha=alpha,
                bench_weights=bench_cands,
                industry=industry,
                float_mv=float_mv_map(mv_frame, day),
                style=style_map(style, day),
                covariance=returns,
                w_prev=current_weights or None,
            )
        except (ValueError, PortfolioError) as exc:
            stats.hold_fallback += 1
            enhanced.add_note(f"{day.isoformat()}: 指增优化异常，保持现状（{exc}）")
            return []

        band_dev, band_violations = _band_check(
            outcome.weights, bench_cands, optimizer.stock_band
        )
        max_industry, zero_bench, mv_std, style_std = _exposure_diagnostics(
            outcome.exposures
        )
        cover_rate = float(outcome.exposures.get("cover_rate", 0.0))
        final_tier = bool(outcome.thresholds.get(FINAL_TIER_FLAG, False))
        enhanced.relax_rounds[int(outcome.relax_rounds)] += 1
        if final_tier:
            enhanced.final_tier_days += 1
        if band_dev >= enhanced.max_band_dev:
            enhanced.max_band_dev = band_dev
            enhanced.max_band_dev_date = day
        enhanced.band_violations += band_violations
        if enhanced.cover_rate_min is None or cover_rate < enhanced.cover_rate_min:
            enhanced.cover_rate_min = cover_rate
            enhanced.cover_rate_min_date = day
        enhanced.records.append(
            {
                "date": day,
                "status": str(outcome.status),
                "relax_rounds": int(outcome.relax_rounds),
                "final_turnover_free": final_tier,
                "turnover": float(outcome.turnover),
                "n_bench": len(bench),
                "n_bench_candidates": len(bench_candidates),
                "n_candidates": len(candidates),
                "invested_ratio": float(invested_ratio),
                "cover_rate": cover_rate,
                "max_band_dev": band_dev,
                "band_violations": band_violations,
                "max_industry_ratio": max_industry,
                "zero_bench_weight": zero_bench,
                "market_value_std": mv_std,
                "style_std_max": style_std,
                "thresholds": json.dumps(outcome.thresholds, ensure_ascii=False),
            }
        )

        if final_tier:
            enhanced.add_note(
                f"{day.isoformat()}: 常规放松用尽，终局台阶取消换手约束后求解成功"
            )
        if band_violations:
            enhanced.add_note(
                f"{day.isoformat()}: 个股带超带 {band_violations} 只，最大偏离 {band_dev:.6f}"
            )
        if outcome.status == STATUS_HELD:
            enhanced.held_days += 1
            stats.hold_fallback += 1
            enhanced.add_note(f"{day.isoformat()}: 指增优化失败，保持现状")
            return []
        if outcome.status == STATUS_RELAXED:
            stats.relaxed_attempts += 1
            stats.relaxed_success += 1

        try:
            target_weights = normalize_weights(dict(outcome.weights))
            rounded = round_weights_to_lots(target_weights, prices, nav)
        except (ValueError, E2EError) as exc:
            stats.hold_fallback += 1
            enhanced.add_note(f"{day.isoformat()}: 目标权重取整失败，保持现状（{exc}）")
            return []

        target_volumes = {
            str(row["instrument"]): int(row["volume"])
            for row in rounded.orders.iter_rows(named=True)
        }
        diff = diff_orders(target_volumes, current_volumes)
        orders = [
            Order(str(row["instrument"]), str(row["side"]), int(row["volume"]))
            for row in diff.iter_rows(named=True)
        ]
        stats.orders += len(orders)
        return orders

    return signal


# ---------------------------------------------------------------------------
# 运行
# ---------------------------------------------------------------------------


def run_e2e(
    config: E2EConfig,
    *,
    trainer: BaselineTrainer | None = None,
    optimizer: PortfolioOptimizer | None = None,
    enhanced_optimizer: EnhancedOptimizer | None = None,
) -> E2EResult:
    """执行端到端回测并落盘，返回 :class:`E2EResult`。

    ``trainer`` / ``optimizer`` / ``enhanced_optimizer`` 用于依赖注入（测试传假对象，
    不加载真模型、不跑真 cvxpy）。
    """
    started = time.monotonic()
    if config.end < config.start:
        raise E2EError(f"区间非法：start={config.start} > end={config.end}")

    train_config: TrainConfig | None = None
    config_path = config.resolved_train_config_path
    if config_path.exists():
        train_config = load_train_config(config_path)
    elif trainer is None:
        raise E2EError(f"缺少训练配置且未注入 trainer：{config_path}")

    # -- 训练 / 回测股票池一致性：训练在池内做截面 z-score，回测必须同池 ----------
    trained_universe = train_config.universe if train_config is not None else None
    if trained_universe != config.universe:
        raise E2EError(
            f"训练配置 universe={trained_universe!r} 与回测 universe="
            f"{config.universe!r} 不一致，请显式给出匹配的 --universe"
        )

    # -- 策略配置校验（issue #71）------------------------------------------
    strategy_config = config.resolved_strategy_config()
    if strategy_config.is_index_enhanced and config.benchmark is None:
        raise E2EError("index_enhanced 策略必须给出 --benchmark")

    # -- 行情窗口：start 之前留 lookback + 因子余量，供因子与收益矩阵预热 ----------
    calendar = load_calendar(config.data_dir)
    window_start = _window_start(
        calendar, config.start, config.lookback_days + FACTOR_WINDOW_MARGIN
    )
    loaded = load_bars(config.data_dir, start=window_start, end=config.end)
    if loaded.height == 0:
        raise E2EError(f"行情窗口 [{window_start}, {config.end}] 内没有数据")
    bars = loaded
    #: 引擎侧行情：universe 非空时放宽到「窗口内曾属于该池」的并集，保证调样后离池的
    #: 持仓仍能卖出 / 估值（逐日 PIT 裁剪会让离池票在 broker 眼里变成停牌）。
    engine_bars = loaded
    if config.universe is not None:
        bars = filter_bars_to_universe(loaded, config.universe, data_dir=config.data_dir)
        if bars.height == 0:
            raise E2EError(f"股票池 {config.universe!r} 在窗口内没有行情")
        pool = members_range(
            config.universe, window_start, config.end, data_dir=config.data_dir
        )
        engine_bars = loaded.filter(
            pl.col(INSTRUMENT_COL).is_in(pool[INSTRUMENT_COL].unique().to_list())
        )
    universe_mean_daily = mean_daily_instruments(bars)
    actions = load_corporate_actions(
        config.data_dir, start=config.start, end=config.end
    )

    # -- 因子 → 数据集 → 一次性打分 ------------------------------------------
    factors = discover_factors(config.factor_library_dir)
    feature_columns = list(factors)
    if train_config is not None and list(train_config.feature_columns) != feature_columns:
        raise E2EError(
            "模型训练配置的特征列与当前因子库不一致："
            f"{train_config.feature_columns} != {feature_columns}"
        )
    dataset = build_dataset(bars, factors, horizon=config.horizon)

    active_trainer = trainer
    if active_trainer is None:
        active_trainer = BaselineTrainer.load(
            config.model_dir, label=LABEL_COL, feature_columns=feature_columns
        )
    scores = active_trainer.predict(dataset)
    score_by_day = _group_scores(scores, config.start, config.end)
    logger.info(
        "打分覆盖 %d 个信号日，行情窗口 %s ~ %s，%d 行（策略 %s）",
        len(score_by_day),
        window_start,
        config.end,
        bars.height,
        strategy_config.strategy,
    )

    close_panel = build_close_panel(engine_bars)
    returns_panel = build_returns_panel(engine_bars)

    stats = SignalStats()
    enhanced_stats: EnhancedStats | None = None
    if strategy_config.is_index_enhanced:
        enhanced_stats = EnhancedStats()
        enhanced_optimizer = (
            enhanced_optimizer
            if enhanced_optimizer is not None
            else build_enhanced_optimizer(strategy_config)
        )
        signal_fn = make_enhanced_signal_fn(
            data_dir=config.data_dir,
            score_by_day=score_by_day,
            close_panel=close_panel,
            returns_panel=returns_panel,
            bars=engine_bars,
            calendar=calendar,
            strategy_config=strategy_config,
            optimizer=enhanced_optimizer,
            lookback_days=config.lookback_days,
            stats=stats,
            enhanced=enhanced_stats,
        )
    else:
        active_optimizer = (
            optimizer if optimizer is not None else build_stock_optimizer(strategy_config)
        )
        signal_fn = make_signal_fn(
            score_by_day,
            close_panel,
            returns_panel,
            active_optimizer,
            top_k=config.top_k,
            lookback_days=config.lookback_days,
            stats=stats,
        )

    result = BacktestEngine(Broker(FeeModel())).run(
        engine_bars,
        calendar,
        actions,
        signal_fn,
        config.start,
        config.end,
        config.initial_cash,
    )

    benchmark = _load_benchmark(config, result)
    benchmark_summary = benchmark[0] if benchmark is not None else None
    benchmark_series = benchmark[1] if benchmark is not None else None

    enhanced_log = enhanced_stats.log_frame if enhanced_stats is not None else None
    nav_path, report_path, states_path, png_path, benchmark_path = _write_outputs(
        config,
        result,
        stats,
        train_config,
        close_panel,
        benchmark_summary,
        benchmark_series,
        universe_mean_daily,
        strategy_config,
        enhanced_stats,
    )
    enhanced_log_path = None
    if enhanced_log is not None:
        enhanced_log_path = config.out_dir / "enhanced_log.parquet"
        enhanced_log.write_parquet(enhanced_log_path)
    holdings_path = config.out_dir / "holdings.parquet"
    holdings = build_holdings_weights(result, close_panel)
    holdings.write_parquet(holdings_path)
    risk_report_path = _write_risk_report(config, holdings)
    metrics = compute_metrics(result)
    logger.info(
        "回测完成：%d 个交易日，期末净值 %.2f，总收益 %.2f%%，耗时 %.1fs",
        result.trading_days,
        result.final_nav,
        result.total_return * 100.0,
        time.monotonic() - started,
    )
    return E2EResult(
        config=config,
        train_config=train_config,
        result=result,
        metrics=metrics,
        stats=stats,
        nav_path=nav_path,
        report_path=report_path,
        states_path=states_path,
        png_path=png_path,
        benchmark=benchmark_summary,
        benchmark_series=benchmark_series,
        benchmark_path=benchmark_path,
        holdings_path=holdings_path,
        risk_report_path=risk_report_path,
        enhanced_log=enhanced_log,
        enhanced_log_path=enhanced_log_path,
        enhanced_stats=enhanced_stats,
    )


def _load_benchmark(
    config: E2EConfig, result: BacktestResult
) -> tuple[BenchmarkSummary, pl.DataFrame] | None:
    """读取基准指数并计算基准 / 超额绩效，未配置 ``--benchmark`` 返回 None。"""
    if config.benchmark is None:
        return None
    index_bars = load_index_bars(
        config.data_dir,
        index_codes=[config.benchmark],
        start=config.start,
        end=config.end,
    )
    if index_bars.height == 0:
        raise E2EError(
            f"index_bars 缺少基准 {config.benchmark} 在 "
            f"[{config.start}, {config.end}] 的数据，请先跑 daily_update"
        )
    benchmark_close = index_bars.select(DATE_COL, "close")
    portfolio_nav = result.nav.select(DATE_COL, "nav")
    summary = benchmark_performance(portfolio_nav, benchmark_close)
    series = benchmark_nav_series(portfolio_nav, benchmark_close)
    if series.height == 0:
        raise E2EError("组合与基准没有共同交易日，无法计算超额绩效")
    logger.info(
        "基准 %s：%d 个对齐交易日，超额年化 %.4f%%，跟踪误差 %.4f%%，IR %.4f",
        config.benchmark,
        summary.n_days,
        (summary.excess_annualized or 0.0) * 100.0,
        (summary.tracking_error or 0.0) * 100.0,
        summary.information_ratio or 0.0,
    )
    return summary, series


def _write_risk_report(config: E2EConfig, holdings: pl.DataFrame) -> Path | None:
    """按需产出持仓风险分析四表；失败只告警，不影响回测主流程。"""
    if not config.risk_report:
        return None
    if holdings.height == 0:
        logger.warning("持仓为空，跳过持仓风险分析")
        return None
    from scripts.risk_report import MV_SHARE_CONST, run as run_risk_report

    try:
        text, _ = run_risk_report(
            holdings=holdings,
            data_dir=config.data_dir,
            benchmark=config.benchmark,
            share_const=MV_SHARE_CONST,
            check_consistency=False,
            industry_tol=None,
            style_tol=None,
            market_value_tol=None,
        )
    except Exception as exc:  # noqa: BLE001 - 报告属于可选产出
        logger.warning("持仓风险分析产出失败：%s", exc)
        return None
    path = config.out_dir / "risk_report.md"
    path.write_text(text, encoding="utf-8")
    logger.info("持仓风险分析已写入 %s", path)
    return path


def _group_scores(
    scores: pl.DataFrame, start: date, end: date
) -> dict[date, pl.DataFrame]:
    """把打分表切成 ``{信号日: (date, instrument, score)}``，只保留区间内交易日。"""
    required = [DATE_COL, INSTRUMENT_COL, SCORE_COL]
    missing = [col for col in required if col not in scores.columns]
    if missing:
        raise E2EError(f"打分表缺少列：{missing}")
    window = scores.select(*required).filter(
        (pl.col(DATE_COL) >= start)
        & (pl.col(DATE_COL) <= end)
        & pl.col(SCORE_COL).is_not_null()
        & pl.col(SCORE_COL).is_finite()
    )
    groups = window.partition_by(DATE_COL, maintain_order=True, include_key=True)
    return {frame[DATE_COL][0]: frame for frame in groups}


def _window_start(calendar: pl.DataFrame, ref_date: date, trading_days: int) -> date:
    """取 ``ref_date`` 往前 ``trading_days`` 个开市日作为行情窗口起点。"""
    days = (
        calendar.filter(pl.col("is_open") & (pl.col(DATE_COL) <= ref_date))
        .sort(DATE_COL)[DATE_COL]
        .to_list()
    )
    if not days:
        return ref_date
    index = max(0, len(days) - trading_days)
    return days[index]


# ---------------------------------------------------------------------------
# 绩效指标
# ---------------------------------------------------------------------------


def compute_metrics(result: BacktestResult) -> dict[str, Any]:
    """计算区间绩效指标与成交 / 拒单 / 公司行为统计。"""
    nav = result.nav
    n_days = nav.height
    final_nav = result.final_nav
    total_return = result.total_return
    if n_days > 0 and result.initial_cash > 0.0 and final_nav > 0.0:
        annualized = (final_nav / result.initial_cash) ** (
            TRADING_DAYS_PER_YEAR / n_days
        ) - 1.0
    else:
        annualized = 0.0

    navs = nav["nav"].to_numpy() if n_days else np.array([result.initial_cash])
    if navs.size >= 2:
        prev = navs[:-1]
        rets = np.divide(
            navs[1:] - prev, prev, out=np.zeros_like(prev), where=prev > 0.0
        )
        std = float(rets.std(ddof=1)) if rets.size >= 2 else 0.0
        sharpe = float(rets.mean() / std * math.sqrt(TRADING_DAYS_PER_YEAR)) if std > 0 else 0.0
    else:
        sharpe = 0.0
    peak = np.maximum.accumulate(navs) if navs.size else navs
    drawdown = navs / peak - 1.0 if navs.size else navs
    max_drawdown = float(drawdown.min()) if drawdown.size else 0.0
    total_turnover = float(nav["turnover"].sum()) if n_days else 0.0

    reasons = Counter(str(rej.reason) for rej in result.rejects)
    return {
        "trading_days": n_days,
        "initial_cash": result.initial_cash,
        "final_nav": final_nav,
        "total_return": total_return,
        "annualized_return": annualized,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "total_turnover": total_turnover,
        "n_fills": len(result.fills),
        "n_rejects": len(result.rejects),
        "reject_reasons": dict(reasons.most_common(TOP_REJECT_REASONS)),
        "n_corporate_actions": len(result.corporate_actions),
        "n_signal_calls": result.nav.height,
    }


# ---------------------------------------------------------------------------
# 落盘
# ---------------------------------------------------------------------------


def _write_outputs(
    config: E2EConfig,
    result: BacktestResult,
    stats: SignalStats,
    train_config: TrainConfig | None,
    close_panel: pl.DataFrame,
    benchmark_summary: BenchmarkSummary | None = None,
    benchmark_series: pl.DataFrame | None = None,
    universe_mean_daily: float = 0.0,
    strategy_config: StrategyConfig | None = None,
    enhanced_stats: EnhancedStats | None = None,
) -> tuple[Path, Path, Path, Path, Path | None]:
    out_dir = config.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    nav_path = out_dir / "nav.parquet"
    report_path = out_dir / "report.md"
    states_path = out_dir / "account_states.md"
    png_path = out_dir / "nav.png"
    benchmark_path = out_dir / "benchmark.parquet" if benchmark_series is not None else None

    result.nav.write_parquet(nav_path)
    if benchmark_path is not None and benchmark_series is not None:
        benchmark_series.write_parquet(benchmark_path)
    metrics = compute_metrics(result)
    report_path.write_text(
        render_report(
            config,
            result,
            stats,
            metrics,
            train_config,
            benchmark_summary,
            universe_mean_daily,
            strategy_config,
            enhanced_stats,
        ),
        encoding="utf-8",
    )
    states_path.write_text(
        render_account_states(config, result, close_panel), encoding="utf-8"
    )
    _draw_nav(result, png_path, benchmark_series)
    return nav_path, report_path, states_path, png_path, benchmark_path


def _fmt_pct(value: float | None) -> str:
    """百分比格式化，None 输出 ``-``。"""
    return "-" if value is None else f"{value:.4%}"


def _fmt_num(value: float | None, digits: int = 4) -> str:
    """数值格式化，None 输出 ``-``。"""
    return "-" if value is None else f"{value:.{digits}f}"


def _render_benchmark_rows(
    config: E2EConfig, benchmark: BenchmarkSummary
) -> list[str]:
    """绩效表里的基准 / 超额行。"""
    return [
        f"| 基准指数 | {config.benchmark} |",
        f"| 基准 / 超额对齐交易日数 | {benchmark.n_days} |",
        f"| 基准区间收益 | {_fmt_pct(benchmark.benchmark_total_return)} |",
        f"| 基准年化（几何） | {_fmt_pct(benchmark.benchmark_annualized)} |",
        "| 超额区间收益（组合归一 / 基准归一 − 1） | "
        f"{_fmt_pct(benchmark.excess_total_return)} |",
        "| 超额年化（日超额均值 × 252） | "
        f"{_fmt_pct(benchmark.excess_annualized)} |",
        "| 跟踪误差（超额日收益 std × √252） | "
        f"{_fmt_pct(benchmark.tracking_error)} |",
        "| 信息比率（超额年化 / 跟踪误差） | "
        f"{_fmt_num(benchmark.information_ratio)} |",
    ]


def render_report(
    config: E2EConfig,
    result: BacktestResult,
    stats: SignalStats,
    metrics: dict[str, Any],
    train_config: TrainConfig | None,
    benchmark: BenchmarkSummary | None = None,
    universe_mean_daily: float = 0.0,
    strategy_config: StrategyConfig | None = None,
    enhanced_stats: EnhancedStats | None = None,
) -> str:
    """渲染 ``report.md``：头部配置 + 绩效（含基准 / 超额）+ 成交 / 拒单 / 公司行为统计。

    ``index_enhanced`` 时追加「指增核对」段（覆盖度、个股带、放松轮次、行业快照退化）。
    """
    lines: list[str] = []
    lines.append("# 端到端回测报告（issue #16）")
    lines.append("")
    lines.append("## 配置")
    lines.append("")
    lines.append("| 项 | 值 |")
    lines.append("| --- | --- |")
    rows = [
        ("data_dir", str(config.data_dir)),
        ("model_dir", str(config.model_dir)),
        ("train_config", str(config.resolved_train_config_path)),
        ("factor_library", str(config.factor_library_dir)),
        ("start", config.start.isoformat()),
        ("end", config.end.isoformat()),
        ("horizon", str(config.horizon)),
        ("initial_cash", f"{config.initial_cash:.2f}"),
        ("top_k", str(config.top_k)),
        ("lookback_days", str(config.lookback_days)),
        ("strategy", config.strategy),
        ("rebalance_freq", config.rebalance_freq),
        ("benchmark", config.benchmark or "（未指定）"),
        ("universe", config.universe or "（全市场）"),
        (
            "universe_mean_daily_instruments",
            f"{universe_mean_daily:.2f}" if config.universe is not None else "-",
        ),
        ("out_dir", str(config.out_dir)),
    ]
    if strategy_config is not None and strategy_config.optimize:
        rows.append(
            (
                "strategy.optimize",
                json.dumps(strategy_config.optimize, ensure_ascii=False),
            )
        )
    if train_config is not None:
        rows.extend(
            [
                ("train.start", str(train_config.start)),
                ("train.end", str(train_config.end)),
                ("train.n_rows", str(train_config.n_rows)),
                ("train.presets", train_config.presets),
                ("train.time_limit", f"{train_config.time_limit:.0f}"),
                ("train.universe", train_config.universe or "（全市场）"),
                (
                    "train.feature_columns",
                    ", ".join(train_config.feature_columns),
                ),
            ]
        )
    for key, value in rows:
        lines.append(f"| {key} | {value} |")
    lines.append("")

    lines.append("## 绩效")
    lines.append("")
    lines.append("| 指标 | 值 |")
    lines.append("| --- | --- |")
    lines.append(f"| 交易日数 | {metrics['trading_days']} |")
    lines.append(f"| 初始资金 | {metrics['initial_cash']:.2f} |")
    lines.append(f"| 期末净值 | {metrics['final_nav']:.2f} |")
    lines.append(f"| 总收益 | {metrics['total_return']:.4%} |")
    lines.append(f"| 年化收益 | {metrics['annualized_return']:.4%} |")
    lines.append(f"| 最大回撤 | {metrics['max_drawdown']:.4%} |")
    lines.append(f"| 夏普（日频 √252，rf=0） | {metrics['sharpe']:.4f} |")
    lines.append(f"| 总换手（日换手之和） | {metrics['total_turnover']:.4f} |")
    lines.append(f"| 成交笔数 | {metrics['n_fills']} |")
    lines.append(f"| 拒单笔数 | {metrics['n_rejects']} |")
    lines.append(f"| 公司行为处理笔数 | {metrics['n_corporate_actions']} |")
    if benchmark is not None:
        lines.extend(_render_benchmark_rows(config, benchmark))
    lines.append("")

    if benchmark is not None:
        lines.append(
            "基准 nav 与组合归一净值均以期初（首个共同交易日）归一为 1.0，"
            "`excess_nav = 组合归一净值 / 基准归一净值`；`benchmark.parquet` 落盘对齐序列。"
            "净值图中基准与超额曲线乘以初始资金，与组合 nav 同轴对比。"
        )
        lines.append("")

    lines.append("### 拒单原因 Top")
    lines.append("")
    if metrics["reject_reasons"]:
        for reason, count in metrics["reject_reasons"].items():
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- 无")
    lines.append("")

    lines.append("### 信号与优化")
    lines.append("")
    lines.append(f"- 信号调用次数：{stats.calls}")
    lines.append(f"- 优化求解次数（含放宽重试）：{stats.optimize_calls}")
    lines.append(f"- 打分全空日：{stats.empty_score_days}")
    lines.append(f"- 放宽换手尝试 / 成功：{stats.relaxed_attempts} / {stats.relaxed_success}")
    lines.append(f"- 保持现状降级次数：{stats.hold_fallback}")
    lines.append(f"- 生成订单笔数合计：{stats.orders}")
    lines.append("")

    if enhanced_stats is not None:
        lines.extend(_render_enhanced_rows(strategy_config, enhanced_stats))

    if stats.notes:
        lines.append("### 降级与警告（最多 20 条）")
        lines.append("")
        for note in stats.notes[:20]:
            lines.append(f"- {note}")
        if len(stats.notes) > 20:
            lines.append(f"- ...（其余 {len(stats.notes) - 20} 条略）")
        lines.append("")
    return "\n".join(lines) + "\n"


def _render_enhanced_rows(
    strategy_config: StrategyConfig | None, enhanced: EnhancedStats
) -> list[str]:
    """指增核对段：覆盖度 / 个股带 / 放松轮次 / 行业快照退化（issue #71）。"""
    lines: list[str] = ["### 指增核对（index_enhanced）", ""]
    band = 0.0
    cover_min = 0.0
    if strategy_config is not None and strategy_config.optimize:
        band = float(strategy_config.optimize.get("stock_band", 0.0))
        cover_min = float(strategy_config.optimize.get("cover_rate_min", 0.0))
    lines.append(
        f"- 调仓日 / 非调仓日：{enhanced.rebalance_days} / {enhanced.non_rebalance_days}"
    )
    lines.append(f"- 优化失败保持持仓日数：{enhanced.held_days}")
    lines.append(f"- 终局台阶触发日数（取消换手约束后求解成功）：{enhanced.final_tier_days}")
    lines.append(f"- 基准权重缺失日数：{enhanced.bench_missing_days}")
    rounds = "、".join(
        f"{count} 轮 {days} 日" for count, days in sorted(enhanced.relax_rounds.items())
    )
    lines.append(f"- 放松轮次分布：{rounds or '（无调仓日）'}")
    if enhanced.cover_rate_min is None:
        lines.append("- 最小成分覆盖度：-")
    else:
        lines.append(
            f"- 最小成分覆盖度：{enhanced.cover_rate_min:.4f}"
            f"（{enhanced.cover_rate_min_date}，下限 {cover_min:.2f}）"
        )
    lines.append(
        f"- 个股带最大偏离：{enhanced.max_band_dev:.6f}"
        f"（{enhanced.max_band_dev_date}，带 ±{band:.4f} + 容差 {BAND_CHECK_ABS_SLACK:.0e}）"
    )
    lines.append(f"- 个股带超带累计只次数：{enhanced.band_violations}")
    lines.append(
        f"- 行业快照退化日数（无 PIT 快照，退化到最新一份）：{enhanced.industry_fallback_days}"
    )
    lines.append(
        "- 口径：覆盖度 / 带偏离按**归一后的当日持仓**计（求解日 = 目标权重，held 日 = 漂移后持仓），"
        "`enhanced_log.parquet` 另记 `invested_ratio`（实际已投资比例）。"
    )
    lines.append("")
    if enhanced.notes:
        lines.append("#### 指增 warning（最多 20 条）")
        lines.append("")
        for note in enhanced.notes[:20]:
            lines.append(f"- {note}")
        if len(enhanced.notes) > 20:
            lines.append(f"- ...（其余 {len(enhanced.notes) - 20} 条略）")
        lines.append("")
    return lines


def render_account_states(
    config: E2EConfig, result: BacktestResult, close_panel: pl.DataFrame
) -> str:
    """渲染 ``account_states.md``：抽 3 个有成交交易日给出账户明细供人工核对。"""
    lines: list[str] = ["# 账户状态抽样核对（issue #16）", ""]
    fill_days = sorted({fill.date for fill in result.fills})
    if not fill_days:
        lines.append("区间内没有成交，无法抽样核对。")
        lines.append("")
        return "\n".join(lines)

    sample = _sample_days(fill_days, SAMPLE_DAYS)
    nav_rows = {row["date"]: row for row in result.nav.iter_rows(named=True)}
    nav_dates = [row["date"] for row in result.nav.iter_rows(named=True)]
    nav_index = {day: index for index, day in enumerate(nav_dates)}
    positions = _reconstruct_positions(
        nav_dates,
        result.fills,
        result.corporate_actions,
    )
    fills_by_day: dict[date, list[Any]] = {}
    for fill in result.fills:
        fills_by_day.setdefault(fill.date, []).append(fill)
    rejects_by_day: dict[date, list[Any]] = {}
    for reject in result.rejects:
        rejects_by_day.setdefault(reject.date, []).append(reject)

    lines.append(
        "持仓由区间内全部成交与公司行为回放重建，现金 / 市值 / nav 取自 `nav.parquet`；"
        "总市值应等于 `nav.parquet` 的 market_value（差异来自停牌前向填充价与分红送转时点）。"
        "每日附「勾稽」三项：现金变动 vs 成交流水、持仓明细合计 vs 市值、现金 + 市值 vs nav。"
    )
    lines.append("")
    for day in sample:
        row = nav_rows.get(day)
        if row is None:
            continue
        lines.append(f"## {day.isoformat()}")
        lines.append("")
        if row is not None:
            lines.append(
                f"- 现金：{row['cash']:.2f}    持仓市值：{row['market_value']:.2f}    "
                f"nav：{row['nav']:.2f}    当日成交额：{row['traded_amount']:.2f}    "
                f"换手：{row['turnover']:.4f}"
            )
        lines.append("")
        lines.append("### 持仓明细")
        lines.append("")
        lines.append("| 证券 | 股数 | 收盘价 | 市值 |")
        lines.append("| --- | ---: | ---: | ---: |")
        prices = _close_on(close_panel, day)
        held = positions.get(day, {})
        total_value = 0.0
        for instrument in sorted(held):
            volume = held[instrument]
            price = prices.get(instrument, 0.0)
            value = volume * price
            total_value += value
            lines.append(f"| {instrument} | {volume} | {price:.4f} | {value:.2f} |")
        if not held:
            lines.append("| （空仓） | 0 | - | 0.00 |")
        lines.append(f"| **合计** |  |  | **{total_value:.2f}** |")
        lines.append("")

        lines.append("### 当日成交")
        lines.append("")
        day_fills = fills_by_day.get(day, [])
        if day_fills:
            lines.append("| 证券 | 方向 | 股数 | 成交价 | 现金费用 | 成本合计 |")
            lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
            for fill in day_fills:
                lines.append(
                    f"| {fill.instrument} | {fill.side} | {fill.volume} | "
                    f"{fill.price:.4f} | {fill.fee.cash_cost:.2f} | {fill.fee.total:.2f} |"
                )
        else:
            lines.append("（无）")
        lines.append("")

        lines.extend(
            _render_reconciliation(
                day,
                row,
                day_fills,
                [detail for detail in result.corporate_actions if detail.date == day],
                total_value,
                nav_dates,
                nav_index,
                nav_rows,
            )
        )

        lines.append("### 当日拒单")
        lines.append("")
        day_rejects = rejects_by_day.get(day, [])
        if day_rejects:
            lines.append("| 证券 | 方向 | 意愿股数 | 原因 | 说明 |")
            lines.append("| --- | --- | ---: | --- | --- |")
            for reject in day_rejects:
                lines.append(
                    f"| {reject.instrument} | {reject.side} | {reject.requested} | "
                    f"{reject.reason} | {reject.detail} |"
                )
        else:
            lines.append("（无）")
        lines.append("")
    return "\n".join(lines)


def _render_reconciliation(
    day: date,
    row: dict[str, Any],
    day_fills: list[Any],
    day_actions: list[Any],
    total_value: float,
    nav_dates: list[date],
    nav_index: dict[date, int],
    nav_rows: dict[date, dict[str, Any]],
) -> list[str]:
    """逐日勾稽：现金变动 vs 成交流水 + 分红现金、持仓明细合计 vs ``nav.parquet`` 市值。"""
    lines: list[str] = ["### 勾稽", ""]
    dividends = sum(float(detail.cash_received) for detail in day_actions)
    position = nav_index.get(day)
    if position is None or position == 0:
        lines.append("- 现金勾稽：区间首日无前一日现金，跳过。")
    else:
        previous = nav_rows[nav_dates[position - 1]]
        actual = float(row["cash"]) - float(previous["cash"])
        trades = sum(
            (fill.notional if fill.side == ORDER_SIDE_SELL else -fill.notional)
            - fill.fee.cash_cost
            for fill in day_fills
        )
        lines.append(
            f"- 现金勾稽：现金_t − 现金_(t−1) = {actual:.2f}；"
            f"成交流水（卖 − 买 − 现金费用）＝ {trades:.2f}；"
            f"公司行为现金 ＝ {dividends:.2f}；差 {actual - trades - dividends:.4f}"
        )
    diff = total_value - float(row["market_value"])
    lines.append(
        f"- 市值勾稽：持仓明细合计 {total_value:.2f} vs nav.parquet 市值 "
        f"{float(row['market_value']):.2f}；差 {diff:.4f}"
    )
    lines.append(
        f"- nav 勾稽：现金 + 市值 = {float(row['cash']) + float(row['market_value']):.2f} "
        f"vs nav {float(row['nav']):.2f}；差 "
        f"{float(row['cash']) + float(row['market_value']) - float(row['nav']):.4f}"
    )
    lines.append("")
    return lines


def _sample_days(days: list[date], count: int) -> list[date]:
    """在有成交的交易日里取首、中、尾（不足时去重后全取）。"""
    if count <= 0:
        return []
    if len(days) <= count:
        return list(days)
    if count == 1:
        return [days[0]]
    indices = sorted({0, len(days) // 2, len(days) - 1})
    picked = [days[index] for index in indices]
    return picked[:count]


def _reconstruct_positions(
    nav_dates: list[date],
    fills: list[Any],
    corporate_actions: list[Any],
) -> dict[date, dict[str, int]]:
    """按日回放成交与公司行为，重建每日收盘持仓股数。"""
    fills_by_day: dict[date, list[Any]] = {}
    for fill in fills:
        fills_by_day.setdefault(fill.date, []).append(fill)
    ca_by_day: dict[date, list[Any]] = {}
    for detail in corporate_actions:
        ca_by_day.setdefault(detail.date, []).append(detail)

    positions: dict[str, int] = {}
    snapshots: dict[date, dict[str, int]] = {}
    for day in nav_dates:
        for fill in fills_by_day.get(day, []):
            if fill.side == ORDER_SIDE_BUY:
                positions[fill.instrument] = positions.get(fill.instrument, 0) + fill.volume
            else:
                remaining = positions.get(fill.instrument, 0) - fill.volume
                if remaining > 0:
                    positions[fill.instrument] = remaining
                else:
                    positions.pop(fill.instrument, None)
        # 公司行为在 T+1 日终发生（成交之后），直接采用处理后的股数。
        for detail in ca_by_day.get(day, []):
            if detail.volume_after > 0:
                positions[detail.instrument] = detail.volume_after
            else:
                positions.pop(detail.instrument, None)
        snapshots[day] = dict(positions)
    return snapshots


def build_holdings_weights(
    result: BacktestResult, close_panel: pl.DataFrame
) -> pl.DataFrame:
    """从回测成交流水重建逐日持仓权重 ``(date, instrument, weight)``。

    权重按当日已定价持仓的市值归一（停牌无价持仓不计入，避免把现金混进行业 / 风格
    暴露）。输出落 ``holdings.parquet``，供 ``scripts/risk_report.py`` 消费。
    """
    nav_dates = [row["date"] for row in result.nav.iter_rows(named=True)]
    positions = _reconstruct_positions(nav_dates, result.fills, result.corporate_actions)
    rows: list[dict[str, object]] = []
    for day in nav_dates:
        held = positions.get(day, {})
        if not held:
            continue
        prices = _close_on(close_panel, day)
        values = {
            instrument: volume * prices[instrument]
            for instrument, volume in held.items()
            if prices.get(instrument, 0.0) > 0.0
        }
        total = float(sum(values.values()))
        if total <= 0.0:
            continue
        rows.extend(
            {"date": day, "instrument": instrument, "weight": value / total}
            for instrument, value in sorted(values.items())
        )
    return pl.DataFrame(
        rows,
        schema={"date": pl.Date, "instrument": pl.String, "weight": pl.Float64},
    )


def _draw_nav(
    result: BacktestResult, path: Path, benchmark_series: pl.DataFrame | None = None
) -> None:
    """画净值曲线（Agg 后端，不弹窗）；给了基准序列则叠加基准与超额曲线。

    基准 / 超额都以 ``× 初始资金`` 与组合 nav 同轴对比。
    """
    write_nav_png(result.nav, result.initial_cash, path, benchmark_series)


def write_nav_png(
    nav: pl.DataFrame,
    initial_cash: float,
    path: Path,
    benchmark_series: pl.DataFrame | None = None,
) -> None:
    """把净值 / 基准 / 超额三条曲线画到 ``path``（口径见 :func:`_draw_nav`）。"""
    if nav.height == 0:
        figure, axes = plt.subplots(figsize=(10, 4.5))
        axes.set_title("NAV (empty)")
        figure.savefig(path, dpi=PNG_DPI, bbox_inches="tight")
        plt.close(figure)
        return
    dates = nav[DATE_COL].to_list()
    values = nav["nav"].to_list()
    figure, axes = plt.subplots(figsize=(10, 4.5))
    axes.plot(dates, values, linewidth=1.2, color="#1f77b4", label="portfolio")
    if benchmark_series is not None and benchmark_series.height:
        benchmark_dates = benchmark_series[DATE_COL].to_list()
        axes.plot(
            benchmark_dates,
            [value * initial_cash for value in benchmark_series[BENCHMARK_NAV_COL].to_list()],
            linewidth=1.0,
            color="#ff7f0e",
            label="benchmark (scaled to initial cash)",
        )
        if EXCESS_NAV_COL in benchmark_series.columns:
            axes.plot(
                benchmark_dates,
                [
                    value * initial_cash
                    for value in benchmark_series[EXCESS_NAV_COL].to_list()
                ],
                linewidth=1.0,
                color="#2ca02c",
                label="excess (scaled to initial cash)",
            )
    axes.legend(loc="best")
    axes.set_title("E2E Backtest NAV")
    axes.set_ylabel("NAV (CNY)")
    axes.grid(True, alpha=0.3)
    figure.autofmt_xdate()
    figure.savefig(path, dpi=PNG_DPI, bbox_inches="tight")
    plt.close(figure)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_date(text: str) -> date:
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"日期格式应为 YYYY-MM-DD：{text!r}") from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="端到端回测（M1）：模型打分 → 组合 → 账户")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="缓存目录，默认 data/")
    parser.add_argument(
        "--model-dir", default=DEFAULT_MODEL_DIR, help=f"模型目录，默认 {DEFAULT_MODEL_DIR}"
    )
    parser.add_argument("--start", type=_parse_date, required=True, help="回测起始日 YYYY-MM-DD")
    parser.add_argument("--end", type=_parse_date, required=True, help="回测结束日 YYYY-MM-DD")
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=DEFAULT_INITIAL_CASH,
        help=f"初始资金，默认 {DEFAULT_INITIAL_CASH:.0f}",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help=f"候选数，默认 {DEFAULT_TOP_K}")
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=DEFAULT_LOOKBACK_DAYS,
        help=f"收益 / 因子回溯交易日数，默认 {DEFAULT_LOOKBACK_DAYS}",
    )
    parser.add_argument(
        "--out-dir", default=DEFAULT_OUT_DIR, help=f"产出目录，默认 {DEFAULT_OUT_DIR}"
    )
    parser.add_argument(
        "--factor-library",
        default=str(DEFAULT_FACTOR_LIBRARY_DIR),
        help=f"因子库目录，默认 {DEFAULT_FACTOR_LIBRARY_DIR}",
    )
    parser.add_argument(
        "--horizon", type=int, default=DEFAULT_HORIZON, help=f"标签持有期，默认 {DEFAULT_HORIZON}"
    )
    parser.add_argument(
        "--train-config",
        default=None,
        help="训练配置 JSON 路径，默认取 model-dir 同级约定",
    )
    parser.add_argument(
        "--benchmark",
        default=None,
        help="基准指数代码（六位，如 000905），给了就输出基准 / 超额绩效与净值叠加",
    )
    parser.add_argument(
        "--risk-report",
        action="store_true",
        help="额外产出持仓风险分析四表 risk_report.md（issue #69）",
    )
    parser.add_argument(
        "--universe",
        default=None,
        help="股票池：命名池名（hs300/zz500/zz1000/zz2000）或自定义池文件路径；须与训练一致",
    )
    parser.add_argument(
        "--strategy",
        default=DEFAULT_STRATEGY,
        choices=list(STRATEGIES),
        help=f"策略：{list(STRATEGIES)}，默认 {DEFAULT_STRATEGY}",
    )
    parser.add_argument(
        "--rebalance-freq",
        default=DEFAULT_REBALANCE_FREQ,
        choices=list(REBALANCE_FREQS),
        help=f"指增调仓频率，默认 {DEFAULT_REBALANCE_FREQ}",
    )
    parser.add_argument(
        "--strategy-config",
        default=None,
        help="策略配置 JSON 路径（issue #70 格式）；与命令行同名参数须一致",
    )
    return parser


def _print_summary(result: E2EResult) -> None:
    metrics = result.metrics
    print("\n# 端到端回测完成")
    print(f"区间：{result.config.start} ~ {result.config.end}（策略 {result.config.strategy}）")
    print(f"期末净值 {metrics['final_nav']:.2f}    总收益 {metrics['total_return']:.4%}")
    print(
        f"年化 {metrics['annualized_return']:.4%}    最大回撤 {metrics['max_drawdown']:.4%}    "
        f"夏普 {metrics['sharpe']:.4f}"
    )
    print(
        f"成交 {metrics['n_fills']} 笔    拒单 {metrics['n_rejects']} 笔    "
        f"公司行为 {metrics['n_corporate_actions']} 笔"
    )
    if metrics["reject_reasons"]:
        top = ", ".join(f"{k}={v}" for k, v in metrics["reject_reasons"].items())
        print(f"拒单原因：{top}")
    if result.benchmark is not None:
        bench = result.benchmark
        print(
            f"基准 {result.config.benchmark}：年化 {_fmt_pct(bench.benchmark_annualized)}    "
            f"超额年化 {_fmt_pct(bench.excess_annualized)}    "
            f"跟踪误差 {_fmt_pct(bench.tracking_error)}    "
            f"IR {_fmt_num(bench.information_ratio)}"
        )
    print(
        f"优化：求解 {result.stats.optimize_calls} 次，放宽成功 {result.stats.relaxed_success} 次，"
        f"保持现状 {result.stats.hold_fallback} 次"
    )
    print(f"净值：{result.nav_path}")
    print(f"报告：{result.report_path}")
    print(f"账户抽样：{result.states_path}")
    print(f"净值图：{result.png_path}")
    if result.benchmark_path is not None:
        print(f"基准序列：{result.benchmark_path}")
    if result.holdings_path is not None:
        print(f"持仓权重：{result.holdings_path}")
    if result.risk_report_path is not None:
        print(f"风险分析：{result.risk_report_path}")
    if result.enhanced_stats is not None:
        stats = result.enhanced_stats
        cover = (
            f"{stats.cover_rate_min:.4f}"
            if stats.cover_rate_min is not None
            else "-"
        )
        print(
            f"指增：调仓 {stats.rebalance_days} 日，保持持仓 {stats.held_days} 日，"
            f"终局台阶 {stats.final_tier_days} 日，"
            f"最小覆盖度 {cover}，个股带最大偏离 {stats.max_band_dev:.6f}"
            f"（超带 {stats.band_violations} 只次），行业快照退化 {stats.industry_fallback_days} 日"
        )
    if result.enhanced_log_path is not None:
        print(f"指增日志：{result.enhanced_log_path}")


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        pass

    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    config = E2EConfig(
        data_dir=Path(args.data_dir),
        model_dir=Path(args.model_dir),
        out_dir=Path(args.out_dir),
        start=args.start,
        end=args.end,
        initial_cash=args.initial_cash,
        top_k=args.top_k,
        lookback_days=args.lookback_days,
        factor_library_dir=Path(args.factor_library),
        horizon=args.horizon,
        train_config_path=Path(args.train_config) if args.train_config else None,
        benchmark=args.benchmark,
        risk_report=args.risk_report,
        universe=args.universe,
        strategy=args.strategy,
        rebalance_freq=args.rebalance_freq,
        strategy_config_path=(
            Path(args.strategy_config) if args.strategy_config else None
        ),
    )
    result = run_e2e(config)
    _print_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
