"""端到端 M1（issue #16）：训练好的模型 → AutoGluon 打分 → 凸优化 → 账户回测。

链路位置
--------
与 ``scripts/train_baseline.py`` 配对，读取它的产物 ``<model-dir>`` 与同级训练配置
JSON，在 ``[start, end]`` 区间跑一次完整的 :class:`~quant.backtest.engine.BacktestEngine`
账户回测（T+1 开盘成交、涨跌停、整手、费用、公司行为）。

无前视的说明（关键）
--------------------
回测开始前对**整个加载窗口**一次性 ``predict``，再把打分按信号日 T 查表。这不构成
前视，因为 :func:`~quant.automl.dataset.build_dataset` 里每只票在 T 日的特征只由
``date <= T`` 的行情算出（因子含 rolling / shift，逐日截面 z-score 只用当日截面），
``predict`` 不读 ``label``。等价于「每天收盘后用当日可得数据打分」，只是把逐日调用
合并成一次批量推理。同理，收益面板与收盘价面板都按 ``date <= T`` 切片后再喂给优化器。

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
        --out-dir runs/e2e

落盘（``--out-dir``）:: nav.parquet / report.md / account_states.md / nav.png
"""
from __future__ import annotations

import argparse
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
from quant.data.cache import (  # noqa: E402
    load_bars,
    load_calendar,
    load_corporate_actions,
)
from quant.labels.open_to_open import DEFAULT_HORIZON  # noqa: E402
from quant.portfolio.optimizer import (  # noqa: E402
    InfeasibleError,
    PortfolioError,
    PortfolioOptimizer,
)
from quant.portfolio.roundlot import round_weights_to_lots  # noqa: E402
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

    @property
    def resolved_train_config_path(self) -> Path:
        """训练配置路径：显式指定优先，否则按 ``model-dir`` 的同级约定。"""
        return (
            self.train_config_path
            if self.train_config_path is not None
            else default_train_config_path(self.model_dir)
        )


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
# 运行
# ---------------------------------------------------------------------------


def run_e2e(
    config: E2EConfig,
    *,
    trainer: BaselineTrainer | None = None,
    optimizer: PortfolioOptimizer | None = None,
) -> E2EResult:
    """执行端到端回测并落盘，返回 :class:`E2EResult`。

    ``trainer`` / ``optimizer`` 用于依赖注入（测试传假对象，不加载真模型、不跑真 cvxpy）。
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

    # -- 行情窗口：start 之前留 lookback + 因子余量，供因子与收益矩阵预热 ----------
    calendar = load_calendar(config.data_dir)
    window_start = _window_start(
        calendar, config.start, config.lookback_days + FACTOR_WINDOW_MARGIN
    )
    bars = load_bars(config.data_dir, start=window_start, end=config.end)
    if bars.height == 0:
        raise E2EError(f"行情窗口 [{window_start}, {config.end}] 内没有数据")
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
        "打分覆盖 %d 个信号日，行情窗口 %s ~ %s，%d 行",
        len(score_by_day),
        window_start,
        config.end,
        bars.height,
    )

    close_panel = build_close_panel(bars)
    returns_panel = build_returns_panel(bars)

    stats = SignalStats()
    active_optimizer = optimizer if optimizer is not None else PortfolioOptimizer()
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
        bars,
        calendar,
        actions,
        signal_fn,
        config.start,
        config.end,
        config.initial_cash,
    )

    nav_path, report_path, states_path, png_path = _write_outputs(
        config, result, stats, train_config, close_panel
    )
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
    )


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
) -> tuple[Path, Path, Path, Path]:
    out_dir = config.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    nav_path = out_dir / "nav.parquet"
    report_path = out_dir / "report.md"
    states_path = out_dir / "account_states.md"
    png_path = out_dir / "nav.png"

    result.nav.write_parquet(nav_path)
    metrics = compute_metrics(result)
    report_path.write_text(
        render_report(config, result, stats, metrics, train_config), encoding="utf-8"
    )
    states_path.write_text(
        render_account_states(config, result, close_panel), encoding="utf-8"
    )
    _draw_nav(result, png_path)
    return nav_path, report_path, states_path, png_path


def render_report(
    config: E2EConfig,
    result: BacktestResult,
    stats: SignalStats,
    metrics: dict[str, Any],
    train_config: TrainConfig | None,
) -> str:
    """渲染 ``report.md``：头部配置 + 绩效 + 成交 / 拒单 / 公司行为统计。"""
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
        ("out_dir", str(config.out_dir)),
    ]
    if train_config is not None:
        rows.extend(
            [
                ("train.start", str(train_config.start)),
                ("train.end", str(train_config.end)),
                ("train.n_rows", str(train_config.n_rows)),
                ("train.presets", train_config.presets),
                ("train.time_limit", f"{train_config.time_limit:.0f}"),
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

    if stats.notes:
        lines.append("### 降级与警告（最多 20 条）")
        lines.append("")
        for note in stats.notes[:20]:
            lines.append(f"- {note}")
        if len(stats.notes) > 20:
            lines.append(f"- ...（其余 {len(stats.notes) - 20} 条略）")
        lines.append("")
    return "\n".join(lines) + "\n"


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
    positions = _reconstruct_positions(
        [row["date"] for row in result.nav.iter_rows(named=True)],
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


def _draw_nav(result: BacktestResult, path: Path) -> None:
    """画净值曲线（Agg 后端，不弹窗）。"""
    nav = result.nav
    if nav.height == 0:
        figure, axes = plt.subplots(figsize=(10, 4.5))
        axes.set_title("NAV (empty)")
        figure.savefig(path, dpi=PNG_DPI, bbox_inches="tight")
        plt.close(figure)
        return
    dates = nav[DATE_COL].to_list()
    values = nav["nav"].to_list()
    figure, axes = plt.subplots(figsize=(10, 4.5))
    axes.plot(dates, values, linewidth=1.2, color="#1f77b4")
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
    return parser


def _print_summary(result: E2EResult) -> None:
    metrics = result.metrics
    print("\n# 端到端回测完成")
    print(f"区间：{result.config.start} ~ {result.config.end}")
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
    print(
        f"优化：求解 {result.stats.optimize_calls} 次，放宽成功 {result.stats.relaxed_success} 次，"
        f"保持现状 {result.stats.hold_fallback} 次"
    )
    print(f"净值：{result.nav_path}")
    print(f"报告：{result.report_path}")
    print(f"账户抽样：{result.states_path}")
    print(f"净值图：{result.png_path}")


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
    )
    result = run_e2e(config)
    _print_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
