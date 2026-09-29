"""每日生产跑批主流程：T 日收盘 → 调仓单（架构 3.5 生产侧）。

时序（数据已由 ``scripts/daily_update.py`` 更新完）
--------------------------------------------------
1. 解析信号日 ``T``（最近开市日），加载 :class:`quant.daily.virtual_account.VirtualAccount`；
2. 跑批前体检 :func:`quant.data.validate.validate`，不过即中止（纪律第 6 条）；
3. 加载行情窗口：最近 ``lookback_days`` 个交易日 + 因子最长窗口余量；
4. 重算 ``factor_library`` 下全部因子 → :func:`quant.automl.dataset.build_dataset`
   → :meth:`quant.automl.trainer.BaselineTrainer.predict` 得 T 日截面打分；
5. 打分 top-K（并上当前持仓）进 :class:`quant.portfolio.optimizer.PortfolioOptimizer`，
   ``w_prev`` 由虚拟账户持仓按最近收盘价折权重；
6. :func:`quant.portfolio.roundlot.round_weights_to_lots` 转目标股数，
   与当前持仓 diff 出调仓订单；
7. 用 T 日涨跌停 / 停牌预过滤明显示不可执行的订单（**仅预判，真实成交在 T+1**）；
8. 落盘调仓单与报告，标记账户跑批日（同日重复跑幂等）。

无前视
------
因子与打分只用 ``date <= T`` 的行情；订单参考价为 T 日收盘（T+1 开盘价此时未知），
真实成交价与费用由 T+1 人工跟单后通过
:meth:`~quant.daily.virtual_account.VirtualAccount.apply_actual_fills` 回录。

降级策略
--------
优化不可行（:class:`~quant.portfolio.optimizer.InfeasibleError`）时：
先放宽 ``max_turnover`` 到 :data:`MAX_TURNOVER_RELAXED` 重试一次；仍不可行则保持现有持仓，
订单为空，并在报告里以 ``hold_fallback=True`` 显著标记。
"""
from __future__ import annotations

import dataclasses
import importlib
import json
import logging
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from quant.automl.dataset import (
    DATE_COL,
    INSTRUMENT_COL,
    FactorCompute,
    build_dataset,
)
from quant.automl.rolling import (
    DEFAULT_REGISTRY_DIR,
    RollingConfig,
    TrainerFactory,
    TrainerLoader,
    resolve_model,
)
from quant.automl.trainer import DEFAULT_MODEL_DIR, SCORE_COL, BaselineTrainer
from quant.backtest.broker import PRICE_EPSILON
from quant.data.cache import load_bars, load_calendar
from quant.data.schema import board_of
from quant.data.validate import validate
from quant.daily.briefing import (
    build_briefing_risk_report,
    recent_factor_ic,
    write_briefing,
)
from quant.daily.strategy import (
    DEFAULT_STRATEGY,
    StrategyConfig,
    build_enhanced_optimizer,
    build_stock_optimizer,
    is_rebalance_day,
)
from quant.daily.virtual_account import (
    DEFAULT_ACCOUNT_DIR,
    DEFAULT_ACCOUNT_NAME,
    DEFAULT_INITIAL_CASH,
    VirtualAccount,
)
from quant.portfolio.enhanced import (
    STATUS_HELD,
    STATUS_RELAXED,
    EnhancedOptimizer,
)
from quant.portfolio.enhanced_inputs import (
    benchmark_weights,
    float_mv_frame,
    float_mv_map,
    industry_map,
    industry_table,
    returns_matrix,
    style_frame,
    style_map,
)
from quant.portfolio.optimizer import (
    InfeasibleError,
    OptimizeResult,
    PortfolioError,
    PortfolioOptimizer,
)
from quant.portfolio.roundlot import round_weights_to_lots
from quant.universe.members import members

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 配置（默认值集中在此）
# ---------------------------------------------------------------------------

#: 打分进入组合的候选证券数（top-K）。
TOP_K_DEFAULT: int = 50

#: 行情窗口的交易日数（因子与协方差回看）。
LOOKBACK_DAYS_DEFAULT: int = 250

#: 因子最长窗口余量（交易日）：在 ``lookback_days`` 之外多取，供因子 rolling / shift 预热。
FACTOR_WINDOW_MARGIN: int = 60

#: 收益协方差最多使用的交易日数。
RETURNS_WINDOW_DEFAULT: int = 250

#: 降级时放宽后的双边换手上限（等价于取消换手约束）。
MAX_TURNOVER_RELAXED: float = 1.0

#: 报告里列出的打分条数。
TOP_SCORES_IN_REPORT: int = 10

#: 因子库目录（相对仓库根）。
DEFAULT_FACTOR_LIBRARY_DIR: Path = Path("factor_library")

#: 调仓单与报告落盘目录（``runs/`` 不入 git）。
DEFAULT_ORDERS_DIR: Path = Path("runs/orders")
DEFAULT_REPORTS_DIR: Path = Path("runs/reports")

#: 调仓单表 schema（``side`` 为 buy / sell；``ref_price`` 为 T 日收盘参考价）。
ORDER_SCHEMA: pl.Schema = pl.Schema(
    {
        "date": pl.Date,
        "instrument": pl.String,
        "board": pl.String,
        "side": pl.String,
        "volume": pl.Int64,
        "ref_price": pl.Float64,
        "est_amount": pl.Float64,
        "current_volume": pl.Int64,
        "target_volume": pl.Int64,
    }
)

#: 被预过滤订单的表 schema（在调仓单基础上加 ``reason``）。
FILTERED_SCHEMA: pl.Schema = pl.Schema({**ORDER_SCHEMA, "reason": pl.String})

#: 目标权重 → 股数的中间表 schema（即 roundlot 输出）。
#: 见 :data:`quant.portfolio.roundlot.ORDERS_SCHEMA`。

#: 打分表 schema。
SCORES_SCHEMA: pl.Schema = pl.Schema(
    {DATE_COL: pl.Date, INSTRUMENT_COL: pl.String, SCORE_COL: pl.Float64}
)

#: 账户持仓 diff 的中间表 schema。
DIFF_SCHEMA: pl.Schema = pl.Schema(
    {
        "instrument": pl.String,
        "side": pl.String,
        "volume": pl.Int64,
    }
)


# ---------------------------------------------------------------------------
# 异常与报告
# ---------------------------------------------------------------------------


class DailyError(Exception):
    """每日跑批无法继续。"""


@dataclass
class DailyReport:
    """一次每日跑批的产出。

    Attributes
    ----------
    date:
        信号日 T。
    already_ran:
        该信号日此前已跑过（账户内已有 ``last_pipeline_date >= T``），本次跳过。
    relaxed:
        优化首次不可行，放宽换手约束后成功。
    hold_fallback:
        放宽后仍不可行，保持现有持仓（订单为空）。
    scores:
        T 日全部候选证券的截面打分 ``(date, instrument, score)``。
    top_scores:
        打分 top :data:`TOP_SCORES_IN_REPORT`，``[(instrument, score), ...]``。
    target_weights:
        优化给出的目标权重。
    target_volumes:
        整手取整后的目标股数。
    orders / filtered:
        可执行调仓单 / 被预过滤的订单（含 ``reason``）。
    holdings:
        当前持仓摘要（列见 :data:`quant.daily.virtual_account.HOLDINGS_SCHEMA`）。
    exposure:
        敞口摘要：现金、持仓市值、nav、现金比例、持仓比例、持仓只数。
    briefing_path:
        每日 markdown 简报的落盘路径；``dry_run`` 或幂等跳过时为 ``None``。
    """

    date: date
    account_name: str
    nav: float = 0.0
    cash: float = 0.0
    market_value: float = 0.0
    strategy: str = DEFAULT_STRATEGY
    universe: str | None = None
    benchmark: str | None = None
    rebalance_freq: str = "D"
    already_ran: bool = False
    relaxed: bool = False
    hold_fallback: bool = False
    n_candidates: int = 0
    scores: pl.DataFrame = field(
        default_factory=lambda: pl.DataFrame(schema=SCORES_SCHEMA)
    )
    top_scores: list[tuple[str, float]] = field(default_factory=list)
    target_weights: dict[str, float] = field(default_factory=dict)
    current_weights: dict[str, float] = field(default_factory=dict)
    target_volumes: dict[str, int] = field(default_factory=dict)
    orders: pl.DataFrame = field(
        default_factory=lambda: pl.DataFrame(schema=ORDER_SCHEMA)
    )
    filtered: pl.DataFrame = field(
        default_factory=lambda: pl.DataFrame(schema=FILTERED_SCHEMA)
    )
    holdings: pl.DataFrame = field(default_factory=pl.DataFrame)
    exposure: dict[str, float] = field(default_factory=dict)
    turnover: float = 0.0
    notes: list[str] = field(default_factory=list)
    orders_path: Path | None = None
    orders_csv_path: Path | None = None
    report_path: Path | None = None
    briefing_path: Path | None = None


# ---------------------------------------------------------------------------
# 纯函数：订单 diff 与过滤
# ---------------------------------------------------------------------------


def diff_orders(
    target_volumes: Mapping[str, int], current_volumes: Mapping[str, int]
) -> pl.DataFrame:
    """目标股数 − 当前持仓 → 调仓订单。

    逐票取 ``delta = target − current``：正数生成买单，负数生成卖单，0 不出单。
    ``side == "sell"`` 的行排在 ``buy`` 之前，让卖出回款可用于同日买入。
    停牌、涨跌停等可执行性判断不在本函数内。
    """
    instruments = sorted(set(target_volumes) | set(current_volumes))
    rows: list[dict[str, object]] = []
    for instrument in instruments:
        target = int(target_volumes.get(instrument, 0))
        current = int(current_volumes.get(instrument, 0))
        delta = target - current
        if delta == 0:
            continue
        rows.append(
            {
                "instrument": instrument,
                "side": "buy" if delta > 0 else "sell",
                "volume": abs(delta),
            }
        )
    if not rows:
        return pl.DataFrame(schema=DIFF_SCHEMA)
    frame = pl.DataFrame(rows, schema=DIFF_SCHEMA)
    order = pl.col("side").replace_strict({"sell": 0, "buy": 1}, return_dtype=pl.Int8)
    return frame.with_columns(order.alias("_order")).sort(["_order", "instrument"]).drop(
        "_order"
    )


def filter_executable(
    orders: pl.DataFrame, ref_bars: pl.DataFrame
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """用 T 日行情预过滤明显不可执行的订单。

    ``ref_bars`` 为信号日各证券行情行（``instrument`` / ``close`` / ``limit_up`` /
    ``limit_down``）。规则：

    - T 日无行情行、或收盘价缺失 / 非正 → 疑似停牌，过滤；
    - 买单 T 日收盘价触及涨停 → 过滤；
    - 卖单 T 日收盘价触及跌停 → 过滤。

    返回 ``(可执行行, 被过滤行)``，被过滤行带 ``reason``。**这只是对 T+1 开盘的预判，
    真实成交由 broker 按 T+1 开盘价与涨跌停判定。**
    """
    indexed = {
        str(row["instrument"]): row for row in ref_bars.iter_rows(named=True)
    }
    executable: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in orders.iter_rows(named=True):
        instrument = str(row["instrument"])
        side = str(row["side"])
        bar = indexed.get(instrument)
        reason: str | None = None
        if bar is None:
            reason = "suspended"
        else:
            close = _finite(bar.get("close"))
            if close <= 0.0:
                reason = "suspended"
            elif side == "buy":
                limit_up = _finite_or_none(bar.get("limit_up"))
                if limit_up is not None and close >= limit_up - PRICE_EPSILON:
                    reason = "limit_up"
            else:
                limit_down = _finite_or_none(bar.get("limit_down"))
                if limit_down is not None and close <= limit_down + PRICE_EPSILON:
                    reason = "limit_down"
        if reason is None:
            executable.append(dict(row))
        else:
            rejected.append({**row, "reason": reason})
    return executable, rejected


# ---------------------------------------------------------------------------
# 因子加载
# ---------------------------------------------------------------------------


def discover_factors(factor_library_dir: str | Path) -> dict[str, FactorCompute]:
    """动态加载因子库下全部 ``*.py``，返回有序 ``{因子名: compute}``。

    文件名即因子名，忽略下划线开头的文件。导入前把因子库父目录加入 ``sys.path``，
    使 ``importlib`` 能按包路径加载。
    """
    root = Path(factor_library_dir)
    if not root.is_dir():
        raise DailyError(f"因子库目录不存在：{root}")
    parent = str(root.resolve().parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    mapping: dict[str, FactorCompute] = {}
    for path in sorted(root.glob("*.py")):
        name = path.stem
        if name.startswith("_"):
            continue
        module = importlib.import_module(f"{root.name}.{name}")
        compute = getattr(module, "compute", None)
        if compute is None or not callable(compute):
            raise DailyError(f"因子 {name!r} 未定义 compute(data)")
        mapping[name] = compute
    if not mapping:
        raise DailyError(f"因子库目录下没有可用因子：{root}")
    return mapping


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def run_daily(
    data_dir: str | Path,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    *,
    account_name: str = DEFAULT_ACCOUNT_NAME,
    ref_date: date | None = None,
    top_k: int = TOP_K_DEFAULT,
    lookback_days: int = LOOKBACK_DAYS_DEFAULT,
    factor_library_dir: str | Path = DEFAULT_FACTOR_LIBRARY_DIR,
    account_dir: str | Path = DEFAULT_ACCOUNT_DIR,
    orders_dir: str | Path = DEFAULT_ORDERS_DIR,
    reports_dir: str | Path = DEFAULT_REPORTS_DIR,
    optimizer: PortfolioOptimizer | None = None,
    enhanced_optimizer: EnhancedOptimizer | None = None,
    strategy_config: StrategyConfig | None = None,
    trainer: BaselineTrainer | None = None,
    rolling_config: RollingConfig | None = None,
    registry_dir: str | Path = DEFAULT_REGISTRY_DIR,
    rolling_trainer_factory: TrainerFactory | None = None,
    rolling_trainer_loader: TrainerLoader | None = None,
    initial_cash: float = DEFAULT_INITIAL_CASH,
    validate_data: bool = True,
    dry_run: bool = False,
) -> DailyReport:
    """执行一次每日跑批，返回 :class:`DailyReport`。

    数据必须已由 ``scripts/daily_update.py`` 更新到 ``ref_date``。``dry_run=True`` 时
    只计算不落盘，也不改动虚拟账户。

    ``strategy_config`` 缺省（``None``）时按现状走全市场量化选股 + PortfolioOptimizer；
    给出后按配置的 ``strategy`` 分派：``stock_selection`` 仍走 PortfolioOptimizer，
    ``index_enhanced`` 走 :class:`~quant.portfolio.enhanced.EnhancedOptimizer`，候选集为
    基准成分 ∪ 打分为主 top-K ∪ 当前持仓。``universe`` 非空时打分先裁决到池内。

    ``rolling_config`` 非空时启用滚动重训（issue #34）：按
    :func:`quant.automl.rolling.resolve_model` 到期重训 / 复用注册表里的最近版本，
    忽略 ``model_dir`` 静态模型；``dry_run=True`` 时只复用不重训。
    ``rolling_trainer_factory`` / ``rolling_trainer_loader`` 为测试注入点。
    """
    data_dir = Path(data_dir)
    config = strategy_config if strategy_config is not None else StrategyConfig()
    effective_top_k = config.top_k if strategy_config is not None else top_k
    account_path = VirtualAccount.path_for(Path(account_dir), account_name)
    signal_day = resolve_reference_date(data_dir, ref_date)
    account = VirtualAccount.load_or_create(
        account_path, name=account_name, initial_cash=initial_cash
    )

    # -- 幂等：同日（或更早）已跑过则直接跳过，不改账户 ---------------------
    if account.last_pipeline_date is not None and signal_day <= account.last_pipeline_date:
        return DailyReport(
            date=signal_day,
            account_name=account_name,
            already_ran=True,
            nav=account.latest_nav,
            cash=account.cash,
            notes=[
                f"{signal_day.isoformat()} 已跑过（last_pipeline_date="
                f"{account.last_pipeline_date.isoformat()}），本次跳过"
            ],
        )

    # -- 体检（纪律第 6 条）------------------------------------------------
    if validate_data:
        report = validate(data_dir, end=signal_day, lookback_days=lookback_days)
        if not report.ok:
            raise DailyError(
                "数据体检未通过，中止跑批："
                + "；".join(issue.message for issue in report.errors)
            )

    # -- 行情窗口 ----------------------------------------------------------
    calendar = load_calendar(data_dir, end=signal_day)
    window_days = lookback_days + FACTOR_WINDOW_MARGIN
    start = _window_start(calendar, signal_day, window_days)
    bars = load_bars(data_dir, start=start, end=signal_day)
    if bars.height == 0:
        raise DailyError(f"行情窗口 [{start}, {signal_day}] 内没有数据")
    if bars.filter(pl.col(DATE_COL) == signal_day).height == 0:
        raise DailyError(f"信号日 {signal_day} 没有行情")

    # -- 因子 → 模型打分 ---------------------------------------------------
    factors = discover_factors(factor_library_dir)
    model_notes: list[str] = []
    if rolling_config is not None:
        resolved = resolve_model(
            data_dir,
            signal_day,
            factors,
            rolling_config,
            registry_dir=registry_dir,
            universe=config.universe,
            trainer_factory=rolling_trainer_factory,
            trainer_loader=rolling_trainer_loader,
            no_train=dry_run,
        )
        active_trainer = resolved.trainer
        if resolved.action == "retrained" and resolved.meta is not None:
            model_notes.append(
                f"模型滚动重训：v_{resolved.train_end.isoformat()}（训练区间 "
                f"{resolved.meta.train_start.isoformat()} ~ "
                f"{resolved.meta.train_end.isoformat()}，"
                f"{resolved.meta.n_rows} 行）"
            )
        else:
            model_notes.append(
                f"复用模型版本 v_{resolved.train_end.isoformat()}（未到期）"
            )
    else:
        active_trainer = (
            trainer if trainer is not None else BaselineTrainer.load(model_dir)
        )
    dataset = build_dataset(bars, factors)
    scoring = dataset.filter(pl.col(DATE_COL) == signal_day)
    if scoring.height == 0:
        raise DailyError(f"信号日 {signal_day} 没有因子数据")
    scores = _valid_scores(active_trainer.predict(scoring))
    if scores.height == 0:
        raise DailyError("模型打分在信号日全部为缺失，无法构建候选集")
    scores = scores.sort([SCORE_COL, INSTRUMENT_COL], descending=[True, False])

    # -- 池内裁决（universe 非空时打分只保留当日 PIT 成员）----------------
    if config.universe is not None:
        pool = members(config.universe, signal_day, data_dir=data_dir)
        if not pool:
            raise DailyError(
                f"股票池 {config.universe!r} 在 {signal_day} 没有成员，无法构建候选集"
            )
        scores = scores.filter(pl.col(INSTRUMENT_COL).is_in(sorted(pool)))

    # -- 账户估值与当前权重 ------------------------------------------------
    prices = _latest_prices(bars, signal_day)
    for instrument, position in account.positions.items():
        if position.volume > 0 and instrument not in prices:
            if position.avg_cost <= 0:
                raise DailyError(f"持仓 {instrument} 缺少价格且成本为 0，无法估值")
            prices[instrument] = position.avg_cost
    nav = account.nav(prices)
    market_value = account.market_value(prices)
    current_weights = account.weights(prices)
    current_volumes = {
        instrument: position.volume
        for instrument, position in account.positions.items()
        if position.volume > 0
    }

    # -- 候选集与组合优化（按 strategy 分派，含降级）-----------------------
    if config.is_index_enhanced:
        result, n_candidates, notes = _optimize_enhanced_path(
            data_dir=data_dir,
            bars=bars,
            calendar=calendar,
            scores=scores,
            signal_day=signal_day,
            prices=prices,
            current_weights=current_weights,
            current_volumes=current_volumes,
            config=config,
            top_k=effective_top_k,
            optimizer=enhanced_optimizer,
        )
    else:
        result, n_candidates, notes = _optimize_stock_path(
            bars=bars,
            scores=scores,
            signal_day=signal_day,
            prices=prices,
            current_weights=current_weights,
            current_volumes=current_volumes,
            config=config,
            top_k=effective_top_k,
            optimizer=optimizer,
        )
    relaxed = result.relaxed
    hold_fallback = result.hold_fallback

    if result.optimize is None:
        target_weights = dict(current_weights)
        target_volumes = dict(current_volumes)
    else:
        target_weights = _normalize_weights(result.optimize.weights)
        rounded = round_weights_to_lots(target_weights, prices, nav)
        target_volumes = {
            str(row["instrument"]): int(row["volume"])
            for row in rounded.orders.iter_rows(named=True)
        }

    # -- 调仓单 diff + 预过滤 ---------------------------------------------
    diff = diff_orders(target_volumes, current_volumes)
    ref_bars = bars.filter(pl.col(DATE_COL) == signal_day).select(
        INSTRUMENT_COL, "close", "limit_up", "limit_down"
    )
    executable, rejected = filter_executable(diff, ref_bars)
    orders = _enrich_orders(
        executable, signal_day, prices, current_volumes, target_volumes
    )
    filtered = _enrich_orders(
        rejected, signal_day, prices, current_volumes, target_volumes, with_reason=True
    )

    holdings = account.holdings(prices)
    report = DailyReport(
        date=signal_day,
        account_name=account_name,
        nav=nav,
        cash=account.cash,
        market_value=market_value,
        strategy=config.strategy,
        universe=config.universe,
        benchmark=config.benchmark,
        rebalance_freq=config.rebalance_freq,
        relaxed=relaxed,
        hold_fallback=hold_fallback,
        n_candidates=n_candidates,
        scores=scores,
        top_scores=[
            (str(row[INSTRUMENT_COL]), float(row[SCORE_COL]))
            for row in scores.head(TOP_SCORES_IN_REPORT).iter_rows(named=True)
        ],
        target_weights=dict(target_weights),
        current_weights=dict(current_weights),
        target_volumes=dict(target_volumes),
        orders=orders,
        filtered=filtered,
        holdings=holdings,
        exposure={
            "cash": account.cash,
            "market_value": market_value,
            "nav": nav,
            "cash_ratio": account.cash / nav if nav > 0 else 0.0,
            "position_ratio": market_value / nav if nav > 0 else 0.0,
            "n_positions": float(len(current_volumes)),
        },
        turnover=float(result.turnover),
        notes=model_notes + notes,
    )

    if not dry_run:
        report.orders_path = Path(orders_dir) / f"{signal_day.isoformat()}.parquet"
        report.orders_csv_path = Path(orders_dir) / f"{signal_day.isoformat()}.csv"
        _write_orders(report.orders, report.orders_path, report.orders_csv_path)
        report.report_path = Path(reports_dir) / f"{signal_day.isoformat()}.json"
        _write_report(report, report.report_path)
        report.briefing_path = Path(reports_dir) / f"{signal_day.isoformat()}.md"
        _write_briefing(
            data_dir=data_dir,
            bars=bars,
            factors=factors,
            report=report,
            signal_day=signal_day,
            config=config,
        )
        account.record_nav(signal_day, prices)
        account.last_pipeline_date = signal_day
        account.save(account_path)

    return report


# ---------------------------------------------------------------------------
# 优化与降级
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _OptimizeOutcome:
    """优化结果 + 降级标记；``optimize`` 为 None 表示保持现持仓。"""

    optimize: OptimizeResult | None
    relaxed: bool = False
    hold_fallback: bool = False
    turnover: float = 0.0


def _optimize(
    optimizer: PortfolioOptimizer,
    alpha: Mapping[str, float],
    candidates: Sequence[str],
    returns: pl.DataFrame,
    w_prev: Mapping[str, float],
) -> _OptimizeOutcome:
    """先按原约束优化；不可行时放宽 ``max_turnover`` 重试；再失败则保持现持仓。"""
    try:
        result = optimizer.optimize(alpha, list(candidates), returns, w_prev)
        return _OptimizeOutcome(optimize=result, turnover=result.turnover)
    except InfeasibleError:
        logger.warning("组合优化不可行，放宽 max_turnover 至 %.1f 重试", MAX_TURNOVER_RELAXED)
    relaxed_optimizer = dataclasses.replace(
        optimizer, max_turnover=MAX_TURNOVER_RELAXED
    )
    try:
        result = relaxed_optimizer.optimize(alpha, list(candidates), returns, w_prev)
        return _OptimizeOutcome(
            optimize=result, relaxed=True, turnover=result.turnover
        )
    except InfeasibleError:
        logger.error("放宽换手后组合仍不可行，保持现有持仓")
        return _OptimizeOutcome(optimize=None, relaxed=True, hold_fallback=True)


# ---------------------------------------------------------------------------
# 策略分派
# ---------------------------------------------------------------------------


def _optimize_stock_path(
    *,
    bars: pl.DataFrame,
    scores: pl.DataFrame,
    signal_day: date,
    prices: Mapping[str, float],
    current_weights: Mapping[str, float],
    current_volumes: Mapping[str, int],
    config: StrategyConfig,
    top_k: int,
    optimizer: PortfolioOptimizer | None,
) -> tuple[_OptimizeOutcome, int, list[str]]:
    """stock_selection 路径：top-K ∪ 当前持仓 → PortfolioOptimizer（含放宽换手）。"""
    score_map = {
        str(row[INSTRUMENT_COL]): float(row[SCORE_COL])
        for row in scores.iter_rows(named=True)
    }
    if not score_map:
        raise DailyError("打分全空，无法构建候选集")
    min_score = min(score_map.values())
    top_instruments = [str(inst) for inst in scores.head(top_k)[INSTRUMENT_COL].to_list()]
    candidates = list(top_instruments) + [
        instrument
        for instrument in current_volumes
        if instrument not in set(top_instruments)
    ]
    candidates = [instrument for instrument in candidates if instrument in prices]
    if not candidates:
        raise DailyError("候选证券为空，无法优化")
    alpha = {
        instrument: score_map.get(instrument, min_score) for instrument in candidates
    }

    active_optimizer = optimizer if optimizer is not None else build_stock_optimizer(config)
    returns = _returns_frame(bars, candidates, signal_day)
    outcome = _optimize(active_optimizer, alpha, candidates, returns, current_weights)
    notes: list[str] = []
    if outcome.relaxed:
        notes.append(
            f"组合优化不可行，已放宽 max_turnover 至 {MAX_TURNOVER_RELAXED:.1f} 重试成功"
        )
    if outcome.hold_fallback:
        notes.append("*** 放宽后仍不可行：保持现有持仓，本次不调仓 ***")
    return outcome, len(candidates), notes


def _optimize_enhanced_path(
    *,
    data_dir: Path,
    bars: pl.DataFrame,
    calendar: pl.DataFrame,
    scores: pl.DataFrame,
    signal_day: date,
    prices: Mapping[str, float],
    current_weights: Mapping[str, float],
    current_volumes: Mapping[str, int],
    config: StrategyConfig,
    top_k: int,
    optimizer: EnhancedOptimizer | None,
) -> tuple[_OptimizeOutcome, int, list[str]]:
    """index_enhanced 路径：基准成分 ∪ top-K ∪ 当前持仓 → EnhancedOptimizer。

    基准权重取信号日或之前最近的 PIT 日频权重；非调仓日保持现有持仓。
    EnhancedOptimizer 内部自带放松阶梯，``STATUS_HELD`` 视为保持现持仓降级。
    """
    notes: list[str] = []
    if config.benchmark is None:
        raise DailyError("index_enhanced 策略缺少 benchmark")
    bench_weights = benchmark_weights(data_dir, config.benchmark, signal_day)
    if not bench_weights:
        raise DailyError(
            f"基准 {config.benchmark} 在 {signal_day} 及以前没有权重数据，"
            "请先跑 index 更新（build_daily_tables）"
        )

    score_map = {
        str(row[INSTRUMENT_COL]): float(row[SCORE_COL])
        for row in scores.iter_rows(named=True)
    }
    if not score_map:
        raise DailyError("打分全空，无法构建候选集")
    min_score = min(score_map.values())
    top_instruments = [str(inst) for inst in scores.head(top_k)[INSTRUMENT_COL].to_list()]
    ordered = list(
        dict.fromkeys([*sorted(bench_weights), *top_instruments, *current_volumes])
    )
    candidates = [instrument for instrument in ordered if instrument in prices]
    if not candidates:
        raise DailyError("候选证券为空，无法优化")

    open_days = calendar.filter(pl.col("is_open"))[DATE_COL].to_list()
    if not is_rebalance_day(open_days, signal_day, config.rebalance_freq):
        notes.append(
            f"{signal_day.isoformat()} 非 {config.rebalance_freq} 调仓日，保持现有持仓"
        )
        return _OptimizeOutcome(optimize=None, hold_fallback=True), len(candidates), notes

    alpha = {
        instrument: score_map.get(instrument, min_score) for instrument in candidates
    }
    active_optimizer = (
        optimizer if optimizer is not None else build_enhanced_optimizer(config)
    )
    returns = _returns_frame(bars, candidates, signal_day)
    industry, float_mv, style = _enhanced_exposures(
        data_dir, bars, signal_day, candidates, bench_weights
    )
    candidate_set = set(candidates)
    w_prev = {
        instrument: float(weight)
        for instrument, weight in current_weights.items()
        if instrument in candidate_set
    }
    try:
        enhanced = active_optimizer.optimize_day(
            date=signal_day,
            instruments=candidates,
            alpha=alpha,
            bench_weights=bench_weights,
            industry=industry,
            float_mv=float_mv,
            style=style,
            covariance=returns,
            w_prev=w_prev or None,
        )
    except (ValueError, PortfolioError) as exc:
        logger.error("指增优化异常，保持现有持仓：%s", exc)
        notes.append(f"*** 指增优化异常：保持现有持仓，本次不调仓（{exc}）***")
        return (
            _OptimizeOutcome(optimize=None, relaxed=True, hold_fallback=True),
            len(candidates),
            notes,
        )

    if enhanced.status == STATUS_HELD:
        notes.append("*** 指增优化失败：保持现有持仓，本次不调仓 ***")
        return (
            _OptimizeOutcome(optimize=None, relaxed=True, hold_fallback=True),
            len(candidates),
            notes,
        )
    relaxed = enhanced.status == STATUS_RELAXED
    if relaxed:
        notes.append("指增优化经放松阶梯后求解成功")
    outcome = _OptimizeOutcome(
        optimize=OptimizeResult(
            weights=dict(enhanced.weights),
            objective=float(enhanced.objective if enhanced.objective is not None else 0.0),
            turnover=float(enhanced.turnover),
            status=enhanced.status,
        ),
        relaxed=relaxed,
        turnover=float(enhanced.turnover),
    )
    return outcome, len(candidates), notes


def _enhanced_exposures(
    data_dir: Path,
    bars: pl.DataFrame,
    signal_day: date,
    candidates: Sequence[str],
    bench_weights: Mapping[str, float],
) -> tuple[dict[str, str], dict[str, float], dict[str, dict[str, float]]]:
    """组装指增优化所需的行业 / 流通市值 / 六风格暴露映射（均按 cutoff 截取）。

    行业优先取 ``effective_from <= signal_day`` 的快照；仓库暂无历史快照时退化到
    最新一份并记 warning（与 :mod:`quant.portfolio.enhanced_inputs` 约定一致）。
    """
    covered = sorted(set(candidates) | set(bench_weights))
    industry_frame, fallback = industry_table(data_dir, signal_day, covered)
    if fallback:
        logger.warning(
            "%s 没有不晚于该日的行业快照，行业归属退化到表内最新一份（数据可得性限制）",
            signal_day,
        )
    return (
        industry_map(industry_frame),
        float_mv_map(float_mv_frame(data_dir, bars), signal_day),
        style_map(style_frame(bars), signal_day),
    )


# ---------------------------------------------------------------------------
# 数据准备工具
# ---------------------------------------------------------------------------


def resolve_reference_date(data_dir: Path, requested: date | None) -> date:
    """返回 ``<= requested``（缺省今天）的最近开市日。"""
    data_dir = Path(data_dir)
    end = requested or date.today()
    try:
        calendar = load_calendar(data_dir, end=end)
    except FileNotFoundError as exc:
        raise DailyError(f"缺少交易日历：{data_dir}") from exc
    opened = calendar.filter(pl.col("is_open"))
    if opened.height == 0:
        raise DailyError(f"交易日历中没有 <= {end.isoformat()} 的开市日")
    return opened["date"].max()


def _window_start(calendar: pl.DataFrame, ref_date: date, trading_days: int) -> date:
    """取 ``ref_date`` 往前 ``trading_days`` 个开市日作为行情窗口起点。"""
    days = (
        calendar.filter(pl.col("is_open") & (pl.col("date") <= ref_date))
        .sort("date")["date"]
        .to_list()
    )
    if not days:
        return ref_date
    index = max(0, len(days) - trading_days)
    return days[index]


def _normalize_weights(weights: Mapping[str, float]) -> dict[str, float]:
    """把优化器解出的目标权重归一到 ``Σw <= 1``。

    求解器在等式约束 ``Σw = 1`` 上会留 ~1e-6 量级的浮点超调，超过整手取整模块
    ``WEIGHT_SUM_TOL`` 的容忍度；这里仅在总和超过 1 时按比例缩回，保证不透支。
    """
    total = float(sum(weights.values()))
    if total <= 0.0:
        raise DailyError("优化器给出的目标权重之和非正")
    if total > 1.0:
        return {instrument: value / total for instrument, value in weights.items()}
    return dict(weights)


def _valid_scores(scores: pl.DataFrame) -> pl.DataFrame:
    """校验打分表结构并剔除缺失 / 非有限打分。"""
    missing = [col for col in SCORES_SCHEMA if col not in scores.columns]
    if missing:
        raise DailyError(f"打分表缺少列：{missing}")
    return scores.select(list(SCORES_SCHEMA)).filter(
        pl.col(SCORE_COL).is_not_null() & pl.col(SCORE_COL).is_finite()
    )


def _latest_prices(bars: pl.DataFrame, ref_date: date) -> dict[str, float]:
    """每只证券 ``<= ref_date`` 的最近一个正收盘价。"""
    valid = bars.filter(
        (pl.col(DATE_COL) <= ref_date)
        & pl.col("close").is_not_null()
        & pl.col("close").is_finite()
        & (pl.col("close") > 0)
    ).sort([INSTRUMENT_COL, DATE_COL])
    if valid.height == 0:
        return {}
    latest = valid.group_by(INSTRUMENT_COL).agg(pl.col("close").last())
    return {
        str(row[INSTRUMENT_COL]): float(row["close"])
        for row in latest.iter_rows(named=True)
    }


def _returns_frame(
    bars: pl.DataFrame, instruments: Sequence[str], ref_date: date
) -> pl.DataFrame:
    """候选证券的日收益矩阵（列顺序与 ``instruments`` 一致）。

    口径与 :func:`quant.portfolio.enhanced_inputs.returns_matrix` 一致。
    """
    return returns_matrix(
        bars, instruments, ref_date, window=RETURNS_WINDOW_DEFAULT
    )


def _enrich_orders(
    rows: Sequence[Mapping[str, Any]],
    day: date,
    prices: Mapping[str, float],
    current_volumes: Mapping[str, int],
    target_volumes: Mapping[str, int],
    *,
    with_reason: bool = False,
) -> pl.DataFrame:
    """把 diff / 过滤结果补全为落盘 schema，附板块、参考价与目标 / 当前股数。"""
    schema = FILTERED_SCHEMA if with_reason else ORDER_SCHEMA
    records: list[dict[str, Any]] = []
    for row in rows:
        instrument = str(row["instrument"])
        volume = int(row["volume"])
        price = float(prices.get(instrument, 0.0))
        record: dict[str, Any] = {
            "date": day,
            "instrument": instrument,
            "board": board_of(instrument),
            "side": str(row["side"]),
            "volume": volume,
            "ref_price": price,
            "est_amount": volume * price,
            "current_volume": int(current_volumes.get(instrument, 0)),
            "target_volume": int(target_volumes.get(instrument, 0)),
        }
        if with_reason:
            record["reason"] = str(row.get("reason", ""))
        records.append(record)
    if not records:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(records).select(list(schema)).cast(schema)


# ---------------------------------------------------------------------------
# 落盘
# ---------------------------------------------------------------------------


def _write_orders(orders: pl.DataFrame, parquet: Path, csv_path: Path) -> None:
    parquet.parent.mkdir(parents=True, exist_ok=True)
    orders.write_parquet(parquet)
    orders.write_csv(csv_path)


def _write_report(report: DailyReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_report_dict(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_briefing(
    *,
    data_dir: Path,
    bars: pl.DataFrame,
    factors: Mapping[str, FactorCompute],
    report: DailyReport,
    signal_day: date,
    config: StrategyConfig,
) -> None:
    """渲染并落盘每日 markdown 简报（调仓明细 / 敞口 / 因子近端 RankIC）。

    因子近期表现复用跑批已加载的 ``bars`` 与 ``factors``；行业敞口与指增风险四表
    按当日 PIT 截面组装，数据不足时对应板块留扩展点。
    """
    instruments = (
        sorted(set(report.holdings[INSTRUMENT_COL].to_list()))
        if report.holdings.height
        else []
    )
    industry_frame, _ = industry_table(data_dir, signal_day, instruments)
    write_briefing(
        report,
        report.briefing_path,
        factor_ic=recent_factor_ic(bars, factors, signal_day),
        industry=industry_map(industry_frame),
        risk_report=build_briefing_risk_report(
            data_dir,
            bars,
            report,
            benchmark=config.benchmark,
            industry=industry_frame,
        ),
        notes=report.notes,
    )


def _report_dict(report: DailyReport) -> dict[str, Any]:
    """把报告转成可 JSON 序列化的字典。"""

    def records(frame: pl.DataFrame) -> list[dict[str, Any]]:
        if frame.height == 0:
            return []
        out: list[dict[str, Any]] = []
        for row in frame.to_dicts():
            out.append(
                {
                    key: (value.isoformat() if isinstance(value, date) else value)
                    for key, value in row.items()
                }
            )
        return out

    return {
        "date": report.date.isoformat(),
        "account_name": report.account_name,
        "strategy": report.strategy,
        "universe": report.universe,
        "benchmark": report.benchmark,
        "rebalance_freq": report.rebalance_freq,
        "already_ran": report.already_ran,
        "relaxed": report.relaxed,
        "hold_fallback": report.hold_fallback,
        "nav": report.nav,
        "cash": report.cash,
        "market_value": report.market_value,
        "turnover": report.turnover,
        "n_candidates": report.n_candidates,
        "top_scores": [
            {"instrument": instrument, "score": score}
            for instrument, score in report.top_scores
        ],
        "target_weights": report.target_weights,
        "target_volumes": report.target_volumes,
        "orders": records(report.orders),
        "filtered": records(report.filtered),
        "holdings": records(report.holdings),
        "exposure": report.exposure,
        "notes": report.notes,
        "briefing_path": (
            str(report.briefing_path) if report.briefing_path is not None else None
        ),
    }


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------


def _finite(value: object) -> float:
    """把可为 None / NaN / inf 的数值归一化为有限浮点，缺省 0.0。"""
    if value is None:
        return 0.0
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _finite_or_none(value: object) -> float | None:
    """返回有限浮点，缺失 / 非有限时为 None。"""
    if value is None:
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


__all__ = [
    "DailyError",
    "DailyReport",
    "FACTOR_WINDOW_MARGIN",
    "FILTERED_SCHEMA",
    "LOOKBACK_DAYS_DEFAULT",
    "MAX_TURNOVER_RELAXED",
    "ORDER_SCHEMA",
    "TOP_K_DEFAULT",
    "diff_orders",
    "discover_factors",
    "filter_executable",
    "resolve_reference_date",
    "run_daily",
]
