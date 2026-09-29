"""``scripts`` 端到端编排测试（issue #16）。

覆盖 ``scripts/train_baseline.py`` 与 ``scripts/backtest_e2e.py`` 的**核心编排函数**：

- 训练集构造的丢弃规则与 ``--max-rows`` 截断；
- ``train_baseline`` 真实调用注入 trainer 的 ``train`` / ``save`` 并落盘训练配置；
- ``run_e2e`` 的信号函数端到端走通（优化被调、生成订单、落盘四个产物）；
- 打分全空、优化不可行两条降级路径；
- 训练配置与因子库特征列不一致时报错。

全部用合成数据 + 假 trainer / 假 optimizer，不碰真实 ``data/`` 与 ``runs/``，不加载
或训练 AutoGluon。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from quant.automl.dataset import INSTRUMENT_COL
from quant.data.schema import (
    CORPORATE_ACTIONS,
    DAILY_BARS,
    FLOAT_MV,
    INDEX_BARS,
    INDEX_MEMBERS,
    INDEX_WEIGHTS,
    INDUSTRY,
    INSTRUMENT_INFO,
    TRADE_CALENDAR,
)
from quant.portfolio.enhanced import EnhancedOptimizer
from quant.portfolio.optimizer import InfeasibleError, PortfolioOptimizer
from scripts import backtest_e2e as e2e
from scripts import train_baseline as tb

FACTOR_LIBRARY = Path(__file__).resolve().parents[1] / "factor_library"

# 周一
START = date(2026, 1, 5)


# ---------------------------------------------------------------------------
# 合成数据
# ---------------------------------------------------------------------------


def _open_days(n_days: int, start: date = START) -> list[date]:
    days: list[date] = []
    cursor = start
    while len(days) < n_days:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor += timedelta(days=1)
    return days


def _instruments(n: int) -> list[str]:
    return [f"{600000 + i:06d}.SH" for i in range(n)]


def _bars(instruments: list[str], days: list[date]) -> pl.DataFrame:
    """各票价格中枢不同、温和上行带小幅震荡，涨跌幅远小于涨跌停带。"""
    rows: list[dict[str, object]] = []
    for i, instrument in enumerate(instruments):
        base = 5.0 + 3.0 * i
        for d, day in enumerate(days):
            price = base + 0.15 * d + 0.2 * ((d + i) % 4)
            rows.append(
                {
                    "date": day,
                    "instrument": instrument,
                    "open": price,
                    "high": price * 1.01,
                    "low": price * 0.99,
                    "close": price,
                    "vwap": price,
                    "volume": 1000.0 + 10.0 * i + d,
                    "amount": price * 1000.0,
                    "adjfactor": 1.0,
                    "limit_up": None,
                    "limit_down": None,
                }
            )
    return pl.DataFrame(rows, schema=DAILY_BARS).sort(["instrument", "date"])


def _write_data_dir(
    root: Path, *, n_instruments: int = 6, n_days: int = 80
) -> tuple[Path, list[str], list[date]]:
    """写出一份最小可用的 ``data/`` 缓存，返回 ``(data_dir, instruments, days)``。"""
    data_dir = root / "data"
    (data_dir / "bars").mkdir(parents=True, exist_ok=True)
    instruments = _instruments(n_instruments)
    days = _open_days(n_days)

    calendar_rows: list[dict[str, object]] = []
    cursor = days[0]
    while cursor <= days[-1]:
        calendar_rows.append({"date": cursor, "is_open": cursor.weekday() < 5})
        cursor += timedelta(days=1)
    pl.DataFrame(calendar_rows, schema=TRADE_CALENDAR).write_parquet(
        data_dir / "calendar.parquet"
    )
    pl.DataFrame(
        [
            {
                "instrument": instrument,
                "name": instrument,
                "board": "main",
                "list_date": date(2020, 1, 1),
                "delist_date": None,
            }
            for instrument in instruments
        ],
        schema=INSTRUMENT_INFO,
    ).write_parquet(data_dir / "instruments.parquet")
    pl.DataFrame(schema=CORPORATE_ACTIONS).write_parquet(
        data_dir / "corporate_actions.parquet"
    )

    bars = _bars(instruments, days)
    for year in sorted(set(bars["date"].dt.year().to_list())):
        chunk = bars.filter(pl.col("date").dt.year() == year)
        chunk.write_parquet(data_dir / "bars" / f"{int(year):04d}.parquet")
    return data_dir, instruments, days


def _write_index_bars(
    data_dir: Path, days: list[date], code: str = "000905", base: float = 5000.0
) -> None:
    """写一份指数日线（issue #67），收盘温和上行。"""
    rows = [
        {
            "date": day,
            "index_code": code,
            "open": base + float(i),
            "high": base + float(i) + 5.0,
            "low": base + float(i) - 5.0,
            "close": base + float(i),
            "volume": 1.0e9,
        }
        for i, day in enumerate(days)
    ]
    pl.DataFrame(rows, schema=INDEX_BARS).write_parquet(data_dir / "index_bars.parquet")


def _write_index_members(
    data_dir: Path,
    days: list[date],
    instruments: list[str],
    code: str = "000300",
) -> None:
    """写一份命名池成分表：``code`` 在每天都包含 ``instruments``。"""
    rows = [
        {"date": day, "instrument": instrument, "index_code": code}
        for day in days
        for instrument in instruments
    ]
    pl.DataFrame(rows, schema=INDEX_MEMBERS).write_parquet(
        data_dir / "index_members.parquet"
    )


def _write_index_weights(
    data_dir: Path, days: list[date], instruments: list[str], code: str = "000852"
) -> None:
    """写一份日频等权指数权重（PIT 基准），供指增路径读基准。"""
    share = 1.0 / len(instruments)
    rows = [
        {"date": day, "instrument": instrument, "index_code": code, "weight": share}
        for day in days
        for instrument in instruments
    ]
    pl.DataFrame(rows, schema=INDEX_WEIGHTS).write_parquet(
        data_dir / "index_weights.parquet"
    )


def _write_industry(data_dir: Path, instruments: list[str]) -> None:
    """写一份全覆盖的行业归属（issue #65 口径），指增的行业约束需要它。"""
    pl.DataFrame(
        [
            {
                "instrument": instrument,
                "industry_l1": "信息技术",
                "industry_l2": "半导体",
                "effective_from": date(2020, 1, 1),
            }
            for instrument in instruments
        ],
        schema=INDUSTRY,
    ).write_parquet(data_dir / "industry.parquet")


def _write_float_mv(
    data_dir: Path, days: list[date], instruments: list[str], value: float = 1.0e7
) -> None:
    """写一份常数流通市值（千元），替代等效市值分支。"""
    rows = [
        {"date": day, "instrument": instrument, "float_mv": value}
        for day in days
        for instrument in instruments
    ]
    pl.DataFrame(rows, schema=FLOAT_MV).write_parquet(data_dir / "float_mv.parquet")


# ---------------------------------------------------------------------------
# 假 trainer / 假 optimizer
# ---------------------------------------------------------------------------


class _FakeTrainer:
    """固定打分的假 trainer；``load`` 忽略模型目录，``predict`` 按证券查表。

    ``score_map`` 为 ``None`` 时所有打分记 null（用于「打分全空」用例）。
    """

    score_map: dict[str, float] | None = {}

    def __init__(self, score_map: dict[str, float] | None = None) -> None:
        self._score_map = score_map if score_map is not None else {}

    @classmethod
    def load(cls, path: Any, **kwargs: Any) -> "_FakeTrainer":
        return cls(dict(cls.score_map) if cls.score_map is not None else None)

    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        if self._score_map is None:
            return df.select("date", "instrument").with_columns(
                pl.lit(None, dtype=pl.Float64).alias("score")
            )
        mapping = self._score_map
        return df.select("date", "instrument").with_columns(
            pl.col("instrument")
            .map_elements(lambda value: mapping.get(str(value)), return_dtype=pl.Float64)
            .alias("score")
        )


class _RecordingTrainer:
    """记录 ``train`` / ``save`` 是否被调用的假 trainer，用于训练编排用例。"""

    def __init__(self) -> None:
        self.trained = False
        self.saved = False
        self.rows = 0
        self.feature_columns: list[str] | None = None

    def train(self, train_df: pl.DataFrame, **kwargs: Any) -> "_RecordingTrainer":
        self.trained = True
        self.rows = train_df.height
        return self

    def save(self) -> str:
        self.saved = True
        return "unused"


@dataclass
class _InfeasibleOptimizer:
    """始终抛 ``InfeasibleError`` 的假优化器；``replace`` 后共享 ``calls`` 计数。"""

    max_turnover: float = 0.30
    calls: list[bool] = field(default_factory=list)

    def optimize(
        self,
        alpha: Any,
        instruments: Any,
        returns: Any,
        w_prev: Any = None,
        **kwargs: Any,
    ) -> Any:
        self.calls.append(True)
        raise InfeasibleError("测试：强制不可行")


def _score_map(instruments: list[str]) -> dict[str, float]:
    return {inst: float(len(instruments) - i) for i, inst in enumerate(instruments)}


def _write_train_config(
    model_dir: Path,
    *,
    feature_columns: list[str],
    start: date | None = None,
    end: date | None = None,
    n_rows: int = 0,
    universe: str | None = None,
) -> Path:
    path = tb.default_train_config_path(model_dir)
    tb.write_train_config(
        path,
        tb.TrainConfig(
            data_dir="data",
            start=start.isoformat() if start else None,
            end=end.isoformat() if end else None,
            n_rows=n_rows,
            feature_columns=list(feature_columns),
            presets="medium_quality",
            time_limit=60.0,
            horizon=1,
            max_rows=0,
            universe=universe,
        ),
    )
    return path


def _e2e_config(
    tmp_path: Path,
    data_dir: Path,
    days: list[date],
    *,
    lookback_days: int = 20,
    top_k: int = 6,
    model_dir: Path | None = None,
    train_config_path: Path | None = None,
) -> e2e.E2EConfig:
    return e2e.E2EConfig(
        data_dir=data_dir,
        model_dir=model_dir if model_dir is not None else tmp_path / "model",
        out_dir=tmp_path / "e2e",
        start=days[20],
        end=days[-1],
        initial_cash=1_000_000.0,
        top_k=top_k,
        lookback_days=lookback_days,
        factor_library_dir=FACTOR_LIBRARY,
        train_config_path=train_config_path,
    )


# ---------------------------------------------------------------------------
# train_baseline：数据集构造
# ---------------------------------------------------------------------------


def test_prepare_training_dataset_drops_null_label_and_all_null_features() -> None:
    instruments = _instruments(3)
    days = _open_days(10)
    bars = _bars(instruments, days)

    def factor_gated(data: pl.DataFrame) -> pl.DataFrame:
        # 第一只票恒 null：当两个特征都走这条 gate 时，该票样本全部特征为空。
        return data.select(
            "date",
            "instrument",
            pl.when(pl.col("instrument") != instruments[0])
            .then(pl.col("close"))
            .otherwise(None)
            .alias("value"),
        )

    dataset, summary = tb.prepare_training_dataset(
        bars, {"fa": factor_gated, "fb": factor_gated}
    )

    assert summary.total_rows == 30
    # horizon=1 需要 T+1 与 T+2 两行未来行情，每票尾部 2 行 label 为空。
    assert summary.dropped_null_label == 6
    # 第一只票 10 行两列特征全空。
    assert summary.dropped_all_null_feature == 10
    # 两处丢弃有 2 行重叠（第一只票的尾部）：30 − 6 − 10 + 2 = 16。
    assert summary.kept_rows == 16
    assert dataset.height == 16
    assert dataset[INSTRUMENT_COL].n_unique() == 2
    assert dataset.filter(pl.col("label").is_null()).height == 0
    assert dataset.filter(pl.col("fa").is_null()).height == 0


def test_prepare_training_dataset_max_rows_truncates() -> None:
    instruments = _instruments(3)
    days = _open_days(30)
    bars = _bars(instruments, days)
    factors = {"fa": lambda data: data.select("date", "instrument", pl.col("close").alias("value"))}

    dataset, summary = tb.prepare_training_dataset(bars, factors, max_rows=25)
    assert dataset.height == 25
    assert summary.kept_rows == 25


# ---------------------------------------------------------------------------
# train_baseline：编排
# ---------------------------------------------------------------------------


def test_train_baseline_invokes_trainer_and_writes_config(tmp_path: Path) -> None:
    data_dir, _, days = _write_data_dir(tmp_path)
    model_dir = tmp_path / "model"
    trainer = _RecordingTrainer()

    report = tb.train_baseline(
        data_dir,
        model_dir,
        factor_library_dir=FACTOR_LIBRARY,
        start=days[5],
        end=days[-1],
        time_limit=42.0,
        trainer=trainer,
    )

    assert trainer.trained, "train_baseline 必须调用 trainer.train"
    assert trainer.saved, "train_baseline 必须调用 trainer.save"
    assert trainer.rows == report.summary.kept_rows
    assert report.summary.kept_rows > 0
    assert report.leaderboard is None  # 假 trainer 没有 predictor

    assert report.config_path == tb.default_train_config_path(model_dir)
    loaded = tb.load_train_config(report.config_path)
    assert loaded.feature_columns == list(report.summary.feature_columns)
    assert loaded.n_rows == report.summary.kept_rows
    assert loaded.time_limit == pytest.approx(42.0)
    assert loaded.start == days[5].isoformat()
    assert loaded.end == days[-1].isoformat()
    assert loaded.universe is None


def test_train_baseline_universe_filters_and_records(tmp_path: Path) -> None:
    """``--universe`` 把行情裁到池内，训练配置记录池名，报告记录池内日均票数。"""
    data_dir, instruments, days = _write_data_dir(tmp_path)
    _write_index_members(data_dir, days, instruments[:3])
    model_dir = tmp_path / "model"
    trainer = _RecordingTrainer()

    report = tb.train_baseline(
        data_dir,
        model_dir,
        factor_library_dir=FACTOR_LIBRARY,
        start=days[5],
        end=days[-1],
        time_limit=5.0,
        trainer=trainer,
        universe="hs300",
    )

    assert report.summary.universe == "hs300"
    assert report.summary.mean_daily_instruments == pytest.approx(3.0)
    assert report.summary.n_instruments == 3
    assert trainer.rows == report.summary.kept_rows > 0

    loaded = tb.load_train_config(report.config_path)
    assert loaded.universe == "hs300"


# ---------------------------------------------------------------------------
# run_e2e：端到端走通
# ---------------------------------------------------------------------------


def _build_e2e_fixture(
    tmp_path: Path,
    *,
    scores: dict[str, float] | None,
) -> tuple[e2e.E2EConfig, list[date], _FakeTrainer]:
    data_dir, instruments, days = _write_data_dir(tmp_path)
    factors = tb.discover_factors(FACTOR_LIBRARY)
    _write_train_config(
        tmp_path / "model",
        feature_columns=list(factors),
        start=days[0],
        end=days[-1],
        n_rows=100,
    )
    trainer = _FakeTrainer(scores if scores is not None else _score_map(instruments))
    return _e2e_config(tmp_path, data_dir, days), days, trainer


def test_run_e2e_generates_orders_and_writes_artifacts(tmp_path: Path) -> None:
    config, _days, trainer = _build_e2e_fixture(tmp_path, scores=None)
    optimizer = PortfolioOptimizer(lam=1.0, kappa=0.002, w_max=0.5, max_turnover=0.30)

    result = e2e.run_e2e(config, trainer=trainer, optimizer=optimizer)

    # 编排真实被调：优化器求解过，且产生了成交。
    assert result.stats.calls > 0
    assert result.stats.optimize_calls > 0
    assert len(result.result.fills) > 0
    assert result.metrics["trading_days"] > 0
    assert result.metrics["final_nav"] > 0

    for path in (result.nav_path, result.report_path, result.states_path, result.png_path):
        assert path.exists(), f"缺少产物 {path}"
    assert pl.read_parquet(result.nav_path).height == result.metrics["trading_days"]

    assert result.holdings_path is not None and result.holdings_path.exists()
    holdings = pl.read_parquet(result.holdings_path)
    assert set(holdings.columns) == {"date", "instrument", "weight"}
    assert holdings.height > 0
    assert holdings["date"].null_count() == 0

    report_text = result.report_path.read_text(encoding="utf-8")
    assert "## 配置" in report_text
    assert "initial_cash" in report_text
    assert "top_k" in report_text
    assert "lookback_days" in report_text

    states_text = result.states_path.read_text(encoding="utf-8")
    assert "持仓明细" in states_text
    assert "当日成交" in states_text
    assert "### 勾稽" in states_text


def test_run_e2e_empty_scores_produces_no_orders(tmp_path: Path) -> None:
    config, _days, trainer = _build_e2e_fixture(tmp_path, scores=None)
    trainer = _FakeTrainer(None)  # 全 null 打分
    optimizer = PortfolioOptimizer(w_max=0.5, max_turnover=0.30)

    result = e2e.run_e2e(config, trainer=trainer, optimizer=optimizer)

    assert result.stats.empty_score_days > 0
    assert result.result.fills == []
    assert result.metrics["n_fills"] == 0
    assert result.metrics["final_nav"] == pytest.approx(config.initial_cash)
    assert result.report_path.exists()
    assert "没有成交" in result.states_path.read_text(encoding="utf-8")


def test_run_e2e_infeasible_falls_back_to_hold(tmp_path: Path) -> None:
    config, _days, trainer = _build_e2e_fixture(tmp_path, scores=None)
    optimizer = _InfeasibleOptimizer()

    result = e2e.run_e2e(config, trainer=trainer, optimizer=optimizer)  # type: ignore[arg-type]

    assert len(optimizer.calls) > 0, "优化器必须被调用"
    assert result.stats.relaxed_attempts > 0
    assert result.stats.hold_fallback > 0
    assert result.stats.notes, "降级应记录 warning"
    assert result.result.fills == []
    assert result.metrics["n_fills"] == 0
    assert "保持现状" in result.report_path.read_text(encoding="utf-8")


def test_run_e2e_feature_mismatch_raises(tmp_path: Path) -> None:
    data_dir, _instruments, days = _write_data_dir(tmp_path)
    _write_train_config(
        tmp_path / "model",
        feature_columns=["not_a_factor"],
        start=days[0],
        end=days[-1],
    )
    config = _e2e_config(tmp_path, data_dir, days)
    trainer = _FakeTrainer({})

    with pytest.raises(e2e.E2EError, match="特征列"):
        e2e.run_e2e(config, trainer=trainer)


def test_run_e2e_missing_config_without_trainer_raises(tmp_path: Path) -> None:
    data_dir, _instruments, days = _write_data_dir(tmp_path)
    config = _e2e_config(tmp_path, data_dir, days)
    with pytest.raises(e2e.E2EError, match="训练配置"):
        e2e.run_e2e(config)


def test_run_e2e_universe_mismatch_raises(tmp_path: Path) -> None:
    """训练配置有 universe 而回测没给 → 报错（训练在池内 z-score，不能混用）。"""
    data_dir, instruments, days = _write_data_dir(tmp_path)
    _write_index_members(data_dir, days, instruments[:3])
    factors = tb.discover_factors(FACTOR_LIBRARY)
    _write_train_config(
        tmp_path / "model",
        feature_columns=list(factors),
        start=days[0],
        end=days[-1],
        universe="hs300",
    )
    config = _e2e_config(tmp_path, data_dir, days)  # universe 缺省 None

    with pytest.raises(e2e.E2EError, match="universe"):
        e2e.run_e2e(config, trainer=_FakeTrainer({}))


def test_run_e2e_universe_filters_and_reports(tmp_path: Path) -> None:
    """回测与训练同池时跑通，报告记录池内日均票数。"""
    data_dir, instruments, days = _write_data_dir(tmp_path)
    _write_index_members(data_dir, days, instruments[:3])
    factors = tb.discover_factors(FACTOR_LIBRARY)
    _write_train_config(
        tmp_path / "model",
        feature_columns=list(factors),
        start=days[0],
        end=days[-1],
        universe="hs300",
    )
    config = replace(_e2e_config(tmp_path, data_dir, days), universe="hs300")
    trainer = _FakeTrainer(_score_map(instruments))
    optimizer = PortfolioOptimizer(w_max=0.5, max_turnover=0.30)

    result = e2e.run_e2e(config, trainer=trainer, optimizer=optimizer)

    assert result.result.fills, "池内回测应产生成交"
    assert result.stats.calls > 0
    report_text = result.report_path.read_text(encoding="utf-8")
    assert "| universe | hs300 |" in report_text
    assert "| universe_mean_daily_instruments | 3.00 |" in report_text
    assert "| train.universe | hs300 |" in report_text


def test_run_e2e_benchmark_wiring(tmp_path: Path) -> None:
    """``--benchmark`` 接线：算超额绩效、落 benchmark.parquet、报告含超额指标。"""
    config, days, trainer = _build_e2e_fixture(tmp_path, scores=None)
    _write_index_bars(config.data_dir, days)
    config = replace(config, benchmark="000905")
    optimizer = PortfolioOptimizer(lam=1.0, kappa=0.002, w_max=0.5, max_turnover=0.30)

    result = e2e.run_e2e(config, trainer=trainer, optimizer=optimizer)

    assert result.benchmark is not None
    assert result.benchmark.n_days > 0
    assert result.benchmark_path is not None
    assert result.benchmark_path.exists()
    series = pl.read_parquet(result.benchmark_path)
    assert {"benchmark_nav", "excess_nav", "excess_ret"} <= set(series.columns)
    assert series.height == result.benchmark.n_days

    report_text = result.report_path.read_text(encoding="utf-8")
    for token in ("基准指数", "跟踪误差", "信息比率", "000905"):
        assert token in report_text, f"报告缺少 {token}"
    assert result.png_path.exists()


def test_run_e2e_benchmark_missing_data_raises(tmp_path: Path) -> None:
    config, _days, trainer = _build_e2e_fixture(tmp_path, scores=None)
    config = replace(config, benchmark="000905")
    with pytest.raises(e2e.E2EError, match="index_bars"):
        e2e.run_e2e(config, trainer=trainer)


def test_run_e2e_risk_report_wiring(tmp_path: Path) -> None:
    """``--risk-report`` 接线：落 holdings.parquet 与 risk_report.md，失败不影响回测。"""
    config, _days, trainer = _build_e2e_fixture(tmp_path, scores=None)
    config = replace(config, risk_report=True)
    optimizer = PortfolioOptimizer(lam=1.0, kappa=0.002, w_max=0.5, max_turnover=0.30)

    result = e2e.run_e2e(config, trainer=trainer, optimizer=optimizer)

    assert result.risk_report_path is not None
    assert result.risk_report_path.exists()
    text = result.risk_report_path.read_text(encoding="utf-8")
    assert "## 指数分布" in text


# ---------------------------------------------------------------------------
# run_e2e：指增分派（issue #71）
# ---------------------------------------------------------------------------


def _enhanced_config(
    tmp_path: Path,
    data_dir: Path,
    days: list[date],
    *,
    top_k: int = 4,
    strategy_config_path: Path | None = None,
    rebalance_freq: str = "D",
) -> e2e.E2EConfig:
    return replace(
        _e2e_config(tmp_path, data_dir, days, top_k=top_k),
        strategy="index_enhanced",
        universe="zz1000",
        benchmark="000852",
        rebalance_freq=rebalance_freq,
        strategy_config_path=strategy_config_path,
    )


def _write_enhanced_fixture(
    tmp_path: Path, *, rebalance_freq: str = "D"
) -> tuple[e2e.E2EConfig, list[str], _FakeTrainer]:
    """指增用合成数据：池内 4 票同时是基准成分，行业 / 流通市值齐备。"""
    data_dir, instruments, days = _write_data_dir(tmp_path)
    pool = instruments[:4]
    _write_index_members(data_dir, days, pool, code="000852")
    _write_index_weights(data_dir, days, pool, code="000852")
    _write_index_bars(data_dir, days, code="000852")
    _write_industry(data_dir, instruments)
    _write_float_mv(data_dir, days, pool)

    factors = tb.discover_factors(FACTOR_LIBRARY)
    _write_train_config(
        tmp_path / "model",
        feature_columns=list(factors),
        start=days[0],
        end=days[-1],
        n_rows=100,
        universe="zz1000",
    )
    trainer = _FakeTrainer(_score_map(instruments))
    config = _enhanced_config(
        tmp_path, data_dir, days, top_k=len(pool), rebalance_freq=rebalance_freq
    )
    return config, pool, trainer


def _loose_enhanced_optimizer() -> EnhancedOptimizer:
    """合成样本只有 4 票，带约束按相对基准放得很宽，只验证分派与求解链路。"""
    return EnhancedOptimizer(
        stock_band=0.05,
        cover_rate_min=0.5,
        turnover_max=1.0,
        industry_exposure=100.0,
        market_value_exposure=100.0,
        style_exposure=100.0,
    )


def test_run_e2e_index_enhanced_dispatch_and_log(tmp_path: Path) -> None:
    """指增分派走通：真实 EnhancedOptimizer 求解、落 enhanced_log.parquet、报告含指增核对。"""
    config, pool, trainer = _write_enhanced_fixture(tmp_path)

    result = e2e.run_e2e(
        config, trainer=trainer, enhanced_optimizer=_loose_enhanced_optimizer()
    )

    assert result.result.fills, "指增路径应产生成交"
    assert result.stats.calls > 0
    assert result.stats.optimize_calls > 0

    # 产出的日志表：逐调仓日一行，覆盖度 / 个股带 / 放松轮数齐备。
    assert result.enhanced_log_path is not None and result.enhanced_log_path.exists()
    log = pl.read_parquet(result.enhanced_log_path)
    assert set(e2e.ENHANCED_LOG_SCHEMA) <= set(log.columns)
    assert log.height == result.enhanced_stats.rebalance_days
    assert log["n_candidates"].min() >= len(pool)
    assert log["cover_rate"].min() >= 0.5
    assert result.enhanced_stats.cover_rate_min >= 0.5
    assert result.enhanced_stats.band_violations == 0
    assert result.enhanced_stats.held_days == 0
    assert result.enhanced_stats.industry_fallback_days == 0

    report_text = result.report_path.read_text(encoding="utf-8")
    assert "| strategy | index_enhanced |" in report_text
    assert "### 指增核对（index_enhanced）" in report_text
    assert "最小成分覆盖度" in report_text


def test_run_e2e_index_enhanced_requires_benchmark(tmp_path: Path) -> None:
    config, _pool, trainer = _write_enhanced_fixture(tmp_path)
    config = replace(config, benchmark=None)
    with pytest.raises(e2e.E2EError, match="benchmark"):
        e2e.run_e2e(config, trainer=trainer)


def test_run_e2e_index_enhanced_conflicting_strategy_config(tmp_path: Path) -> None:
    """``--strategy-config`` 与命令行不一致时报错，避免两套口径混用。"""
    config, _pool, trainer = _write_enhanced_fixture(tmp_path)
    path = tmp_path / "strategy.json"
    path.write_text(
        json.dumps(
            {
                "strategy": "stock_selection",
                "universe": "zz1000",
                "benchmark": "000852",
                "top_k": 4,
                "rebalance_freq": "D",
            }
        ),
        encoding="utf-8",
    )
    config = replace(config, strategy_config_path=path)
    with pytest.raises(e2e.E2EError, match="不一致"):
        e2e.run_e2e(config, trainer=trainer)


def test_run_e2e_index_enhanced_strategy_config_matches(tmp_path: Path) -> None:
    """``--strategy-config`` 与命令行一致时按 JSON 的 optimize 构造优化器。"""
    config, _pool, trainer = _write_enhanced_fixture(tmp_path)
    path = tmp_path / "strategy.json"
    path.write_text(
        json.dumps(
            {
                "strategy": "index_enhanced",
                "universe": "zz1000",
                "benchmark": "000852",
                "top_k": 4,
                "rebalance_freq": "D",
                "optimize": {
                    "stock_band": 0.05,
                    "cover_rate_min": 0.5,
                    "turnover_max": 1.0,
                    "industry_exposure": 100.0,
                    "market_value_exposure": 100.0,
                    "style_exposure": 100.0,
                },
            }
        ),
        encoding="utf-8",
    )
    config = replace(config, strategy_config_path=path)

    result = e2e.run_e2e(config, trainer=trainer)

    assert result.result.fills
    # 报告记录优化器参数，且个股带按 JSON 的 0.05 判定。
    report_text = result.report_path.read_text(encoding="utf-8")
    assert "strategy.optimize" in report_text
    assert "带 ±0.0500" in report_text


def test_run_e2e_stock_selection_default_unchanged(tmp_path: Path) -> None:
    """默认策略仍是 stock_selection：不落 enhanced_log，报告无指增段。"""
    config, _days, trainer = _build_e2e_fixture(tmp_path, scores=None)
    optimizer = PortfolioOptimizer(lam=1.0, kappa=0.002, w_max=0.5, max_turnover=0.30)

    result = e2e.run_e2e(config, trainer=trainer, optimizer=optimizer)

    assert result.enhanced_log is None
    assert result.enhanced_log_path is None
    assert result.enhanced_stats is None
    assert "指增核对" not in result.report_path.read_text(encoding="utf-8")


def test_band_check_flags_violation() -> None:
    bench = {"A": 0.5, "B": 0.5}
    # 完全贴基准：无偏离。
    assert e2e._band_check(dict(bench), bench, 0.005) == (0.0, 0)
    # A 超配 0.01，超出 0.005 带 + 松弛；满仓约束下超配必然伴随另一只低配，故两只都算超带。
    dev, violations = e2e._band_check({"A": 0.51, "B": 0.49}, bench, 0.005)
    assert dev == pytest.approx(0.01, abs=1e-12)
    assert violations == 2
    # 空基准：不产生判定。
    assert e2e._band_check({"A": 1.0}, {}, 0.005) == (0.0, 0)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def test_relax_turnover_takes_floor() -> None:
    optimizer = PortfolioOptimizer(max_turnover=0.30)
    relaxed = e2e.relax_turnover(optimizer)
    # 0.3 × 3 = 0.9 < 1，取下限 1.0，保证空仓首次建仓可行。
    assert relaxed.max_turnover == pytest.approx(1.0)

    wider = e2e.relax_turnover(PortfolioOptimizer(max_turnover=0.5))
    assert wider.max_turnover == pytest.approx(1.5)


def test_sample_days_picks_first_middle_last() -> None:
    days = _open_days(9)
    assert e2e._sample_days(days, 3) == [days[0], days[4], days[8]]
    assert e2e._sample_days(days[:2], 3) == days[:2]


def test_select_returns_aligns_columns_and_fills_zero() -> None:
    instruments = _instruments(2)
    days = _open_days(6)
    bars = _bars(instruments, days)
    panel = e2e.build_returns_panel(bars)
    frame = e2e.select_returns(panel, [instruments[1], instruments[0], "999999.SH"], days[-1], 5)
    assert list(frame.columns) == [instruments[1], instruments[0], "999999.SH"]
    assert frame.height <= 5
    # 面板中不存在的证券整列填 0
    assert frame["999999.SH"].to_list() == [0.0] * frame.height
