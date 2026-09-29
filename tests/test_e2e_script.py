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
    INDEX_BARS,
    INSTRUMENT_INFO,
    TRADE_CALENDAR,
)
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

    report_text = result.report_path.read_text(encoding="utf-8")
    assert "## 配置" in report_text
    assert "initial_cash" in report_text
    assert "top_k" in report_text
    assert "lookback_days" in report_text

    states_text = result.states_path.read_text(encoding="utf-8")
    assert "持仓明细" in states_text
    assert "当日成交" in states_text


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
