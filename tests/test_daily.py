"""``quant.daily`` 单元测试：虚拟账户持久化 / 回录、跑批端到端、降级与订单 diff。

全部用例用合成行情 + 假 trainer，不触网、不做真实训练。数据缓存写到 ``tmp_path``，
形态与 ``data/`` 契约一致（calendar / instruments / bars / corporate_actions /
industry），以便跑批前的 :func:`quant.data.validate.validate` 正常通过。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import polars as pl
import pytest

from quant.data.schema import (
    CORPORATE_ACTIONS,
    DAILY_BARS,
    INDEX_WEIGHTS,
    INDUSTRY,
    INSTRUMENT_INFO,
    TRADE_CALENDAR,
)
from quant.daily import pipeline as pipeline_module
from quant.daily.pipeline import (
    DailyError,
    diff_orders,
    resolve_reference_date,
    run_daily,
)
from quant.daily.strategy import STRATEGY_INDEX_ENHANCED, StrategyConfig
from quant.daily.virtual_account import ActualFill, VirtualAccount
from quant.portfolio.optimizer import OptimizeResult, PortfolioOptimizer

FACTOR_LIBRARY = Path(__file__).resolve().parents[1] / "factor_library"

START = date(2026, 1, 5)  # 周一


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


def _price(i: int, d: int) -> float:
    """温和单调上行 + 小幅震荡，保证涨跌幅远小于涨跌停带。"""
    return 10.0 + 0.5 * i + 0.05 * d + 0.1 * ((d + i) % 3)


def _bars(instruments: list[str], days: list[date]) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for i, instrument in enumerate(instruments):
        for d, day in enumerate(days):
            price = _price(i, d)
            rows.append(
                {
                    "date": day,
                    "instrument": instrument,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "vwap": price,
                    "volume": 1000.0,
                    "amount": price * 1000.0,
                    "adjfactor": 1.0,
                    "limit_up": None,
                    "limit_down": None,
                }
            )
    return pl.DataFrame(rows, schema=DAILY_BARS).sort(["instrument", "date"])


def _write_data_dir(
    root: Path, *, n_instruments: int = 30, n_days: int = 60
) -> tuple[Path, list[str], list[date]]:
    """写出一份最小可用的 ``data/`` 缓存，返回 ``(data_dir, instruments, days)``。"""
    data_dir = root / "data"
    (data_dir / "bars").mkdir(parents=True, exist_ok=True)
    instruments = _instruments(n_instruments)
    days = _open_days(n_days)

    calendar_rows = []
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
    # 行业归属（issue #65）：全量覆盖，跑批前的 validate 覆盖率检查才能通过。
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
    bars = _bars(instruments, days)
    for year in sorted(set(bars["date"].dt.year().to_list())):
        chunk = bars.filter(pl.col("date").dt.year() == year)
        chunk.write_parquet(data_dir / "bars" / f"{int(year):04d}.parquet")
    return data_dir, instruments, days


def _write_index_weights(
    data_dir: Path,
    day: date,
    instruments: list[str],
    *,
    code: str = "000905",
) -> None:
    """写一份日频指数权重（等权），供指增路径读基准。"""
    share = 1.0 / len(instruments)
    rows = [
        {
            "date": day,
            "instrument": instrument,
            "index_code": code,
            "weight": share,
        }
        for instrument in instruments
    ]
    pl.DataFrame(rows, schema=INDEX_WEIGHTS).write_parquet(
        data_dir / "index_weights.parquet"
    )


# ---------------------------------------------------------------------------
# 假 trainer
# ---------------------------------------------------------------------------


class _FakeTrainer:
    """固定打分的假 trainer；``load`` 忽略模型目录。"""

    score_map: dict[str, float] = {}

    def __init__(self, score_map: dict[str, float]) -> None:
        self._score_map = score_map

    @classmethod
    def load(cls, path: Any, **kwargs: Any) -> "_FakeTrainer":
        return cls(dict(cls.score_map))

    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.select("date", "instrument").with_columns(
            pl.col("instrument")
            .map_elements(lambda value: self._score_map.get(str(value), 0.0), return_dtype=pl.Float64)
            .alias("score")
        )


def _score_map(instruments: list[str]) -> dict[str, float]:
    # 代码序越前打分越高，给优化器一个明确的截面排序。
    return {instrument: float(len(instruments) - i) for i, instrument in enumerate(instruments)}


@pytest.fixture()
def fake_trainer(monkeypatch: pytest.MonkeyPatch) -> type[_FakeTrainer]:
    _FakeTrainer.score_map = {}
    monkeypatch.setattr(pipeline_module, "BaselineTrainer", _FakeTrainer)
    return _FakeTrainer


# ---------------------------------------------------------------------------
# 虚拟账户
# ---------------------------------------------------------------------------


def test_virtual_account_roundtrip(tmp_path: Path) -> None:
    account = VirtualAccount.create(name="t", initial_cash=1_000_000.0)
    account.apply_actual_fills(
        [ActualFill("600000.SH", "buy", 100, 10.0, fee=5.0, date=date(2026, 1, 5))]
    )
    account.record_nav(date(2026, 1, 5), {"600000.SH": 11.0})
    account.last_pipeline_date = date(2026, 1, 5)

    path = VirtualAccount.path_for(tmp_path / "account", "t")
    account.save(path)
    assert path.exists()

    loaded = VirtualAccount.load(path)
    assert loaded.cash == pytest.approx(1_000_000.0 - 100 * 10.0 - 5.0)
    position = loaded.positions["600000.SH"]
    assert position.volume == 100
    assert position.sellable == 0
    assert position.cost_basis == pytest.approx(1005.0)
    assert [record.date for record in loaded.nav_history] == [date(2026, 1, 5)]
    assert loaded.nav_history[0].nav == pytest.approx(loaded.cash + 1100.0)
    assert loaded.last_pipeline_date == date(2026, 1, 5)
    assert loaded.fills == account.fills
    # created_at 落盘、updated_at 在 save 时刷新
    assert loaded.created_at
    assert loaded.updated_at


def test_virtual_account_apply_actual_fills_sell_and_settle() -> None:
    account = VirtualAccount.create(initial_cash=100_000.0)
    account.apply_actual_fills(
        [ActualFill("600000.SH", "buy", 200, 10.0, fee=5.0, date=date(2026, 1, 5))]
    )
    assert account.cash == pytest.approx(100_000.0 - 2000.0 - 5.0)
    account.settle_new_day()
    assert account.positions["600000.SH"].sellable == 200

    account.apply_actual_fills(
        [ActualFill("600000.SH", "sell", 200, 12.0, fee=5.5, date=date(2026, 1, 6))]
    )
    assert account.cash == pytest.approx(100_000.0 - 2000.0 - 5.0 + 2400.0 - 5.5)
    assert "600000.SH" not in account.positions
    assert [fill.side for fill in account.fills] == ["buy", "sell"]


def test_virtual_account_load_or_create(tmp_path: Path) -> None:
    path = VirtualAccount.path_for(tmp_path, "default")
    created = VirtualAccount.load_or_create(path, initial_cash=500_000.0)
    assert created.cash == 500_000.0
    created.save(path)

    reloaded = VirtualAccount.load_or_create(path, initial_cash=999.0)
    assert reloaded.cash == 500_000.0


def test_virtual_account_weights_and_stats() -> None:
    account = VirtualAccount.create(initial_cash=100_000.0)
    account.apply_actual_fills(
        [ActualFill("600000.SH", "buy", 1000, 10.0, date=date(2026, 1, 5))]
    )
    prices = {"600000.SH": 20.0}
    # 现金 90,000，持仓市值 20,000，nav 110,000
    assert account.nav(prices) == pytest.approx(110_000.0)
    assert account.weights(prices)["600000.SH"] == pytest.approx(20_000.0 / 110_000.0)

    account.record_nav(date(2026, 1, 5), prices)
    account.record_nav(date(2026, 1, 6), {"600000.SH": 10.0})
    stats = account.stats()
    assert stats["n_days"] == 2
    assert stats["max_drawdown"] < 0
    # 同日重复记录不追加
    account.record_nav(date(2026, 1, 6), {"600000.SH": 10.0})
    assert len(account.nav_history) == 2


# ---------------------------------------------------------------------------
# 订单 diff
# ---------------------------------------------------------------------------


def test_diff_orders_directions_and_volumes() -> None:
    target = {"600000.SH": 300, "600001.SH": 500}
    current = {"600000.SH": 100, "600002.SH": 200}
    frame = diff_orders(target, current)
    rows = {row["instrument"]: row for row in frame.to_dicts()}

    # 手算：600000 目标 300 当前 100 → 买 200；600001 新开 500 → 买 500；
    # 600002 目标 0 当前 200 → 清仓卖 200。
    assert rows["600000.SH"]["side"] == "buy"
    assert rows["600000.SH"]["volume"] == 200
    assert rows["600001.SH"]["side"] == "buy"
    assert rows["600001.SH"]["volume"] == 500
    assert rows["600002.SH"]["side"] == "sell"
    assert rows["600002.SH"]["volume"] == 200
    assert frame.height == 3
    # 卖单排在买单之前（卖出回款供买入使用）
    assert frame["side"].to_list()[0] == "sell"


def test_diff_orders_no_change_is_empty() -> None:
    frame = diff_orders({"600000.SH": 100}, {"600000.SH": 100})
    assert frame.height == 0
    assert frame.columns == ["instrument", "side", "volume"]


# ---------------------------------------------------------------------------
# 端到端
# ---------------------------------------------------------------------------


def test_run_daily_end_to_end(
    tmp_path: Path, fake_trainer: type[_FakeTrainer]
) -> None:
    data_dir, instruments, days = _write_data_dir(tmp_path)
    fake_trainer.score_map = _score_map(instruments)
    account_dir = tmp_path / "account"
    orders_dir = tmp_path / "orders"
    reports_dir = tmp_path / "reports"

    report = run_daily(
        data_dir,
        tmp_path / "model",
        account_name="default",
        factor_library_dir=FACTOR_LIBRARY,
        account_dir=account_dir,
        orders_dir=orders_dir,
        reports_dir=reports_dir,
        initial_cash=1_000_000.0,
    )

    ref_date = days[-1]
    assert report.date == ref_date
    assert not report.already_ran
    assert report.scores.height == len(instruments)
    assert len(report.top_scores) == 10
    # 归一后权重不超过 1，整手取整才不会透支
    assert sum(report.target_weights.values()) <= 1.0 + 1e-9
    assert report.orders.height > 0
    # 每票买入数量符合主板整手规则
    for row in report.orders.iter_rows(named=True):
        assert row["volume"] > 0
        assert row["volume"] % 100 == 0
        assert row["ref_price"] > 0
        assert row["date"] == ref_date
    # 调仓单落盘：parquet + csv
    assert report.orders_path == orders_dir / f"{ref_date.isoformat()}.parquet"
    assert report.orders_path.exists()
    assert report.orders_csv_path is not None and report.orders_csv_path.exists()
    loaded = pl.read_parquet(report.orders_path)
    assert loaded.height == report.orders.height
    # 报告落盘
    assert report.report_path is not None and report.report_path.exists()

    # 账户：记录当日净值与跑批日，持仓不改（真实持仓以回录成交为准）
    account = VirtualAccount.load(VirtualAccount.path_for(account_dir, "default"))
    assert account.last_pipeline_date == ref_date
    assert [record.date for record in account.nav_history] == [ref_date]
    assert account.nav_history[0].nav == pytest.approx(1_000_000.0)
    assert account.positions == {}
    assert report.nav == pytest.approx(1_000_000.0)


def test_run_daily_same_day_is_idempotent(
    tmp_path: Path, fake_trainer: type[_FakeTrainer]
) -> None:
    data_dir, instruments, days = _write_data_dir(tmp_path)
    fake_trainer.score_map = _score_map(instruments)
    account_dir = tmp_path / "account"
    orders_dir = tmp_path / "orders"

    kwargs: dict[str, Any] = dict(
        account_name="default",
        factor_library_dir=FACTOR_LIBRARY,
        account_dir=account_dir,
        orders_dir=orders_dir,
        reports_dir=tmp_path / "reports",
        initial_cash=1_000_000.0,
    )
    first = run_daily(data_dir, tmp_path / "model", **kwargs)
    account_path = VirtualAccount.path_for(account_dir, "default")
    account_bytes = account_path.read_bytes()
    orders_bytes = first.orders_path.read_bytes() if first.orders_path else b""

    second = run_daily(data_dir, tmp_path / "model", **kwargs)

    assert second.already_ran
    assert second.orders.height == 0
    # 账户与调仓单文件均未被二次改动
    assert account_path.read_bytes() == account_bytes
    assert first.orders_path is not None
    assert first.orders_path.read_bytes() == orders_bytes
    account = VirtualAccount.load(account_path)
    assert len(account.nav_history) == 1


# ---------------------------------------------------------------------------
# 降级
# ---------------------------------------------------------------------------


def test_run_daily_relaxes_turnover_when_infeasible(
    tmp_path: Path, fake_trainer: type[_FakeTrainer]
) -> None:
    """空账户首次建仓换手必为 1，超过默认 0.3 → 放宽后成功。"""
    data_dir, instruments, _ = _write_data_dir(tmp_path)
    fake_trainer.score_map = _score_map(instruments)

    report = run_daily(
        data_dir,
        tmp_path / "model",
        factor_library_dir=FACTOR_LIBRARY,
        account_dir=tmp_path / "account",
        orders_dir=tmp_path / "orders",
        reports_dir=tmp_path / "reports",
        initial_cash=1_000_000.0,
    )
    assert report.relaxed
    assert not report.hold_fallback
    assert report.orders.height > 0
    assert any("放宽" in note for note in report.notes)


def test_run_daily_hold_fallback_when_still_infeasible(
    tmp_path: Path, fake_trainer: type[_FakeTrainer]
) -> None:
    """单票上限过低导致容量不足 1，放宽换手也救不回来 → 保持现持仓。"""
    data_dir, instruments, _ = _write_data_dir(tmp_path)
    fake_trainer.score_map = _score_map(instruments)

    # 30 只 × 0.02 × 0.96 < 1，sum(w)=1 不可行，放宽 max_turnover 无效。
    optimizer = PortfolioOptimizer(w_max=0.02, max_turnover=0.5)
    report = run_daily(
        data_dir,
        tmp_path / "model",
        factor_library_dir=FACTOR_LIBRARY,
        account_dir=tmp_path / "account",
        orders_dir=tmp_path / "orders",
        reports_dir=tmp_path / "reports",
        initial_cash=1_000_000.0,
        optimizer=optimizer,
    )
    assert report.hold_fallback
    assert report.orders.height == 0
    assert any("保持现有持仓" in note for note in report.notes)
    # 账户仍标记已跑批，净值照记
    account = VirtualAccount.load(
        VirtualAccount.path_for(tmp_path / "account", "default")
    )
    assert account.last_pipeline_date == report.date


# ---------------------------------------------------------------------------
# 策略分派
# ---------------------------------------------------------------------------


@dataclass
class _RecordingOptimizer:
    """记录候选集并返回等权解的假 stock optimizer。"""

    calls: list[list[str]] = field(default_factory=list)
    w_max: float = 0.05
    max_turnover: float = 0.30

    def optimize(
        self,
        alpha: Any,
        instruments: Any,
        returns: Any,
        w_prev: Any = None,
        **kwargs: Any,
    ) -> OptimizeResult:
        candidates = list(instruments)
        self.calls.append(candidates)
        share = 1.0 / len(candidates)
        return OptimizeResult(
            weights={instrument: share for instrument in candidates},
            objective=0.0,
            turnover=1.0,
            status="optimal",
        )


class _FakeEnhancedOptimizer:
    """记录 ``optimize_day`` 入参并返回等权解的假指增优化器。"""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def optimize_day(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        instruments = list(kwargs["instruments"])
        share = 1.0 / len(instruments)
        return SimpleNamespace(
            weights={instrument: share for instrument in instruments},
            status="optimal",
            turnover=1.0,
            objective=0.0,
        )


def test_run_daily_stock_selection_strategy_config(
    tmp_path: Path, fake_trainer: type[_FakeTrainer]
) -> None:
    """stock_selection 配置走假 optimizer，候选数受 top_k 限制。"""
    data_dir, instruments, days = _write_data_dir(tmp_path)
    fake_trainer.score_map = _score_map(instruments)
    recorder = _RecordingOptimizer()
    config = StrategyConfig(strategy="stock_selection", top_k=5)

    report = run_daily(
        data_dir,
        tmp_path / "model",
        factor_library_dir=FACTOR_LIBRARY,
        account_dir=tmp_path / "account",
        orders_dir=tmp_path / "orders",
        reports_dir=tmp_path / "reports",
        initial_cash=1_000_000.0,
        strategy_config=config,
        optimizer=recorder,
    )

    assert report.strategy == "stock_selection"
    assert report.universe is None
    assert recorder.calls, "注入的 optimizer 必须被调用"
    assert len(recorder.calls[0]) == 5  # top_k ∪ 空持仓
    assert report.n_candidates == 5
    assert report.orders.height > 0


def test_run_daily_index_enhanced_strategy_config(
    tmp_path: Path, fake_trainer: type[_FakeTrainer]
) -> None:
    """index_enhanced 配置走假 enhanced optimizer，基准成分进候选集。"""
    data_dir, instruments, days = _write_data_dir(tmp_path)
    fake_trainer.score_map = _score_map(instruments)
    bench = instruments[:10]
    _write_index_weights(data_dir, days[-1], bench, code="000905")
    enhanced = _FakeEnhancedOptimizer()
    config = StrategyConfig(
        strategy=STRATEGY_INDEX_ENHANCED,
        benchmark="000905",
        top_k=5,
        rebalance_freq="D",
    )

    report = run_daily(
        data_dir,
        tmp_path / "model",
        factor_library_dir=FACTOR_LIBRARY,
        account_dir=tmp_path / "account",
        orders_dir=tmp_path / "orders",
        reports_dir=tmp_path / "reports",
        initial_cash=1_000_000.0,
        strategy_config=config,
        enhanced_optimizer=enhanced,
    )

    assert report.strategy == STRATEGY_INDEX_ENHANCED
    assert report.benchmark == "000905"
    assert report.rebalance_freq == "D"
    assert enhanced.calls, "注入的 enhanced optimizer 必须被调用"
    kwargs = enhanced.calls[0]
    assert set(kwargs["bench_weights"]) == set(bench)
    assert set(bench) <= set(kwargs["instruments"])
    assert kwargs["w_prev"] is None
    assert report.orders.height > 0


# ---------------------------------------------------------------------------
# 边界
# ---------------------------------------------------------------------------


def test_resolve_reference_date_non_trading(tmp_path: Path) -> None:
    data_dir, _, days = _write_data_dir(tmp_path, n_days=5)
    # 请求周六：应回退到最近的周五
    saturday = days[-1] + timedelta(days=1)
    assert saturday.weekday() == 5
    assert resolve_reference_date(data_dir, saturday) == days[-1]


def test_discover_factors_loads_library() -> None:
    factors = pipeline_module.discover_factors(FACTOR_LIBRARY)
    assert "mom_5" in factors
    assert "vol_ratio_5" in factors
    assert all(callable(compute) for compute in factors.values())


def test_resolve_reference_date_raises_without_calendar(tmp_path: Path) -> None:
    with pytest.raises(DailyError):
        resolve_reference_date(tmp_path, date(2026, 1, 5))
