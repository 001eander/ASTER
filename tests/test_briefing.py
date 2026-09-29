"""``quant.daily.briefing`` 单元测试：简报渲染、因子近端 RankIC、跑批落盘。

沿用 ``tests/test_daily.py`` 的合成数据与假 trainer 模式，不触网、不做真实训练。
"""
from __future__ import annotations

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
from quant.daily.briefing import (
    FactorIC,
    recent_factor_ic,
    render_briefing,
)
from quant.daily.pipeline import (
    FILTERED_SCHEMA,
    ORDER_SCHEMA,
    DailyReport,
    run_daily,
)
from quant.daily.strategy import STRATEGY_INDEX_ENHANCED, StrategyConfig
from quant.daily.virtual_account import HOLDINGS_SCHEMA, ActualFill, VirtualAccount

FACTOR_LIBRARY = Path(__file__).resolve().parents[1] / "factor_library"

START = date(2026, 1, 5)  # 周一
REPORT_DATE = date(2026, 3, 30)


# ---------------------------------------------------------------------------
# 合成数据（与 test_daily 同模式）
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
            .map_elements(
                lambda value: self._score_map.get(str(value), 0.0),
                return_dtype=pl.Float64,
            )
            .alias("score")
        )


def _write_index_weights(
    data_dir: Path, day: date, instruments: list[str], *, code: str = "000905"
) -> None:
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


def _score_map(instruments: list[str]) -> dict[str, float]:
    return {
        instrument: float(len(instruments) - i)
        for i, instrument in enumerate(instruments)
    }


@pytest.fixture()
def fake_trainer(monkeypatch: pytest.MonkeyPatch) -> type[_FakeTrainer]:
    _FakeTrainer.score_map = {}
    monkeypatch.setattr(pipeline_module, "BaselineTrainer", _FakeTrainer)
    return _FakeTrainer


# ---------------------------------------------------------------------------
# 手工报告（渲染用，不跑批）
# ---------------------------------------------------------------------------


def _orders_frame() -> pl.DataFrame:
    rows = [
        {
            "date": REPORT_DATE,
            "instrument": "600000.SH",
            "board": "main",
            "side": "sell",
            "volume": 100,
            "ref_price": 12.5,
            "est_amount": 1250.0,
            "current_volume": 300,
            "target_volume": 200,
        },
        {
            "date": REPORT_DATE,
            "instrument": "600001.SH",
            "board": "cyb",
            "side": "buy",
            "volume": 200,
            "ref_price": 8.0,
            "est_amount": 1600.0,
            "current_volume": 0,
            "target_volume": 200,
        },
    ]
    return pl.DataFrame(rows, schema=ORDER_SCHEMA)


def _filtered_frame() -> pl.DataFrame:
    rows = [
        {
            "date": REPORT_DATE,
            "instrument": "600002.SH",
            "board": "main",
            "side": "buy",
            "volume": 100,
            "ref_price": 9.0,
            "est_amount": 900.0,
            "current_volume": 0,
            "target_volume": 100,
            "reason": "limit_up",
        }
    ]
    return pl.DataFrame(rows, schema=FILTERED_SCHEMA)


def _holdings_frame() -> pl.DataFrame:
    rows = [
        {
            "instrument": "600000.SH",
            "volume": 300,
            "sellable": 300,
            "avg_cost": 10.0,
            "price": 12.5,
            "market_value": 3750.0,
            "weight": 0.5,
        }
    ]
    return pl.DataFrame(rows, schema=HOLDINGS_SCHEMA)


def _sample_report(
    *,
    orders: pl.DataFrame | None = None,
    filtered: pl.DataFrame | None = None,
    holdings: pl.DataFrame | None = None,
) -> DailyReport:
    return DailyReport(
        date=REPORT_DATE,
        account_name="default",
        nav=1_000_000.0,
        cash=500_000.0,
        market_value=500_000.0,
        strategy="stock_selection",
        universe="zz1000",
        benchmark=None,
        turnover=0.2,
        n_candidates=25,
        orders=_orders_frame() if orders is None else orders,
        filtered=_filtered_frame() if filtered is None else filtered,
        holdings=_holdings_frame() if holdings is None else holdings,
        target_weights={"600000.SH": 0.2, "600001.SH": 0.3},
        current_weights={"600000.SH": 0.5},
        exposure={
            "cash": 500_000.0,
            "market_value": 500_000.0,
            "nav": 1_000_000.0,
            "cash_ratio": 0.5,
            "position_ratio": 0.5,
            "n_positions": 1.0,
        },
    )


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------


def test_render_briefing_has_all_sections() -> None:
    report = _sample_report()
    factor_ic = [
        FactorIC(
            name="mom_5",
            window_days=20,
            n_days=18,
            mean_ic=0.03,
            std_ic=0.1,
            icir=0.3,
            win_rate=0.55,
        )
    ]
    text = render_briefing(
        report,
        factor_ic=factor_ic,
        industry={"600000.SH": "信息技术", "600001.SH": "金融"},
        notes=["组合优化不可行，已放宽 max_turnover 重试成功"],
    )

    for section in (
        "## 概览",
        "## 调仓明细",
        "### 被预过滤订单",
        "## 个股敞口",
        "## 行业 / 风险敞口",
        "### 行业持仓分布",
        "## 因子近期表现",
    ):
        assert section in text
    assert "# 每日简报 2026-03-30" in text
    # 订单方向中文化、板块中文化
    assert "买入" in text and "卖出" in text
    assert "创业板" in text
    # 个股与目标 / 当前偏离
    assert "600001.SH" in text
    assert "信息技术" in text
    assert "mom_5" in text
    assert "正向" in text
    assert "放宽 max_turnover" in text


def test_render_briefing_no_orders() -> None:
    report = _sample_report(
        orders=pl.DataFrame(schema=ORDER_SCHEMA),
        filtered=pl.DataFrame(schema=FILTERED_SCHEMA),
        holdings=pl.DataFrame(schema=HOLDINGS_SCHEMA),
    )
    text = render_briefing(report)
    assert "今日无调仓。" in text
    assert "本次没有被预过滤的订单。" in text
    assert "空仓。" in text


def test_render_briefing_shows_filtered_reason() -> None:
    report = _sample_report()
    text = render_briefing(report)
    assert "被预过滤订单" in text
    assert "600002.SH" in text
    assert "涨停" in text
    # 未预过滤订单不应出现在被预过滤表里
    assert "600001.SH" in text


def test_render_briefing_index_enhanced_without_risk_report_notes_extension() -> None:
    report = _sample_report()
    report.strategy = "index_enhanced"
    report.benchmark = "000905"
    text = render_briefing(report)
    assert "风险四表未接入" in text


# ---------------------------------------------------------------------------
# 因子近期表现
# ---------------------------------------------------------------------------


def _drift_bars(
    instruments: list[str], days: list[date], rates: list[float]
) -> pl.DataFrame:
    """每只证券按固定日收益 ``rates[i]`` 演化，open-to-open 标签恰为 ``rates[i]``。"""
    rows: list[dict[str, object]] = []
    for i, instrument in enumerate(instruments):
        level = 10.0
        for day in days:
            rows.append(
                {
                    "date": day,
                    "instrument": instrument,
                    "open": level,
                    "high": level,
                    "low": level,
                    "close": level,
                    "vwap": level,
                    "volume": 1000.0,
                    "amount": level * 1000.0,
                    "adjfactor": 1.0,
                    "limit_up": None,
                    "limit_down": None,
                }
            )
            level *= 1.0 + rates[i]
    return pl.DataFrame(rows, schema=DAILY_BARS).sort(["instrument", "date"])


def _factor_by_rank(data: pl.DataFrame, transform: Any) -> pl.DataFrame:
    instruments = data["instrument"].unique().sort().to_list()
    mapping = {instrument: transform(i) for i, instrument in enumerate(instruments)}
    return data.select("date", "instrument").with_columns(
        pl.col("instrument")
        .replace_strict(mapping, return_dtype=pl.Float64)
        .alias("value")
    )


def test_recent_factor_ic_direction() -> None:
    instruments = _instruments(30)
    days = _open_days(60)
    rates = [0.001 * i for i in range(30)]
    bars = _drift_bars(instruments, days, rates)
    center = (len(instruments) - 1) / 2.0

    factors = {
        "trend": lambda data: _factor_by_rank(data, float),
        "quadratic": lambda data: _factor_by_rank(
            data, lambda i: (i - center) ** 2
        ),
    }
    result = {
        item.name: item
        for item in recent_factor_ic(bars, factors, days[-1], window_days=20)
    }

    trend = result["trend"]
    assert trend.direction == "正向"
    assert trend.mean_ic == pytest.approx(1.0, abs=1e-9)
    assert trend.n_days == 18  # 尾部两日无未来行情

    quadratic = result["quadratic"]
    assert quadratic.mean_ic is not None
    assert abs(quadratic.mean_ic) < 1e-9
    assert quadratic.direction in ("中性", "正向", "反向")


def test_recent_factor_ic_rejects_bad_window() -> None:
    bars = _drift_bars(_instruments(3), _open_days(5), [0.0, 0.01, 0.02])
    with pytest.raises(ValueError):
        recent_factor_ic(bars, {}, date(2026, 1, 9), window_days=0)


# ---------------------------------------------------------------------------
# 跑批接线
# ---------------------------------------------------------------------------


def test_run_daily_writes_briefing(
    tmp_path: Path, fake_trainer: type[_FakeTrainer]
) -> None:
    data_dir, instruments, days = _write_data_dir(tmp_path)
    fake_trainer.score_map = _score_map(instruments)
    reports_dir = tmp_path / "reports"

    report = run_daily(
        data_dir,
        tmp_path / "model",
        account_name="default",
        factor_library_dir=FACTOR_LIBRARY,
        account_dir=tmp_path / "account",
        orders_dir=tmp_path / "orders",
        reports_dir=reports_dir,
        initial_cash=1_000_000.0,
    )

    ref_date = days[-1]
    assert report.briefing_path == reports_dir / f"{ref_date.isoformat()}.md"
    assert report.briefing_path.exists()
    text = report.briefing_path.read_text(encoding="utf-8")
    assert "# 每日简报" in text
    assert "## 调仓明细" in text
    assert "## 因子近期表现" in text
    # 幂等分支不产生简报
    second = run_daily(
        data_dir,
        tmp_path / "model",
        account_name="default",
        factor_library_dir=FACTOR_LIBRARY,
        account_dir=tmp_path / "account",
        orders_dir=tmp_path / "orders",
        reports_dir=reports_dir,
        initial_cash=1_000_000.0,
    )
    assert second.already_ran
    assert second.briefing_path is None


def test_run_daily_dry_run_writes_no_briefing(
    tmp_path: Path, fake_trainer: type[_FakeTrainer]
) -> None:
    data_dir, instruments, days = _write_data_dir(tmp_path)
    fake_trainer.score_map = _score_map(instruments)
    reports_dir = tmp_path / "reports"

    report = run_daily(
        data_dir,
        tmp_path / "model",
        account_name="default",
        factor_library_dir=FACTOR_LIBRARY,
        account_dir=tmp_path / "account",
        orders_dir=tmp_path / "orders",
        reports_dir=reports_dir,
        initial_cash=1_000_000.0,
        dry_run=True,
    )

    assert report.briefing_path is None
    assert not (reports_dir / f"{days[-1].isoformat()}.md").exists()
    # dry_run 不改账户
    assert not VirtualAccount.path_for(tmp_path / "account", "default").exists()


class _FakeEnhancedOptimizer:
    """返回等权解的假指增优化器。"""

    def optimize_day(self, **kwargs: Any) -> Any:
        instruments = list(kwargs["instruments"])
        share = 1.0 / len(instruments)
        return SimpleNamespace(
            weights={instrument: share for instrument in instruments},
            status="optimal",
            turnover=1.0,
            objective=0.0,
        )


def test_run_daily_index_enhanced_briefing_includes_risk_tables(
    tmp_path: Path, fake_trainer: type[_FakeTrainer]
) -> None:
    data_dir, instruments, days = _write_data_dir(tmp_path)
    fake_trainer.score_map = _score_map(instruments)
    bench = instruments[:10]
    _write_index_weights(data_dir, days[-1], bench, code="000905")
    account_dir = tmp_path / "account"
    account = VirtualAccount.create(name="default", initial_cash=1_000_000.0)
    account.apply_actual_fills(
        [ActualFill(instruments[0], "buy", 1000, 10.0, date=days[-2])]
    )
    account.settle_new_day()
    account.last_pipeline_date = days[-2]
    account.save(VirtualAccount.path_for(account_dir, "default"))

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
        account_dir=account_dir,
        orders_dir=tmp_path / "orders",
        reports_dir=tmp_path / "reports",
        initial_cash=1_000_000.0,
        strategy_config=config,
        enhanced_optimizer=_FakeEnhancedOptimizer(),
    )

    assert report.briefing_path is not None
    text = report.briefing_path.read_text(encoding="utf-8")
    assert "### 风险四表" in text
    assert "#### 指数分布" in text
    assert "#### 市值分布" in text
    assert "#### 风格暴露" in text
    assert "#### 主动行业暴露" in text
