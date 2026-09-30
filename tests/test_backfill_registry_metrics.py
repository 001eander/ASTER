"""``scripts/backfill_registry_metrics.py`` 的单元测试：metrics 写回、自相关排除、null 替换。

全部用合成行情与 ``tmp_path`` 里的内存因子库，不触网、不读真实 ``data/``。
合成面板沿用 ``tests/test_eval_factor.py`` 的口径：60 只 × 300 天，``vwap`` 与
次日收益挂钩，因子 ``value = vwap`` 拿到稳定正 RankIC，可以真正跑完整评估管线。
"""
from __future__ import annotations

import datetime as dt
import importlib.util
import json
import math
import random
import types
from pathlib import Path

import polars as pl
import pytest

from quant.data.schema import DAILY_BARS
from quant.eval.factor import MAX_CORR_REJECT
from quant.factor_lib.registry import load_registry, save_registry
from quant.factor_lib.schema import STATUS_POOL, Registry

# ---------------------------------------------------------------------------
# 合成数据与因子源码
# ---------------------------------------------------------------------------

N_INSTRUMENTS: int = 60
N_DAYS: int = 300
START: dt.date = dt.date(2021, 9, 29)
SIGNAL_LOADING: float = 0.4
NOISE_LOADING: float = math.sqrt(1.0 - SIGNAL_LOADING**2)
RETURN_SCALE: float = 0.01
SIGNAL_SCALE: float = 0.3
VWAP_BASE: float = 10.0

#: 测试因子：当日 vwap，与合成标签同向。
GOOD_FACTOR_SRC: str = '''\
"""测试因子：当日 vwap。"""
from __future__ import annotations

import polars as pl


def compute(data: pl.DataFrame) -> pl.DataFrame:
    return data.select("date", "instrument", pl.col("vwap").alias("value"))
'''

#: 与 vwap 近似不相关的库因子：证券序号，时不变。
NOISE_FACTOR_SRC: str = '''\
"""测试因子：证券序号，时不变。"""
from __future__ import annotations

import polars as pl


def compute(data: pl.DataFrame) -> pl.DataFrame:
    return data.select(
        "date",
        "instrument",
        pl.col("instrument").str.slice(0, 6).cast(pl.Float64).alias("value"),
    )
'''


def _instrument(index: int) -> str:
    return f"{600000 + index:06d}.SH"


def _make_bars(seed: int = 20260930) -> pl.DataFrame:
    """合成 60 只 × 300 天的 ``DAILY_BARS``，与 test_eval_factor._make_bars 同口径。"""
    rng = random.Random(seed)
    dates = [START + dt.timedelta(days=step) for step in range(N_DAYS)]
    rows: list[dict[str, object]] = []

    for index in range(N_INSTRUMENTS):
        instrument = _instrument(index)
        signals = [rng.gauss(0.0, 1.0) for _ in range(N_DAYS)]
        returns = [0.0] * N_DAYS
        for day in range(2, N_DAYS):
            noise = rng.gauss(0.0, 1.0)
            returns[day] = RETURN_SCALE * (
                SIGNAL_LOADING * signals[day - 2] + NOISE_LOADING * noise
            )
        opens = [0.0] * N_DAYS
        opens[0] = VWAP_BASE + 0.01 * index
        for day in range(1, N_DAYS):
            opens[day] = opens[day - 1] * (1.0 + returns[day])

        for day in range(N_DAYS):
            open_price = opens[day]
            close = open_price * 1.001
            vwap = VWAP_BASE + SIGNAL_SCALE * signals[day]
            volume = 1_000_000.0 + 1000.0 * (day % 7)
            rows.append(
                {
                    "date": dates[day],
                    "instrument": instrument,
                    "open": open_price,
                    "high": max(open_price, close) * 1.01,
                    "low": min(open_price, close) * 0.99,
                    "close": close,
                    "vwap": vwap,
                    "volume": volume,
                    "amount": vwap * volume,
                    "adjfactor": 1.0,
                    "limit_up": None,
                    "limit_down": None,
                }
            )
    return pl.DataFrame(rows, schema=DAILY_BARS).sort(["instrument", "date"])


def _write_library(root: Path, sources: dict[str, str]) -> Path:
    """在 ``root`` 下写因子与一份 metrics 全 null 的 registry.json，返回目录。"""
    root.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    for factor_id, source in sources.items():
        (root / f"{factor_id}.py").write_text(source, encoding="utf-8")
        entries.append(
            {
                "factor_id": factor_id,
                "hypothesis": f"{factor_id} 测试库因子",
                "code_path": f"{root.name}/{factor_id}.py",
                "metrics": {"rank_ic": None, "icir": None, "max_corr": None},
                "direction": {
                    "signal_source": "price",
                    "time_scale": "short",
                    "mechanism": "momentum",
                },
                "lineage": {"op": "seed", "parents": [], "run_id": None, "generation": 0},
                "status": STATUS_POOL,
            }
        )
    (root / "registry.json").write_text(
        json.dumps({"version": 1, "factors": entries}, ensure_ascii=False),
        encoding="utf-8",
    )
    return root


def _load_cli() -> types.ModuleType:
    """按路径加载 ``scripts/backfill_registry_metrics.py``（scripts 不是包）。"""
    path = (
        Path(__file__).resolve().parents[1] / "scripts" / "backfill_registry_metrics.py"
    )
    spec = importlib.util.spec_from_file_location("backfill_registry_metrics_cli", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def cli(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """加载 CLI，并把行情加载换成合成面板，保证不读真实数据。"""
    module = _load_cli()
    data = _make_bars()
    monkeypatch.setattr(module, "load_bars", lambda *args, **kwargs: data)
    return module


# ---------------------------------------------------------------------------
# 回填
# ---------------------------------------------------------------------------


def test_backfill_writes_metrics_and_excludes_self(
    cli: types.ModuleType, tmp_path: Path
) -> None:
    library = _write_library(
        tmp_path / "lib_two", {"good": GOOD_FACTOR_SRC, "noise": NOISE_FACTOR_SRC}
    )

    code = cli.main(["--data-dir", str(tmp_path), "--factor-library-dir", str(library)])

    assert code == cli.EXIT_OK
    registry = load_registry(library)

    good = registry.get("good")
    assert good is not None
    # null 全部被替换为数值。
    assert isinstance(good.metrics["rank_ic"], float) and good.metrics["rank_ic"] > 0.0
    assert isinstance(good.metrics["icir"], float)
    # 库内只有 noise 参与查重；自相关若未排除，good 对自身会得到 1.0。
    assert isinstance(good.metrics["max_corr"], float)
    assert good.metrics["max_corr"] < MAX_CORR_REJECT

    noise = registry.get("noise")
    assert noise is not None
    assert isinstance(noise.metrics["rank_ic"], float)
    assert isinstance(noise.metrics["max_corr"], float)


def test_backfill_single_factor_library_has_no_self_correlation(
    cli: types.ModuleType, tmp_path: Path
) -> None:
    library = _write_library(tmp_path / "lib_solo", {"solo": GOOD_FACTOR_SRC})

    code = cli.main(["--data-dir", str(tmp_path), "--factor-library-dir", str(library)])

    assert code == cli.EXIT_OK
    entry = load_registry(library).get("solo")
    assert entry is not None
    # 库里只有自己，剔除自身后无库因子可比，max_corr 应为 None（若未剔除则为 1.0）。
    assert entry.metrics["max_corr"] is None
    assert isinstance(entry.metrics["rank_ic"], float) and entry.metrics["rank_ic"] > 0.0
    assert isinstance(entry.metrics["icir"], float)


def test_backfill_dry_run_does_not_write(cli: types.ModuleType, tmp_path: Path) -> None:
    library = _write_library(tmp_path / "lib_dry", {"good": GOOD_FACTOR_SRC})
    before = (library / "registry.json").read_text(encoding="utf-8")

    code = cli.main(
        ["--data-dir", str(tmp_path), "--factor-library-dir", str(library), "--dry-run"]
    )

    assert code == cli.EXIT_OK
    assert (library / "registry.json").read_text(encoding="utf-8") == before


def test_backfill_preserves_extra_metric_keys(cli: types.ModuleType, tmp_path: Path) -> None:
    library = _write_library(tmp_path / "lib_extra", {"good": GOOD_FACTOR_SRC})
    raw = json.loads((library / "registry.json").read_text(encoding="utf-8"))
    raw["factors"][0]["metrics"]["custom"] = 0.25
    save_registry(Registry.from_dict(raw), library)

    code = cli.main(["--data-dir", str(tmp_path), "--factor-library-dir", str(library)])

    assert code == cli.EXIT_OK
    entry = load_registry(library).get("good")
    assert entry is not None
    assert entry.metrics["custom"] == pytest.approx(0.25)
    assert isinstance(entry.metrics["rank_ic"], float)


def test_backfill_missing_registry_returns_registry_error(
    cli: types.ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(
        ["--data-dir", str(tmp_path), "--factor-library-dir", str(tmp_path / "nope")]
    )

    assert code == cli.EXIT_REGISTRY_ERROR
    assert "registry 不可用" in capsys.readouterr().err
