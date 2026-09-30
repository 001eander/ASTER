"""``quant.eval.factor`` 的单元测试：因子级评估管线与 CLI。

全部使用合成数据，不触网、不读 ``data/`` 真实缓存。合成面板 60 只证券 × 300 个交易日，
信号 ``sig`` 与次日收益挂钩，由收益链反推开盘价链，使得 ``value = vwap`` 的因果因子
拿到稳定正 RankIC。
"""
from __future__ import annotations

import datetime as dt
import json
import math
import random
from pathlib import Path

import polars as pl
import pytest

from quant.data.schema import DAILY_BARS
from quant.eval import factor as factor_module
from quant.eval.factor import (
    ICIR_MIN,
    MAX_CORR_REJECT,
    MONO_MIN,
    RANK_IC_MIN,
    FactorEvaluation,
    evaluate_factor,
    main,
)
from quant.factor_api.truncation import TruncationResult
from quant.factor_lib.correlation import CorrelationReport

# ---------------------------------------------------------------------------
# 合成数据
# ---------------------------------------------------------------------------

N_INSTRUMENTS: int = 60
N_DAYS: int = 300
START: dt.date = dt.date(2021, 9, 29)
#: 次日收益对信号的暴露，0.4 保证 RankIC 明显为正且门控可通过。
SIGNAL_LOADING: float = 0.4
NOISE_LOADING: float = math.sqrt(1.0 - SIGNAL_LOADING**2)
#: 收益幅度，压到 1% 量级以保证价格链恒正。
RETURN_SCALE: float = 0.01
#: vwap 相对基价的信号幅度，仅影响量纲，不影响 RankIC。
SIGNAL_SCALE: float = 0.3
VWAP_BASE: float = 10.0

#: 因子源码：当日 vwap，与合成标签同向的因果因子。
GOOD_FACTOR_SRC: str = '''
"""测试因子：当日 vwap，与合成标签同向。"""
from __future__ import annotations

import polars as pl


def compute(data: pl.DataFrame) -> pl.DataFrame:
    return data.select("date", "instrument", pl.col("vwap").alias("value"))
'''

#: 方向做反的因子：取 vwap 的相反数。
REVERSED_FACTOR_SRC: str = '''
"""测试因子：-vwap，方向与标签相反。"""
from __future__ import annotations

import polars as pl


def compute(data: pl.DataFrame) -> pl.DataFrame:
    return data.select("date", "instrument", (-pl.col("vwap")).alias("value"))
'''

#: 前视因子：用下一日 close，截断重算应检出。
LOOKAHEAD_FACTOR_SRC: str = '''
"""测试因子：下一日 close，故意引入前视。"""
from __future__ import annotations

import polars as pl


def compute(data: pl.DataFrame) -> pl.DataFrame:
    ordered = data.sort(["instrument", "date"])
    return ordered.with_columns(
        pl.col("close").shift(-1).over("instrument").alias("value")
    ).select("date", "instrument", "value")
'''

#: 输出缺列因子：只给 date / instrument，缺 value。
MISSING_COLUMN_FACTOR_SRC: str = '''
"""测试因子：输出缺 value 列。"""
from __future__ import annotations

import polars as pl


def compute(data: pl.DataFrame) -> pl.DataFrame:
    return data.select("date", "instrument")
'''

#: 抛异常因子。
RAISING_FACTOR_SRC: str = '''
"""测试因子：计算直接抛异常。"""
from __future__ import annotations

import polars as pl


def compute(data: pl.DataFrame) -> pl.DataFrame:
    raise RuntimeError("boom")
'''

#: 声明 WARMUP 的因子，用于验证预热期透传。
WARMUP_FACTOR_SRC: str = '''
"""测试因子：声明 WARMUP=10。"""
from __future__ import annotations

import polars as pl

WARMUP: int = 10


def compute(data: pl.DataFrame) -> pl.DataFrame:
    return data.select("date", "instrument", pl.col("vwap").alias("value"))
'''


#: 库因子：与 GOOD_FACTOR 同源（vwap），用于触发正相关查重。
LIB_DUP_FACTOR_SRC: str = '''
"""测试库因子：当日 vwap，与待查因子完全同源。"""
from __future__ import annotations

import polars as pl


def compute(data: pl.DataFrame) -> pl.DataFrame:
    return data.select("date", "instrument", pl.col("vwap").alias("value"))
'''

#: 库因子：-vwap，与待查因子完全负相关。
LIB_NEG_FACTOR_SRC: str = '''
"""测试库因子：-vwap，与待查因子完全负相关。"""
from __future__ import annotations

import polars as pl


def compute(data: pl.DataFrame) -> pl.DataFrame:
    return data.select("date", "instrument", (-pl.col("vwap")).alias("value"))
'''

#: 库因子：仅按证券序号取值，时不变，与日内 vwap 噪声近似不相关。
LIB_NOISE_FACTOR_SRC: str = '''
"""测试库因子：证券序号，时不变，近似独立于日内信号。"""
from __future__ import annotations

import polars as pl


def compute(data: pl.DataFrame) -> pl.DataFrame:
    return data.select(
        "date",
        "instrument",
        pl.col("instrument").str.slice(0, 6).cast(pl.Int64).alias("value"),
    )
'''


def _instrument(index: int) -> str:
    """生成形如 ``600000.SH`` 的证券代码。"""
    return f"{600000 + index:06d}.SH"


def _make_bars(seed: int = 20260929) -> pl.DataFrame:
    """合成 60 只 × 300 天的 ``DAILY_BARS``。

    每只证券先生成信号 ``sig`` 与噪声，按 ``ret(t+1) = 0.4 sig(t) + sqrt(1-0.16) noise(t)``
    构造 ``t`` 到 ``t+1`` 的收益，再由收益链累乘出开盘价链；``vwap = VWAP_BASE + 0.3 sig``
    暴露信号，使因子 ``value = vwap`` 的次日 IC 稳定为正。
    """
    rng = random.Random(seed)
    dates = [START + dt.timedelta(days=step) for step in range(N_DAYS)]
    rows: list[dict[str, object]] = []

    for index in range(N_INSTRUMENTS):
        instrument = _instrument(index)
        signals = [rng.gauss(0.0, 1.0) for _ in range(N_DAYS)]
        # returns[d] 为 d-1 到 d 的日收益，令 label(t)=returns[t+2] 依赖 signals[t]。
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


def _write(tmp_path: Path, source: str, name: str = "factor") -> Path:
    """把因子源码写到 ``tmp_path`` 并返回路径。"""
    path = tmp_path / f"{name}.py"
    path.write_text(source, encoding="utf-8")
    return path


def _write_library(root: Path, sources: dict[str, str]) -> Path:
    """在 ``root`` 下写若干因子与一份 registry.json，返回目录。

    ``code_path`` 相对仓库根（即 ``root`` 的父目录）书写，与真实因子库一致。
    """
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
                "status": "pool",
            }
        )
    (root / "registry.json").write_text(
        json.dumps({"version": 1, "factors": entries}), encoding="utf-8"
    )
    return root


def _stub_corr(monkeypatch: pytest.MonkeyPatch, max_corr: float) -> None:
    """把相关性查重阶段替换为可控的常数 ``max_corr``，专测折扣公式。"""
    dummy = pl.DataFrame({"date": [dt.date(2024, 1, 1)], "value": [1.0]})
    monkeypatch.setattr(
        factor_module, "load_library_values", lambda *args, **kwargs: {"stub": dummy}
    )
    monkeypatch.setattr(
        factor_module,
        "max_library_corr",
        lambda *args, **kwargs: CorrelationReport({"stub": max_corr}, max_corr),
    )


# ---------------------------------------------------------------------------
# 管线各阶段
# ---------------------------------------------------------------------------


class TestGatePass:
    def test_good_factor_passes_gate(self, tmp_path: Path) -> None:
        data = _make_bars()
        result = evaluate_factor(_write(tmp_path, GOOD_FACTOR_SRC), data)

        assert isinstance(result, FactorEvaluation)
        assert result.ok is True
        assert result.stage == "done"
        assert result.error is None
        assert result.gate_passed is True
        assert result.metrics["rank_ic_mean"] > RANK_IC_MIN
        assert result.metrics["icir"] > ICIR_MIN
        assert result.metrics["mono"] > MONO_MIN
        assert result.metrics["n_days"] > 0
        assert result.metrics["turnover_mean"] is not None
        assert result.truncation is not None and result.truncation["ok"] is True
        assert result.complexity is not None and result.complexity["ok"] is True

    def test_reversed_factor_fails_gate_but_keeps_negative_score(
        self, tmp_path: Path
    ) -> None:
        data = _make_bars()
        result = evaluate_factor(_write(tmp_path, REVERSED_FACTOR_SRC), data)

        assert result.ok is True
        assert result.stage == "done"
        assert result.gate_passed is False
        assert result.metrics["rank_ic_mean"] < 0.0


class TestCorrelationGate:
    """行为相关性查重接入评估管线：门控拒绝与 max_corr 落盘。"""

    def test_duplicate_library_factor_is_rejected(self, tmp_path: Path) -> None:
        data = _make_bars()
        library = _write_library(tmp_path / "lib_dup", {"dup": LIB_DUP_FACTOR_SRC})

        result = evaluate_factor(
            _write(tmp_path, GOOD_FACTOR_SRC), data, factor_library_dir=library
        )

        assert result.ok is True
        assert result.stage == "done"
        # IC 门控本身通过，被拒只因行为相关性判冗余。
        assert result.metrics["rank_ic_mean"] >= RANK_IC_MIN
        assert result.metrics["max_corr"] == pytest.approx(1.0, abs=1e-9)
        assert result.gate_passed is False

    def test_negative_correlation_is_rejected_by_absolute_value(
        self, tmp_path: Path
    ) -> None:
        data = _make_bars()
        library = _write_library(tmp_path / "lib_neg", {"neg": LIB_NEG_FACTOR_SRC})

        result = evaluate_factor(
            _write(tmp_path, GOOD_FACTOR_SRC), data, factor_library_dir=library
        )

        assert result.ok is True
        assert result.metrics["max_corr"] == pytest.approx(1.0, abs=1e-9)
        assert result.gate_passed is False

    def test_uncorrelated_library_factor_passes(self, tmp_path: Path) -> None:
        data = _make_bars()
        library = _write_library(tmp_path / "lib_noise", {"noise": LIB_NOISE_FACTOR_SRC})

        result = evaluate_factor(
            _write(tmp_path, GOOD_FACTOR_SRC), data, factor_library_dir=library
        )

        assert result.ok is True
        assert result.metrics["max_corr"] is not None
        assert result.metrics["max_corr"] < MAX_CORR_REJECT
        assert result.gate_passed is True

    def test_missing_registry_skips_correlation(self, tmp_path: Path) -> None:
        data = _make_bars()

        result = evaluate_factor(
            _write(tmp_path, GOOD_FACTOR_SRC),
            data,
            factor_library_dir=tmp_path / "lib_absent",
        )

        assert result.ok is True
        assert result.metrics["max_corr"] is None
        assert result.gate_passed is True

    def test_empty_registry_skips_correlation(self, tmp_path: Path) -> None:
        data = _make_bars()
        library = _write_library(tmp_path / "lib_empty", {})

        result = evaluate_factor(
            _write(tmp_path, GOOD_FACTOR_SRC), data, factor_library_dir=library
        )

        assert result.ok is True
        assert result.metrics["max_corr"] is None
        assert result.gate_passed is True

    def test_no_library_dir_max_corr_is_none(self, tmp_path: Path) -> None:
        data = _make_bars()

        result = evaluate_factor(_write(tmp_path, GOOD_FACTOR_SRC), data)

        assert result.metrics["max_corr"] is None
        assert result.gate_passed is True

    @pytest.mark.parametrize(
        ("max_corr", "expected_gate"),
        [
            (0.699999, True),
            (MAX_CORR_REJECT, True),
            (MAX_CORR_REJECT + 1e-6, False),
        ],
    )
    def test_threshold_boundary(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        max_corr: float,
        expected_gate: bool,
    ) -> None:
        data = _make_bars()
        dummy = pl.DataFrame({"date": [dt.date(2024, 1, 1)], "value": [1.0]})
        monkeypatch.setattr(
            factor_module, "load_library_values", lambda *args, **kwargs: {"stub": dummy}
        )
        monkeypatch.setattr(
            factor_module,
            "max_library_corr",
            lambda *args, **kwargs: CorrelationReport({"stub": max_corr}, max_corr),
        )

        result = evaluate_factor(
            _write(tmp_path, GOOD_FACTOR_SRC),
            data,
            factor_library_dir=tmp_path,
        )

        assert result.metrics["max_corr"] == max_corr
        assert result.gate_passed is expected_gate


class TestCollinearityDiscount:
    """``score = quality × (1 − max_corr)``：折扣进数值与 notes，门控只是兜底。"""

    def _evaluate_with_corr(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        max_corr: float,
        *,
        source: str = GOOD_FACTOR_SRC,
    ) -> FactorEvaluation:
        _stub_corr(monkeypatch, max_corr)
        return evaluate_factor(
            _write(tmp_path, source), _make_bars(), factor_library_dir=tmp_path
        )

    def test_score_discounted_by_max_corr(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = self._evaluate_with_corr(tmp_path, monkeypatch, 0.4)

        metrics = result.metrics
        assert metrics["quality"] == metrics["rank_ic_mean"]
        assert metrics["corr_discount"] == pytest.approx(0.6)
        assert factor_module._score_payload(result)["score"] == pytest.approx(
            metrics["rank_ic_mean"] * 0.6
        )

    def test_no_library_keeps_full_score(self, tmp_path: Path) -> None:
        result = evaluate_factor(_write(tmp_path, GOOD_FACTOR_SRC), _make_bars())

        metrics = result.metrics
        assert metrics["max_corr"] is None
        assert metrics["corr_discount"] == 1.0
        assert factor_module._score_payload(result)["score"] == pytest.approx(
            metrics["rank_ic_mean"]
        )

    def test_negative_quality_discounted_keeps_sign(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = self._evaluate_with_corr(
            tmp_path, monkeypatch, 0.4, source=REVERSED_FACTOR_SRC
        )

        quality = result.metrics["rank_ic_mean"]
        assert isinstance(quality, float) and quality < 0.0
        score = factor_module._score_payload(result)["score"]
        assert score < 0.0
        assert score == pytest.approx(quality * 0.6)
        assert abs(score) < abs(quality)

    def test_redundant_factor_rejected_but_still_discounted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        over = MAX_CORR_REJECT + 0.2
        result = self._evaluate_with_corr(tmp_path, monkeypatch, over)

        # 折扣与门控并存：分数已按公式压低，门控再兜底拒绝。
        assert result.metrics["rank_ic_mean"] >= RANK_IC_MIN
        assert result.metrics["corr_discount"] == pytest.approx(1.0 - over)
        assert result.gate_passed is False
        score = factor_module._score_payload(result)["score"]
        assert score == pytest.approx(result.metrics["rank_ic_mean"] * (1.0 - over))
        assert 0.0 < score < result.metrics["rank_ic_mean"]

    def test_max_corr_above_one_is_clamped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = self._evaluate_with_corr(tmp_path, monkeypatch, 1.4)

        assert result.metrics["corr_discount"] == 0.0
        assert factor_module._score_payload(result)["score"] == 0.0
        assert result.gate_passed is False

    def test_notes_explain_quality_and_discount(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = self._evaluate_with_corr(tmp_path, monkeypatch, 0.3)

        notes = factor_module._score_payload(result)["notes"]
        assert "quality=" in notes
        assert "×(1-0.30)=" in notes
        assert "score " in notes

    def test_notes_mark_no_library_as_undiscounted(self, tmp_path: Path) -> None:
        result = evaluate_factor(_write(tmp_path, GOOD_FACTOR_SRC), _make_bars())

        notes = factor_module._score_payload(result)["notes"]
        assert "quality=rank_ic" in notes
        assert "未折扣" in notes


class TestHardFailures:
    def test_lookahead_factor_flags_truncation(self, tmp_path: Path) -> None:
        data = _make_bars()
        result = evaluate_factor(_write(tmp_path, LOOKAHEAD_FACTOR_SRC), data)

        assert result.ok is False
        assert result.stage == "truncation"
        assert result.error is not None and "前视" in result.error
        assert result.truncation is not None
        assert result.truncation["ok"] is False
        assert result.metrics == {}

    def test_missing_output_column_is_schema(self, tmp_path: Path) -> None:
        data = _make_bars()
        result = evaluate_factor(_write(tmp_path, MISSING_COLUMN_FACTOR_SRC), data)

        assert result.ok is False
        assert result.stage == "schema"
        assert result.error is not None and "输出" in result.error

    def test_raising_factor_is_compute(self, tmp_path: Path) -> None:
        data = _make_bars()
        result = evaluate_factor(_write(tmp_path, RAISING_FACTOR_SRC), data)

        assert result.ok is False
        assert result.stage == "compute"
        assert result.error is not None and "boom" in result.error

    def test_missing_factor_file_is_load(self, tmp_path: Path) -> None:
        data = _make_bars()
        result = evaluate_factor(tmp_path / "nope.py", data)

        assert result.ok is False
        assert result.stage == "load"

    def test_complexity_overshoot_short_circuits(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(factor_module, "MAX_AST_NODES", 1)
        data = _make_bars()
        result = evaluate_factor(_write(tmp_path, GOOD_FACTOR_SRC), data)

        assert result.ok is False
        assert result.stage == "complexity"
        assert result.error is not None and "复杂度" in result.error
        assert result.complexity is not None and result.complexity["ok"] is False


class TestWarmupPropagation:
    def test_module_warmup_is_passed_to_truncation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, int] = {}

        def probe(compute, data, *, warmup=0, **kwargs):  # noqa: ANN001, ANN003
            seen["warmup"] = warmup
            return TruncationResult(
                ok=True, n_checks=1, failures=(), skipped_reason=None
            )

        monkeypatch.setattr(factor_module, "check_truncation", probe)
        data = _make_bars()
        result = evaluate_factor(_write(tmp_path, WARMUP_FACTOR_SRC), data)

        assert result.ok is True
        assert seen["warmup"] == 10

    def test_default_warmup_when_absent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, int] = {}

        def probe(compute, data, *, warmup=0, **kwargs):  # noqa: ANN001, ANN003
            seen["warmup"] = warmup
            return TruncationResult(
                ok=True, n_checks=1, failures=(), skipped_reason=None
            )

        monkeypatch.setattr(factor_module, "check_truncation", probe)
        data = _make_bars()
        evaluate_factor(_write(tmp_path, GOOD_FACTOR_SRC), data)

        from quant.factor_api.truncation import DEFAULT_WARMUP_DAYS

        assert seen["warmup"] == DEFAULT_WARMUP_DAYS


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCli:
    def test_cli_writes_score_json_on_success(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        data = _make_bars()
        monkeypatch.setattr(factor_module, "load_bars", lambda *args, **kwargs: data)
        factor_path = _write(tmp_path, GOOD_FACTOR_SRC)
        out_path = tmp_path / "score.json"

        code = main(
            [
                str(factor_path),
                "--data-dir",
                str(tmp_path),
                "--factor-library-dir",
                str(tmp_path / "no_lib"),
                "--out",
                str(out_path),
            ]
        )

        assert code == 0
        payload = json.loads(out_path.read_text(encoding="utf-8"))
        assert payload["higher_is_better"] is True
        assert payload["score"] > 0.0
        assert "过门控" in payload["notes"]
        assert payload["details"]["gate_passed"] is True
        metrics = payload["details"]["metrics"]
        assert set(metrics) == {
            "rank_ic_mean",
            "rank_ic_std",
            "icir",
            "ic_win_rate",
            "n_days",
            "mono",
            "turnover_mean",
            "max_corr",
            "quality",
            "corr_discount",
        }
        assert metrics["max_corr"] is None
        assert metrics["corr_discount"] == 1.0
        assert metrics["quality"] == pytest.approx(metrics["rank_ic_mean"])
        assert payload["score"] == pytest.approx(metrics["rank_ic_mean"])
        assert "max_corr=n/a" in payload["notes"]
        assert "未折扣" in payload["notes"]
        assert payload["details"]["truncation"]["ok"] is True

    def test_cli_reversed_factor_writes_negative_score(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        data = _make_bars()
        monkeypatch.setattr(factor_module, "load_bars", lambda *args, **kwargs: data)
        factor_path = _write(tmp_path, REVERSED_FACTOR_SRC)
        out_path = tmp_path / "score.json"

        code = main(
            [
                str(factor_path),
                "--data-dir",
                str(tmp_path),
                "--factor-library-dir",
                str(tmp_path / "no_lib"),
                "--out",
                str(out_path),
            ]
        )

        assert code == 0
        payload = json.loads(out_path.read_text(encoding="utf-8"))
        assert payload["score"] < 0.0
        assert payload["details"]["gate_passed"] is False
        assert "未过门控" in payload["notes"]

    def test_cli_wires_factor_library_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        data = _make_bars()
        monkeypatch.setattr(factor_module, "load_bars", lambda *args, **kwargs: data)
        library = _write_library(tmp_path / "lib_dup", {"dup": LIB_DUP_FACTOR_SRC})
        factor_path = _write(tmp_path, GOOD_FACTOR_SRC)
        out_path = tmp_path / "score.json"

        code = main(
            [
                str(factor_path),
                "--data-dir",
                str(tmp_path),
                "--factor-library-dir",
                str(library),
                "--out",
                str(out_path),
            ]
        )

        assert code == 0
        payload = json.loads(out_path.read_text(encoding="utf-8"))
        assert payload["details"]["metrics"]["max_corr"] == pytest.approx(1.0, abs=1e-9)
        assert payload["details"]["gate_passed"] is False
        assert "max_corr=1.00" in payload["notes"]

    def test_cli_hard_failure_writes_no_score(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        data = _make_bars()
        monkeypatch.setattr(factor_module, "load_bars", lambda *args, **kwargs: data)
        factor_path = _write(tmp_path, LOOKAHEAD_FACTOR_SRC)
        out_path = tmp_path / "score.json"

        code = main(
            [
                str(factor_path),
                "--data-dir",
                str(tmp_path),
                "--factor-library-dir",
                str(tmp_path / "no_lib"),
                "--out",
                str(out_path),
            ]
        )

        assert code == 1
        assert not out_path.exists()
        captured = capsys.readouterr()
        assert "truncation" in captured.err
        assert "前视" in captured.err
