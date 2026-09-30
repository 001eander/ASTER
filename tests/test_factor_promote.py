"""``quant.factor_lib.promote`` 与入库 CLI 的单元测试。

覆盖：成功入库（源码落位 + metrics 回填 + lineage 串接 + status=pool）、
``gate_passed=false`` 拒绝、``factor_id`` 重复拒绝、目标 ``code_path`` 已存在拒绝、
``score.json`` / 参数非法拒绝、保存失败回滚、CLI 退出码与 ``discover_factors`` 契约。
全部在 ``tmp_path`` 合成因子库与 ``score.json``，不触网。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from quant.daily import pipeline as pipeline_module
from quant.factor_lib import (
    STATUS_POOL,
    Direction,
    Lineage,
    Registry,
    load_registry,
    registry_path,
    save_registry,
)
from quant.factor_lib.promote import (
    EXIT_CONFLICT,
    EXIT_GATE_REJECTED,
    EXIT_INPUT_ERROR,
    EXIT_OK,
    GateNotPassed,
    PromoteConflict,
    PromoteInputError,
    PromoteRequest,
    main,
    promote_factor,
    read_gate_metrics,
)

#: 最小可用因子源码：输出 ``(date, instrument, value)`` 常量列。
FACTOR_SOURCE: str = '''\
"""临时入库测试因子。"""
from __future__ import annotations

import polars as pl


def compute(data: pl.DataFrame) -> pl.DataFrame:
    return data.select("date", "instrument", pl.lit(0.0).alias("value"))
'''

#: 默认 score.json 指标：与 ``quant.eval.factor`` 的 details.metrics 同构。
DEFAULT_METRICS: dict[str, float] = {
    "rank_ic_mean": 0.05,
    "rank_ic_std": 0.02,
    "icir": 0.4,
    "ic_win_rate": 0.55,
    "n_days": 200,
    "mono": 0.6,
    "turnover_mean": 0.3,
    "max_corr": 0.31,
    "quality": 0.05,
    "corr_discount": 0.69,
}

FACTOR_ID = "vol_ratio_5_20_neg"
PARENT_ID = "vol_ratio_5_20"
RUN_ID = "20260930T101500"


# ---------------------------------------------------------------------------
# 构造辅助
# ---------------------------------------------------------------------------


def _empty_library(root: Path, name: str = "lib_promote") -> Path:
    """建一个空 registry 的临时因子库目录。"""
    library = root / name
    library.mkdir(parents=True)
    save_registry(Registry(version=1, factors=()), library)
    return library


def _write_factor(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(FACTOR_SOURCE, encoding="utf-8")
    return path


def _write_score(
    path: Path,
    *,
    gate_passed: Any = True,
    metrics: Any = None,
    details: Any = "default",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if details == "default":
        details = {
            "gate_passed": gate_passed,
            "metrics": DEFAULT_METRICS if metrics is None else metrics,
            "complexity": None,
            "truncation": None,
        }
    payload = {
        "score": 0.0179,
        "higher_is_better": True,
        "notes": "过门控",
        "details": details,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _request(library: Path, factor: Path, score: Path, **overrides: Any) -> PromoteRequest:
    fields: dict[str, Any] = {
        "factor_path": factor,
        "score_path": score,
        "factor_id": FACTOR_ID,
        "hypothesis": "5/20 日均量比取负，缩量做多",
        "direction": Direction("volume", "short", "volume_ratio"),
        "lineage": Lineage("mutation", (PARENT_ID,), RUN_ID, 2),
        "library_dir": library,
    }
    fields.update(overrides)
    return PromoteRequest(**fields)


def _source(tmp_path: Path, name: str = "factor.py") -> Path:
    return _write_factor(tmp_path / "runs" / "0007" / "solution" / name)


# ---------------------------------------------------------------------------
# 成功入库
# ---------------------------------------------------------------------------


def test_promote_success_registers_entry(tmp_path: Path) -> None:
    library = _empty_library(tmp_path)
    factor = _source(tmp_path)
    score = _write_score(tmp_path / "runs" / "0007" / "score.json")

    result = promote_factor(_request(library, factor, score))

    entry = load_registry(library).get(FACTOR_ID)
    assert entry is not None
    assert entry.factor_id == FACTOR_ID
    assert entry.hypothesis == "5/20 日均量比取负，缩量做多"
    assert entry.status == STATUS_POOL
    assert entry.direction == Direction("volume", "short", "volume_ratio")
    assert entry.lineage == Lineage("mutation", (PARENT_ID,), RUN_ID, 2)
    assert result.entry == entry
    assert result.factor_id == FACTOR_ID


def test_promote_success_copies_source_and_paths(tmp_path: Path) -> None:
    library = _empty_library(tmp_path)
    factor = _source(tmp_path)
    score = _write_score(tmp_path / "runs" / "0007" / "score.json")

    result = promote_factor(_request(library, factor, score))

    destination = library / f"{FACTOR_ID}.py"
    assert destination.read_text(encoding="utf-8") == FACTOR_SOURCE
    assert result.destination == destination
    assert result.registry_file == registry_path(library)
    assert result.entry.code_path == f"{library.name}/{FACTOR_ID}.py"


def test_promote_backfills_metrics_from_score(tmp_path: Path) -> None:
    library = _empty_library(tmp_path)
    factor = _source(tmp_path)
    score = _write_score(tmp_path / "runs" / "0007" / "score.json")

    promote_factor(_request(library, factor, score))

    entry = load_registry(library).get(FACTOR_ID)
    assert entry is not None
    assert entry.metrics == {"rank_ic": 0.05, "icir": 0.4, "max_corr": 0.31}


def test_promote_maps_null_max_corr(tmp_path: Path) -> None:
    library = _empty_library(tmp_path)
    factor = _source(tmp_path)
    metrics = dict(DEFAULT_METRICS)
    metrics["max_corr"] = None
    score = _write_score(tmp_path / "runs" / "0007" / "score.json", metrics=metrics)

    result = promote_factor(_request(library, factor, score))

    assert result.entry.metrics == {"rank_ic": 0.05, "icir": 0.4, "max_corr": None}


def test_promoted_factor_is_discoverable(tmp_path: Path) -> None:
    library = _empty_library(tmp_path)
    factor = _source(tmp_path)
    score = _write_score(tmp_path / "runs" / "0007" / "score.json")

    promote_factor(_request(library, factor, score))

    factors = pipeline_module.discover_factors(library)
    assert FACTOR_ID in factors
    assert callable(factors[FACTOR_ID])


# ---------------------------------------------------------------------------
# 拒绝路径
# ---------------------------------------------------------------------------


def test_promote_rejects_gate_false(tmp_path: Path) -> None:
    library = _empty_library(tmp_path)
    factor = _source(tmp_path)
    score = _write_score(
        tmp_path / "runs" / "0007" / "score.json", gate_passed=False
    )
    before = load_registry(library)

    with pytest.raises(GateNotPassed, match="未过门控"):
        promote_factor(_request(library, factor, score))

    assert not (library / f"{FACTOR_ID}.py").exists()
    assert load_registry(library) == before


def test_promote_rejects_missing_details(tmp_path: Path) -> None:
    library = _empty_library(tmp_path)
    factor = _source(tmp_path)
    score = _write_score(tmp_path / "runs" / "0007" / "score.json", details=None)

    with pytest.raises(PromoteInputError, match="details"):
        promote_factor(_request(library, factor, score))


def test_promote_rejects_missing_score_file(tmp_path: Path) -> None:
    library = _empty_library(tmp_path)
    factor = _source(tmp_path)

    with pytest.raises(PromoteInputError, match="不存在"):
        promote_factor(
            _request(library, factor, tmp_path / "runs" / "0007" / "score.json")
        )


def test_promote_rejects_non_numeric_metric(tmp_path: Path) -> None:
    library = _empty_library(tmp_path)
    factor = _source(tmp_path)
    metrics = dict(DEFAULT_METRICS)
    metrics["icir"] = "high"
    score = _write_score(tmp_path / "runs" / "0007" / "score.json", metrics=metrics)

    with pytest.raises(PromoteInputError, match="数值或 null"):
        promote_factor(_request(library, factor, score))


def test_promote_rejects_duplicate_factor_id(tmp_path: Path) -> None:
    library = _empty_library(tmp_path)
    factor = _source(tmp_path)
    score = _write_score(tmp_path / "runs" / "0007" / "score.json")
    promote_factor(_request(library, factor, score))
    before = load_registry(library)

    with pytest.raises(PromoteConflict, match="已登记"):
        promote_factor(_request(library, factor, score))

    assert load_registry(library) == before


def test_promote_rejects_existing_code_file(tmp_path: Path) -> None:
    library = _empty_library(tmp_path)
    (library / f"{FACTOR_ID}.py").write_text("# 未登记的占位文件\n", encoding="utf-8")
    factor = _source(tmp_path)
    score = _write_score(tmp_path / "runs" / "0007" / "score.json")

    with pytest.raises(PromoteConflict, match="已存在"):
        promote_factor(_request(library, factor, score))

    assert load_registry(library).get(FACTOR_ID) is None


def test_promote_rejects_missing_source(tmp_path: Path) -> None:
    library = _empty_library(tmp_path)
    score = _write_score(tmp_path / "runs" / "0007" / "score.json")

    with pytest.raises(PromoteInputError, match="源码不存在"):
        promote_factor(
            _request(library, tmp_path / "runs" / "0007" / "solution" / "factor.py", score)
        )


def test_promote_rejects_bad_factor_id(tmp_path: Path) -> None:
    library = _empty_library(tmp_path)
    factor = _source(tmp_path)
    score = _write_score(tmp_path / "runs" / "0007" / "score.json")

    with pytest.raises(PromoteInputError, match="factor_id"):
        promote_factor(_request(library, factor, score, factor_id="../escape"))


def test_promote_rolls_back_source_on_save_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library = _empty_library(tmp_path)
    factor = _source(tmp_path)
    score = _write_score(tmp_path / "runs" / "0007" / "score.json")

    def _boom(*args: Any, **kwargs: Any) -> Path:
        raise OSError("磁盘写失败")

    monkeypatch.setattr("quant.factor_lib.promote.save_registry", _boom)
    with pytest.raises(OSError, match="磁盘写失败"):
        promote_factor(_request(library, factor, score))

    assert not (library / f"{FACTOR_ID}.py").exists()
    assert load_registry(library).get(FACTOR_ID) is None


def test_promote_missing_registry_raises(tmp_path: Path) -> None:
    from quant.factor_lib import FactorLibError

    library = tmp_path / "lib_absent"
    library.mkdir()
    factor = _source(tmp_path)
    score = _write_score(tmp_path / "runs" / "0007" / "score.json")

    with pytest.raises(FactorLibError, match="不存在"):
        promote_factor(_request(library, factor, score))


# ---------------------------------------------------------------------------
# read_gate_metrics
# ---------------------------------------------------------------------------


def test_read_gate_metrics_returns_mapped_values(tmp_path: Path) -> None:
    score = _write_score(tmp_path / "score.json")
    assert read_gate_metrics(score) == {"rank_ic": 0.05, "icir": 0.4, "max_corr": 0.31}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli_args(library: Path, factor: Path, score: Path) -> list[str]:
    return [
        "--factor",
        str(factor),
        "--score",
        str(score),
        "--factor-id",
        FACTOR_ID,
        "--hypothesis",
        "5/20 日均量比取负，缩量做多",
        "--signal-source",
        "volume",
        "--time-scale",
        "short",
        "--mechanism",
        "volume_ratio",
        "--op",
        "mutation",
        "--parent",
        PARENT_ID,
        "--run-id",
        RUN_ID,
        "--generation",
        "2",
        "--factor-library-dir",
        str(library),
    ]


def test_cli_success(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    library = _empty_library(tmp_path)
    factor = _source(tmp_path)
    score = _write_score(tmp_path / "runs" / "0007" / "score.json")

    assert main(_cli_args(library, factor, score)) == EXIT_OK

    out = capsys.readouterr().out
    assert FACTOR_ID in out
    assert "registry" in out
    assert load_registry(library).get(FACTOR_ID) is not None


def test_cli_gate_rejected(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    library = _empty_library(tmp_path)
    factor = _source(tmp_path)
    score = _write_score(tmp_path / "runs" / "0007" / "score.json", gate_passed=False)

    assert main(_cli_args(library, factor, score)) == EXIT_GATE_REJECTED
    assert "拒绝入库" in capsys.readouterr().err


def test_cli_conflict(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    library = _empty_library(tmp_path)
    factor = _source(tmp_path)
    score = _write_score(tmp_path / "runs" / "0007" / "score.json")
    assert main(_cli_args(library, factor, score)) == EXIT_OK

    assert main(_cli_args(library, factor, score)) == EXIT_CONFLICT
    assert "拒绝入库" in capsys.readouterr().err


def test_cli_bad_argument(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    library = _empty_library(tmp_path)
    factor = _source(tmp_path)
    score = _write_score(tmp_path / "runs" / "0007" / "score.json")
    args = _cli_args(library, factor, score)
    args[args.index("--generation") + 1] = "-1"

    assert main(args) == EXIT_INPUT_ERROR
    assert "参数非法" in capsys.readouterr().err
