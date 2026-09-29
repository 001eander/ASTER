"""``quant.factor_lib`` 与 ``discover_factors`` registry 接入的单元测试。

覆盖：schema 严格校验的各非法分支、保存 / 加载 roundtrip、pool 过滤、
注册唯一性冲突，以及 ``discover_factors`` 在有 registry 时跳过 graveyard、
registry 缺失时回退目录扫描。全部用 ``tmp_path`` 合成因子，不触网。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from quant.daily import pipeline as pipeline_module
from quant.daily.pipeline import DailyError
from quant.factor_lib import (
    STATUS_GRAVEYARD,
    STATUS_POOL,
    Direction,
    FactorEntry,
    FactorLibError,
    Lineage,
    Registry,
    load_registry,
    pool_factors,
    register_factor,
    registry_path,
    save_registry,
)

REPO_FACTOR_LIBRARY = Path(__file__).resolve().parents[1] / "factor_library"

#: 最小可用因子源码：输出 ``(date, instrument, value)`` 常量列。
FACTOR_SOURCE: str = '''\
"""临时测试因子。"""
from __future__ import annotations

import polars as pl


def compute(data: pl.DataFrame) -> pl.DataFrame:
    return data.select("date", "instrument", pl.lit(0.0).alias("value"))
'''


# ---------------------------------------------------------------------------
# 构造辅助
# ---------------------------------------------------------------------------


def _entry_raw(
    factor_id: str,
    *,
    status: str = STATUS_POOL,
    code_path: str | None = None,
    metrics: dict[str, Any] | None = None,
    direction: dict[str, Any] | None = None,
    lineage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "factor_id": factor_id,
        "hypothesis": f"{factor_id} 的测试假设",
        "code_path": code_path if code_path is not None else f"lib/{factor_id}.py",
        "metrics": {"rank_ic": 0.05, "icir": 0.4, "max_corr": None}
        if metrics is None
        else metrics,
        "direction": {"signal_source": "price", "time_scale": "short", "mechanism": "momentum"}
        if direction is None
        else direction,
        "lineage": {"op": "seed", "parents": [], "run_id": None, "generation": 0}
        if lineage is None
        else lineage,
        "status": status,
    }


def _entry(factor_id: str, **kwargs: Any) -> FactorEntry:
    return FactorEntry.from_dict(_entry_raw(factor_id, **kwargs))


def _registry(*entries: FactorEntry, version: int = 1) -> Registry:
    return Registry(version=version, factors=tuple(entries))


def _write_factor(root: Path, name: str) -> None:
    (root / f"{name}.py").write_text(FACTOR_SOURCE, encoding="utf-8")


# ---------------------------------------------------------------------------
# schema 校验
# ---------------------------------------------------------------------------


def test_parse_valid_registry() -> None:
    registry = Registry.from_dict(
        {"version": 1, "factors": [_entry_raw("f_a"), _entry_raw("f_b")]}
    )
    assert registry.version == 1
    assert [entry.factor_id for entry in registry.factors] == ["f_a", "f_b"]
    entry = registry.factors[0]
    assert entry.metrics["rank_ic"] == 0.05
    assert entry.metrics["max_corr"] is None
    assert entry.direction == Direction("price", "short", "momentum")
    assert entry.lineage == Lineage("seed", (), None, 0)
    assert registry.get("f_b") is not None
    assert registry.get("nope") is None


def test_registry_rejects_missing_entry_key() -> None:
    raw = _entry_raw("f_a")
    del raw["hypothesis"]
    with pytest.raises(FactorLibError, match="缺少字段"):
        Registry.from_dict({"version": 1, "factors": [raw]})


def test_registry_rejects_extra_entry_key() -> None:
    raw = _entry_raw("f_a")
    raw["extra"] = 1
    with pytest.raises(FactorLibError, match="未知字段"):
        Registry.from_dict({"version": 1, "factors": [raw]})


def test_registry_rejects_unknown_top_level_key() -> None:
    with pytest.raises(FactorLibError, match="未知字段"):
        Registry.from_dict({"version": 1, "factors": [], "note": "x"})


def test_registry_rejects_bad_version() -> None:
    with pytest.raises(FactorLibError, match="版本"):
        Registry.from_dict({"version": 2, "factors": []})


def test_registry_rejects_illegal_status() -> None:
    with pytest.raises(FactorLibError, match="status"):
        Registry.from_dict({"version": 1, "factors": [_entry_raw("f_a", status="alive")]})


def test_registry_rejects_duplicate_factor_id() -> None:
    with pytest.raises(FactorLibError, match="重复"):
        Registry.from_dict(
            {"version": 1, "factors": [_entry_raw("f_a"), _entry_raw("f_a")]}
        )


def test_direction_rejects_missing_key() -> None:
    bad = {"signal_source": "price", "time_scale": "short"}
    with pytest.raises(FactorLibError, match="缺少字段"):
        Direction.from_dict(bad)


def test_direction_rejects_non_string_value() -> None:
    bad = {"signal_source": "price", "time_scale": 5, "mechanism": "momentum"}
    with pytest.raises(FactorLibError, match="非空字符串"):
        Direction.from_dict(bad)


def test_metrics_rejects_non_numeric_value() -> None:
    with pytest.raises(FactorLibError, match="数值或 null"):
        _entry("f_a", metrics={"rank_ic": "high"})


def test_lineage_rejects_unknown_op() -> None:
    with pytest.raises(FactorLibError, match="op"):
        _entry("f_a", lineage={"op": "teleport", "parents": [], "run_id": None, "generation": 0})


def test_lineage_rejects_negative_generation() -> None:
    with pytest.raises(FactorLibError, match="generation"):
        _entry(
            "f_a",
            lineage={"op": "mutation", "parents": ["f_b"], "run_id": "r_1", "generation": -1},
        )


def test_lineage_rejects_non_list_parents() -> None:
    with pytest.raises(FactorLibError, match="parents"):
        _entry(
            "f_a",
            lineage={"op": "mutation", "parents": "f_b", "run_id": "r_1", "generation": 1},
        )


# ---------------------------------------------------------------------------
# 读写 roundtrip
# ---------------------------------------------------------------------------


def test_roundtrip_save_load(tmp_path: Path) -> None:
    directory = tmp_path / "lib_roundtrip"
    directory.mkdir()
    original = _registry(
        _entry("f_a"),
        _entry("f_b", status=STATUS_GRAVEYARD, metrics={"rank_ic": None, "icir": None, "max_corr": 0.9}),
    )
    path = save_registry(original, directory)
    assert path == registry_path(directory)
    assert load_registry(directory) == original


def test_load_registry_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FactorLibError, match="不存在"):
        load_registry(tmp_path / "lib_missing")


def test_save_registry_is_atomic_no_leftover_tmp(tmp_path: Path) -> None:
    directory = tmp_path / "lib_atomic"
    save_registry(_registry(_entry("f_a")), directory)
    leftovers = [p.name for p in directory.iterdir() if p.name != "registry.json"]
    assert leftovers == []


# ---------------------------------------------------------------------------
# pool 过滤与注册
# ---------------------------------------------------------------------------


def test_pool_factors_filters_graveyard() -> None:
    registry = _registry(
        _entry("f_a"),
        _entry("f_b", status=STATUS_GRAVEYARD),
        _entry("f_c"),
    )
    assert [entry.factor_id for entry in pool_factors(registry)] == ["f_a", "f_c"]


def test_register_factor_appends() -> None:
    registry = _registry(_entry("f_a"))
    updated = register_factor(registry, _entry("f_b"))
    assert [entry.factor_id for entry in updated.factors] == ["f_a", "f_b"]
    assert registry.get("f_b") is None  # 原 registry 不被修改


def test_register_factor_duplicate_raises() -> None:
    registry = _registry(_entry("f_a"))
    with pytest.raises(FactorLibError, match="已存在"):
        register_factor(registry, _entry("f_a"))


# ---------------------------------------------------------------------------
# discover_factors 接入
# ---------------------------------------------------------------------------


def test_discover_factors_uses_registry_and_skips_graveyard(tmp_path: Path) -> None:
    root = tmp_path / "lib_graveyard"
    root.mkdir()
    _write_factor(root, "keep_me")
    _write_factor(root, "drop_me")
    save_registry(
        _registry(
            _entry("keep_me", code_path="lib_graveyard/keep_me.py"),
            _entry("drop_me", status=STATUS_GRAVEYARD, code_path="lib_graveyard/drop_me.py"),
        ),
        root,
    )

    factors = pipeline_module.discover_factors(root)

    assert set(factors) == {"keep_me"}
    assert callable(factors["keep_me"])


def test_discover_factors_skips_missing_code_file(tmp_path: Path) -> None:
    root = tmp_path / "lib_missing_code"
    root.mkdir()
    _write_factor(root, "present")
    save_registry(
        _registry(
            _entry("present", code_path="lib_missing_code/present.py"),
            _entry("absent", code_path="lib_missing_code/absent.py"),
        ),
        root,
    )

    factors = pipeline_module.discover_factors(root)

    assert set(factors) == {"present"}


def test_discover_factors_falls_back_to_glob_without_registry(tmp_path: Path) -> None:
    root = tmp_path / "lib_glob"
    root.mkdir()
    _write_factor(root, "alpha")
    _write_factor(root, "beta")
    (root / "_helper.py").write_text("VALUE = 1\n", encoding="utf-8")

    factors = pipeline_module.discover_factors(root)

    assert set(factors) == {"alpha", "beta"}


def test_discover_factors_empty_after_registry_raises(tmp_path: Path) -> None:
    root = tmp_path / "lib_all_graveyard"
    root.mkdir()
    _write_factor(root, "dead")
    save_registry(
        _registry(
            _entry("dead", status=STATUS_GRAVEYARD, code_path="lib_all_graveyard/dead.py")
        ),
        root,
    )

    with pytest.raises(DailyError, match="没有可用因子"):
        pipeline_module.discover_factors(root)


# ---------------------------------------------------------------------------
# 真实因子库契约
# ---------------------------------------------------------------------------


def test_real_registry_covers_all_factor_files() -> None:
    registry = load_registry(REPO_FACTOR_LIBRARY)
    pool_ids = {entry.factor_id for entry in pool_factors(registry)}
    file_stems = {path.stem for path in REPO_FACTOR_LIBRARY.glob("*.py")}
    assert pool_ids == file_stems
    assert len(pool_ids) >= 12


def test_real_library_loads_via_registry() -> None:
    factors = pipeline_module.discover_factors(REPO_FACTOR_LIBRARY)
    assert "mom_5" in factors
    assert set(factors) == {
        entry.factor_id for entry in pool_factors(load_registry(REPO_FACTOR_LIBRARY))
    }
