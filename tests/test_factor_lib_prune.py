"""``quant.factor_lib.prune`` 与整库 CLI 的单元测试：聚簇降级、容量上限、幂等。

全部用确定性合成值长表与 ``tmp_path`` 里的内存 registry，不触网、不读真实 data/。
覆盖：完全相关 / 负相关 / 链式相关的聚簇留强、rank_ic 全 None 时的确定性、
容量 50% 规则与其边界、graveyard 不计入 pool 但计入总条目数、apply 的不可变性、
整库幂等、CLI 干跑不写盘与缺失 registry 的退出码。
"""
from __future__ import annotations

import datetime as dt
import importlib.util
import math
import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from quant.factor_lib.prune import (
    REASON_CAPACITY,
    REASON_CLUSTER,
    PruneDemotion,
    apply_prune,
    build_corr_matrix,
    capacity_limit,
    plan_prune,
    prune,
)
from quant.factor_lib.registry import load_registry, save_registry
from quant.factor_lib.schema import (
    STATUS_GRAVEYARD,
    STATUS_POOL,
    FactorEntry,
    FactorLibError,
    Registry,
)

START: dt.date = dt.date(2024, 1, 1)
N_INSTRUMENTS: int = 60
N_DAYS: int = 5


# ---------------------------------------------------------------------------
# 合成数据与 registry 构造
# ---------------------------------------------------------------------------


def _instrument(index: int) -> str:
    return f"{600000 + index:06d}.SH"


def _panel(values: list[float], *, n_instruments: int = N_INSTRUMENTS) -> pl.DataFrame:
    """把截面向量铺成 ``(date, instrument, value)`` 长表。"""
    rows: list[dict[str, Any]] = []
    for day in range(N_DAYS):
        for index in range(n_instruments):
            rows.append(
                {
                    "date": START + dt.timedelta(days=day),
                    "instrument": _instrument(index),
                    "value": values[index],
                }
            )
    return pl.DataFrame(
        rows,
        schema={"date": pl.Date, "instrument": pl.String, "value": pl.Float64},
    )


def _u(index: int) -> float:
    """基准截面向量。"""
    return math.sin(1.7 * index)


def _w(index: int) -> float:
    """与 :func:`_u` 近似正交的截面向量。"""
    return math.cos(1.7 * index)


def _v(index: int) -> float:
    """与 ``u`` 截面相关约 0.2 的向量。"""
    return 0.2 * _u(index) + math.sqrt(0.96) * _w(index)


def _entry(
    factor_id: str,
    *,
    status: str = STATUS_POOL,
    rank_ic: float | None = None,
    icir: float | None = None,
) -> FactorEntry:
    return FactorEntry.from_dict(
        {
            "factor_id": factor_id,
            "hypothesis": f"{factor_id} 的测试假设",
            "code_path": f"lib/{factor_id}.py",
            "metrics": {"rank_ic": rank_ic, "icir": icir, "max_corr": None},
            "direction": {
                "signal_source": "price",
                "time_scale": "short",
                "mechanism": "momentum",
            },
            "lineage": {"op": "seed", "parents": [], "run_id": None, "generation": 0},
            "status": status,
        }
    )


def _registry(*entries: FactorEntry) -> Registry:
    return Registry(version=1, factors=tuple(entries))


def _chain_values() -> dict[str, pl.DataFrame]:
    """A=u、B=u+v、C=v：A-B 与 B-C 高相关，A-C 低相关，三者在同一连通分量。"""
    return {
        "a": _panel([_u(i) for i in range(N_INSTRUMENTS)]),
        "b": _panel([_u(i) + _v(i) for i in range(N_INSTRUMENTS)]),
        "c": _panel([_v(i) for i in range(N_INSTRUMENTS)]),
    }


# ---------------------------------------------------------------------------
# 相关矩阵
# ---------------------------------------------------------------------------


class TestBuildCorrMatrix:
    def test_upper_triangle_only_and_perfect_corr(self) -> None:
        base = _panel([_u(i) for i in range(N_INSTRUMENTS)])
        scaled = _panel([2.0 * _u(i) + 3.0 for i in range(N_INSTRUMENTS)])

        matrix = build_corr_matrix({"a": base, "b": scaled})

        assert list(matrix) == [("a", "b")]
        assert matrix[("a", "b")] == pytest.approx(1.0, abs=1e-9)

    def test_unavailable_pairs_are_skipped(self) -> None:
        tiny = _panel([_u(i) for i in range(5)], n_instruments=5)

        # 逐日截面只有 5 只证券，低于 min_count，相关系数不可得 → 该对跳过。
        assert build_corr_matrix({"a": tiny, "b": tiny}) == {}


# ---------------------------------------------------------------------------
# 聚簇降级
# ---------------------------------------------------------------------------


class TestClusterPrune:
    def test_keeps_higher_rank_ic(self) -> None:
        values = {
            "a": _panel([_u(i) for i in range(N_INSTRUMENTS)]),
            "b": _panel([2.0 * _u(i) + 3.0 for i in range(N_INSTRUMENTS)]),
        }
        registry = _registry(_entry("a", rank_ic=0.05), _entry("b", rank_ic=0.02))

        plan = plan_prune(registry, values)

        assert plan.demotions == (PruneDemotion("b", REASON_CLUSTER, "a"),)
        assert plan.pool_before == 2
        assert plan.pool_after == 1

    def test_chain_component_keeps_single_survivor(self) -> None:
        registry = _registry(
            _entry("a", rank_ic=0.03),
            _entry("b", rank_ic=0.05),
            _entry("c", rank_ic=0.01),
        )

        plan = plan_prune(registry, _chain_values())

        assert {demotion.factor_id for demotion in plan.demotions} == {"a", "c"}
        assert all(demotion.kept == "b" for demotion in plan.demotions)
        assert all(demotion.reason == REASON_CLUSTER for demotion in plan.demotions)

    def test_negative_corr_clusters_by_absolute_value(self) -> None:
        values = {
            "a": _panel([_u(i) for i in range(N_INSTRUMENTS)]),
            "b": _panel([-2.5 * _u(i) for i in range(N_INSTRUMENTS)]),
        }
        registry = _registry(_entry("a", rank_ic=0.01), _entry("b", rank_ic=0.04))

        plan = plan_prune(registry, values)

        assert plan.demotions == (PruneDemotion("a", REASON_CLUSTER, "b"),)

    def test_all_null_rank_ic_falls_back_to_icir(self) -> None:
        values = {
            "a": _panel([_u(i) for i in range(N_INSTRUMENTS)]),
            "b": _panel([2.0 * _u(i) for i in range(N_INSTRUMENTS)]),
        }
        registry = _registry(_entry("a", icir=0.3), _entry("b", icir=0.5))

        plan = plan_prune(registry, values)

        assert plan.demotions == (PruneDemotion("a", REASON_CLUSTER, "b"),)

    def test_all_metrics_null_breaks_tie_by_factor_id(self) -> None:
        values = {
            "a": _panel([_u(i) for i in range(N_INSTRUMENTS)]),
            "b": _panel([2.0 * _u(i) for i in range(N_INSTRUMENTS)]),
        }
        registry = _registry(_entry("a"), _entry("b"))

        plan = plan_prune(registry, values)

        # 每簇恰好保留 1 个，且并列时保留字典序更小的 factor_id。
        assert len(plan.demotions) == 1
        assert plan.demotions[0] == PruneDemotion("b", REASON_CLUSTER, "a")

    def test_missing_values_produces_no_cluster(self) -> None:
        registry = _registry(
            _entry("a", rank_ic=0.05),
            _entry("b", rank_ic=None),
            _entry("dead_1", status=STATUS_GRAVEYARD),
            _entry("dead_2", status=STATUS_GRAVEYARD),
        )

        plan = plan_prune(registry, {})

        # 无相关可得 → 无簇；总条目 4 → 上限 2，pool 恰为 2，容量也不动。
        assert plan.is_empty


# ---------------------------------------------------------------------------
# 容量上限
# ---------------------------------------------------------------------------


class TestCapacity:
    def test_demotes_weakest_until_within_limit(self) -> None:
        registry = _registry(
            _entry("f_a", rank_ic=0.04),
            _entry("f_b", rank_ic=0.03),
            _entry("f_c", rank_ic=0.02),
            _entry("f_d", rank_ic=0.01),
        )

        plan = plan_prune(registry, {})

        assert plan.capacity_limit == 2
        assert {demotion.factor_id for demotion in plan.demotions} == {"f_c", "f_d"}
        assert all(demotion.reason == REASON_CAPACITY for demotion in plan.demotions)
        assert all(demotion.kept is None for demotion in plan.demotions)
        assert plan.pool_after == 2

    def test_boundary_at_limit_keeps_everything(self) -> None:
        registry = _registry(
            _entry("f_a", rank_ic=0.05),
            _entry("f_b", rank_ic=0.01),
            _entry("dead_1", status=STATUS_GRAVEYARD),
            _entry("dead_2", status=STATUS_GRAVEYARD),
        )

        plan = plan_prune(registry, {})

        # 总条目 4 → 上限 2；pool 恰为 2，不再降级。
        assert plan.pool_before == 2
        assert plan.capacity_limit == capacity_limit(4) == 2
        assert plan.is_empty

    def test_graveyard_counts_toward_total_but_not_pool(self) -> None:
        registry = _registry(
            _entry("f_a", rank_ic=0.05),
            _entry("f_b", rank_ic=0.01),
            _entry("dead", status=STATUS_GRAVEYARD),
        )

        plan = plan_prune(registry, {})

        # 总条目 3 → 上限 1；graveyard 不占 pool 名额，所以 pool 2 仍需降 1 个。
        assert plan.pool_before == 2
        assert plan.capacity_limit == 1
        assert [demotion.factor_id for demotion in plan.demotions] == ["f_b"]

    def test_graveyard_does_not_join_clusters(self) -> None:
        values = {
            "f_a": _panel([_u(i) for i in range(N_INSTRUMENTS)]),
            "dead": _panel([2.0 * _u(i) for i in range(N_INSTRUMENTS)]),
        }
        registry = _registry(
            _entry("f_a", rank_ic=0.05),
            _entry("dead", status=STATUS_GRAVEYARD, rank_ic=0.9),
        )

        plan = plan_prune(registry, values)

        # graveyard 即便与 pool 因子高相关也不成簇；总条目 2 → 上限 1，pool 1 达标。
        assert plan.is_empty

    def test_capacity_limit_zero_empties_single_entry_library(self) -> None:
        registry = _registry(_entry("solo", rank_ic=0.05))

        plan = plan_prune(registry, {})

        # 总条目 1 → 上限 floor(0.5)=0；规则无例外，唯一因子也会被降级（CLI 会提示）。
        assert plan.capacity_limit == 0
        assert plan.demotions == (PruneDemotion("solo", REASON_CAPACITY, None),)
        assert plan.pool_after == 0

    def test_cluster_prune_then_capacity(self) -> None:
        values = {
            "f_a": _panel([_u(i) for i in range(N_INSTRUMENTS)]),
            "f_b": _panel([2.0 * _u(i) for i in range(N_INSTRUMENTS)]),
            "f_c": _panel([_v(i) for i in range(N_INSTRUMENTS)]),
        }
        registry = _registry(
            _entry("f_a", rank_ic=0.05),
            _entry("f_b", rank_ic=0.04),
            _entry("f_c", rank_ic=0.03),
            _entry("dead", status=STATUS_GRAVEYARD),
        )

        plan = plan_prune(registry, values)

        # 总条目 4 → 上限 2。先聚簇降 f_b（同簇保留 f_a），pool 剩 2 个恰好达标。
        assert plan.demotions == (PruneDemotion("f_b", REASON_CLUSTER, "f_a"),)
        assert plan.pool_after == 2


# ---------------------------------------------------------------------------
# 计划执行、不可变与幂等
# ---------------------------------------------------------------------------


class TestApply:
    def test_apply_returns_new_registry_and_keeps_original(self) -> None:
        registry = _registry(_entry("a", rank_ic=0.05), _entry("b", rank_ic=0.01))
        values = {
            "a": _panel([_u(i) for i in range(N_INSTRUMENTS)]),
            "b": _panel([2.0 * _u(i) for i in range(N_INSTRUMENTS)]),
        }
        plan = plan_prune(registry, values)

        updated = apply_prune(registry, plan)

        assert registry.get("b") is not None
        assert registry.get("b").status == STATUS_POOL  # type: ignore[union-attr]
        assert updated.get("a").status == STATUS_POOL  # type: ignore[union-attr]
        assert updated.get("b").status == STATUS_GRAVEYARD  # type: ignore[union-attr]
        assert updated.version == registry.version

    def test_apply_empty_plan_returns_same_registry(self) -> None:
        registry = _registry(
            _entry("a"),
            _entry("b"),
            _entry("dead_1", status=STATUS_GRAVEYARD),
            _entry("dead_2", status=STATUS_GRAVEYARD),
        )

        assert apply_prune(registry, plan_prune(registry, {})) is registry

    def test_apply_rejects_unknown_factor_id(self) -> None:
        registry = _registry(_entry("a"))

        from quant.factor_lib.prune import PrunePlan

        plan = PrunePlan(
            demotions=(PruneDemotion("ghost", REASON_CAPACITY, None),),
            pool_before=1,
            capacity_limit=0,
        )
        with pytest.raises(FactorLibError, match="未知 factor_id"):
            apply_prune(registry, plan)

    def test_prune_is_idempotent(self) -> None:
        values = _chain_values()
        registry = _registry(
            _entry("a", rank_ic=0.03),
            _entry("b", rank_ic=0.05),
            _entry("c", rank_ic=0.01),
        )

        result = prune(registry, values)

        assert result.changed
        assert plan_prune(result.registry, values).is_empty
        assert prune(result.registry, values).registry == result.registry

    def test_capacity_prune_is_idempotent(self) -> None:
        registry = _registry(
            _entry("f_a", rank_ic=0.04),
            _entry("f_b", rank_ic=0.03),
            _entry("f_c", rank_ic=0.02),
            _entry("f_d", rank_ic=0.01),
        )

        result = prune(registry, {})

        assert plan_prune(result.registry, {}).is_empty


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_cli() -> types.ModuleType:
    """按路径加载 ``scripts/prune_factor_library.py``（scripts 不是包）。"""
    path = Path(__file__).resolve().parents[1] / "scripts" / "prune_factor_library.py"
    spec = importlib.util.spec_from_file_location("prune_factor_library_cli", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def cli(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """加载 CLI 模块，并把行情加载与库内取值换成合成桩，保证不读真实数据。"""
    module = _load_cli()
    one_row = pl.DataFrame({"date": [START], "instrument": [_instrument(0)], "value": [0.0]})
    monkeypatch.setattr(module, "load_bars", lambda *args, **kwargs: one_row)
    monkeypatch.setattr(module, "load_library_values", lambda *args, **kwargs: {})
    return module


def _over_capacity_registry() -> Registry:
    return _registry(
        _entry("f_a", rank_ic=0.04),
        _entry("f_b", rank_ic=0.03),
        _entry("f_c", rank_ic=0.02),
        _entry("f_d", rank_ic=0.01),
    )


def test_cli_dry_run_prints_plan_without_writing(
    cli: types.ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    library = tmp_path / "lib_dry"
    library.mkdir()
    save_registry(_over_capacity_registry(), library)
    before = (library / "registry.json").read_text(encoding="utf-8")

    code = cli.main(
        ["--data-dir", str(tmp_path / "data"), "--factor-library-dir", str(library), "--dry-run"]
    )

    assert code == 0
    out = capsys.readouterr().out
    assert "dry-run" in out
    assert "降级 f_c" in out and "降级 f_d" in out
    assert "超出容量上限" in out
    assert (library / "registry.json").read_text(encoding="utf-8") == before


def test_cli_applies_plan_and_writes_graveyard(
    cli: types.ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    library = tmp_path / "lib_write"
    library.mkdir()
    save_registry(_over_capacity_registry(), library)

    code = cli.main(["--data-dir", str(tmp_path / "data"), "--factor-library-dir", str(library)])

    assert code == 0
    assert "已写回" in capsys.readouterr().out
    updated = load_registry(library)
    assert updated.get("f_c").status == STATUS_GRAVEYARD  # type: ignore[union-attr]
    assert updated.get("f_d").status == STATUS_GRAVEYARD  # type: ignore[union-attr]
    assert updated.get("f_a").status == STATUS_POOL  # type: ignore[union-attr]
    assert plan_prune(updated, {}).is_empty


def test_cli_reports_no_change(
    cli: types.ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    library = tmp_path / "lib_noop"
    library.mkdir()
    save_registry(
        _registry(_entry("f_a", rank_ic=0.05), _entry("dead", status=STATUS_GRAVEYARD)),
        library,
    )

    code = cli.main(["--data-dir", str(tmp_path / "data"), "--factor-library-dir", str(library)])

    assert code == 0
    assert "无需整库" in capsys.readouterr().out


def test_cli_warns_when_capacity_limit_is_zero(
    cli: types.ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    library = tmp_path / "lib_solo"
    library.mkdir()
    save_registry(_registry(_entry("solo", rank_ic=0.05)), library)

    code = cli.main(
        ["--data-dir", str(tmp_path / "data"), "--factor-library-dir", str(library), "--dry-run"]
    )

    assert code == 0
    assert "容量上限为 0" in capsys.readouterr().out


def test_cli_missing_registry_returns_registry_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _load_cli()

    code = cli.main(
        ["--data-dir", str(tmp_path / "data"), "--factor-library-dir", str(tmp_path / "nope")]
    )

    assert code == 2
    assert "registry 不可用" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# 循环导入回归
# ---------------------------------------------------------------------------


def test_eval_factor_imports_from_clean_interpreter() -> None:
    """回归（issue #106）：``quant.eval.factor`` 作为首个 import 不得循环导入失败。

    干净解释器里 ``import quant.eval.factor`` 会经 ``quant.factor_lib`` 触达
    ``prune``；prune 旧版在模块导入期依赖 ``quant.eval.factor.MAX_CORR_REJECT``，
    此时 factor 只初始化了一半，抛 partially initialized ImportError。全套件测试
    因导入顺序掩盖了它，故这里单起子进程复现。
    """
    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(repo_root) if not existing else os.pathsep.join([str(repo_root), existing])
    )

    result = subprocess.run(
        [sys.executable, "-c", "import quant.eval.factor"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
