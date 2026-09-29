"""多代进化端到端验证（issue #32）：入库因子质量单调 + 库内相关性受控。

验证目标
--------
M3 的四层防御全部上线后（Context 分流 / Proposal 差异化 / 评分共线性折扣 /
入库查重 + 定期整库），用一次合成的多代进化跑通真实组件，确认「入库因子质量
单调提升，库内相关性分布受控」。本文件不触碰真实 ``factor_library/``，全部在
``tmp_path`` 下用合成面板完成，不真实训练、不触网。

仿真设计
--------
合成面板 60 只证券 × 120 个交易日。构造 4 条彼此独立的正交信号通道，标签
``label(t)`` 是 4 条通道的线性组合（载荷逐条递增）加独立噪声，通道 k 的载荷
越大，暴露该通道的因子 RankIC 越高。4 个通道分别编码在 ``vwap`` / ``close`` /
``amount`` / ``high`` 四列上，``low`` 列是独立的正交噪声，供同族变体叠加。

每代提交三到五类候选（第 0 代没有可复刻对象），代表 Context 分流与 Proposal
差异化之后的提案分布：

- ``core``：本代旗舰，暴露第 g 条通道，质量高于上一代旗舰（质量提升的来源）。
- ``var_a`` / ``var_b``：同族变体，通道 g 叠加不同强度的正交噪声，与旗舰高度
  相关但质量更低。
- ``replica``：换皮复刻，源码等价于上一代旗舰（第 1 代起），入库查重应拒绝。
- ``garbage``：垃圾因子，只暴露正交噪声，与标签无关，IC 门控应拒绝。

同一代的所有候选都对「本代开始时」的库状态评估，再统一登记过门控者。这模拟
内循环一次批量提案：单条入库查重看不到同批兄弟提案，因此同族的旗舰与变体会
一起进入 pool，需要靠定期整库（每 2 代一次）聚簇留强兜底；换皮复刻与垃圾因子
则在入库时就被挡住。四层防御的合力由此可观测。

断言口径
--------
1. 质量单调：每代结束时 pool 内质量分（入库时 ``score.json`` 的折扣后 score）
   的最大值随代数单调不降；被拒的换皮复刻 score 明显低于其 quality，折扣生效。
2. 相关性受控：每次整库后 pool 内两两 ``|corr|`` 全部 ``<= MAX_CORR_REJECT``
   （用 ``build_corr_matrix`` 复核）；最终每个同族只剩旗舰一个。
3. 血统链：registry 内代际因子 lineage.parents 指回存在的 factor_id，generation
   沿登记顺序单调不降；graveyard 条目保留完整 lineage 与指标。
4. 奖励信号一致：同代 gate_passed=False 的 score 全部低于 gate_passed=True 的
   最低 score，不存在「分数高过入库线却被拒」。

确定性
------
所有随机量来自固定 seed 的 :class:`random.Random`，两次运行结果完全一致；面板
规模压在秒级，默认（非 slow）标记即可跑完。
"""
from __future__ import annotations

import datetime as dt
import math
import random
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import pytest

from quant.data.schema import DAILY_BARS
from quant.factor_api.spec import FACTOR_INPUT_COLUMNS

# 先导入 factor_lib 包：quant.eval.factor 单独首次导入时会经
# factor_lib/__init__ → prune → factor 触发循环导入；先加载 factor_lib 可绕开。
from quant.factor_lib.correlation import load_library_values
from quant.factor_lib.prune import build_corr_matrix, prune
from quant.factor_lib.registry import (
    load_registry,
    pool_factors,
    register_factor,
    save_registry,
)
from quant.factor_lib.schema import (
    REGISTRY_VERSION,
    STATUS_GRAVEYARD,
    STATUS_POOL,
    FactorEntry,
    Registry,
)
from quant.eval import factor as factor_module
from quant.eval.factor import MAX_CORR_REJECT, FactorEvaluation, evaluate_factor

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

N_INSTRUMENTS: int = 60
N_DAYS: int = 120
START: dt.date = dt.date(2021, 9, 29)
SEED: int = 20260929

#: 4 条正交信号通道，逐一编码到列上供因子暴露。
CHANNEL_COLUMNS: tuple[str, ...] = ("vwap", "close", "amount", "high")
#: 独立正交噪声所在列，供同族变体叠加。
NOISE_COLUMN: str = "low"

#: 标签对各通道的载荷，逐条递增，保证后一代旗舰质量更高。
CHANNEL_LOADINGS: tuple[float, ...] = (0.30, 0.45, 0.60, 0.75)
#: 标签的独立噪声标准差。
LABEL_NOISE_SIGMA: float = 1.0
#: 收益幅度，压到 1% 量级保证价格链恒正。
RETURN_SCALE: float = 0.01
#: 价格基准与信号在列上的幅度。
PRICE_BASE: float = 10.0
SIGNAL_SCALE: float = 0.3
#: 正交噪声列幅度。
NOISE_SCALE: float = 0.3

#: 仿真代数与整库周期：每 PRUNE_EVERY 代结束跑一次整库。
N_GENERATIONS: int = 4
PRUNE_EVERY: int = 2
LIBRARY_DIRNAME: str = "factor_lib"

_CORE_KIND: str = "flagship"
_VARIANT_KIND: str = "variant"
_REPLICA_KIND: str = "replica"
_GARBAGE_KIND: str = "garbage"


# ---------------------------------------------------------------------------
# 合成行情
# ---------------------------------------------------------------------------


def _instrument(index: int) -> str:
    """生成形如 ``600000.SH`` 的证券代码。"""
    return f"{600000 + index:06d}.SH"


def _make_bars(seed: int = SEED) -> pl.DataFrame:
    """合成 60 只 × 120 天的 ``DAILY_BARS``。

    每只证券生成 4 条独立通道 ``s0..s3`` 与两路独立噪声。标签潜在收益
    ``y(t) = Σ c_k s_k(t) + σ·noise(t)``，``open`` 链由 ``y`` 反推，使
    ``label(t) = returns[t+2] = RETURN_SCALE · y(t)``。通道 k 编码进
    ``CHANNEL_COLUMNS[k]`` 对应列，``NOISE_COLUMN`` 列为正交噪声。
    """
    rng = random.Random(seed)
    dates = [START + dt.timedelta(days=step) for step in range(N_DAYS)]
    rows: list[dict[str, object]] = []

    for index in range(N_INSTRUMENTS):
        instrument = _instrument(index)
        channels = [
            [rng.gauss(0.0, 1.0) for _ in range(N_DAYS)] for _ in range(4)
        ]
        label_noise = [rng.gauss(0.0, 1.0) for _ in range(N_DAYS)]
        factor_noise = [rng.gauss(0.0, 1.0) for _ in range(N_DAYS)]
        latent = [
            sum(CHANNEL_LOADINGS[k] * channels[k][day] for k in range(4))
            + LABEL_NOISE_SIGMA * label_noise[day]
            for day in range(N_DAYS)
        ]
        returns = [0.0] * N_DAYS
        for day in range(2, N_DAYS):
            returns[day] = RETURN_SCALE * latent[day - 2]
        opens = [0.0] * N_DAYS
        opens[0] = PRICE_BASE + 0.01 * index
        for day in range(1, N_DAYS):
            opens[day] = opens[day - 1] * (1.0 + returns[day])

        for day in range(N_DAYS):
            open_price = opens[day]
            close = open_price * 1.001
            rows.append(
                {
                    "date": dates[day],
                    "instrument": instrument,
                    "open": open_price,
                    "high": PRICE_BASE + SIGNAL_SCALE * channels[3][day],
                    "low": NOISE_SCALE * factor_noise[day],
                    "close": PRICE_BASE + SIGNAL_SCALE * channels[1][day],
                    "vwap": PRICE_BASE + SIGNAL_SCALE * channels[0][day],
                    "volume": 1_000_000.0,
                    "amount": PRICE_BASE + SIGNAL_SCALE * channels[2][day],
                    "adjfactor": 1.0,
                    "limit_up": None,
                    "limit_down": None,
                }
            )
    return pl.DataFrame(rows, schema=DAILY_BARS).sort(["instrument", "date"])


# ---------------------------------------------------------------------------
# 候选因子源码
# ---------------------------------------------------------------------------


def _flagship_source(column: str) -> str:
    """暴露单条通道的旗舰因子源码。"""
    return f'''"""测试旗舰因子：暴露 {column} 通道。"""
from __future__ import annotations

import polars as pl


def compute(data: pl.DataFrame) -> pl.DataFrame:
    return data.select("date", "instrument", pl.col("{column}").alias("value"))
'''


def _variant_source(column: str, alpha: float) -> str:
    """通道叠加正交噪声的同族变体源码，``alpha`` 控制噪声强度。"""
    return f'''"""测试同族变体：{column} 通道叠加正交噪声。"""
from __future__ import annotations

import polars as pl

ALPHA: float = {alpha}


def compute(data: pl.DataFrame) -> pl.DataFrame:
    return data.select(
        "date",
        "instrument",
        (pl.col("{column}") + ALPHA * pl.col("{NOISE_COLUMN}")).alias("value"),
    )
'''


def _garbage_source() -> str:
    """只暴露正交噪声、与标签无关的垃圾因子源码。"""
    return f'''"""测试垃圾因子：纯正交噪声。"""
from __future__ import annotations

import polars as pl


def compute(data: pl.DataFrame) -> pl.DataFrame:
    return data.select("date", "instrument", pl.col("{NOISE_COLUMN}").alias("value"))
'''


def _write_factor(directory: Path, name: str, source: str) -> Path:
    """把候选因子源码写到库目录下并返回路径。"""
    path = directory / f"{name}.py"
    path.write_text(source, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 候选与仿真记录
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Candidate:
    """一个待评估候选：id、类别、源码、父因子 id。"""

    candidate_id: str
    kind: str
    source: str
    parent_id: str | None


@dataclass(frozen=True, slots=True)
class _CandidateOutcome:
    """候选跑完评估后的观测。"""

    generation: int
    candidate_id: str
    kind: str
    ok: bool
    gate_passed: bool
    score: float | None
    quality: float | None
    max_corr: float | None
    corr_discount: float | None


@dataclass(frozen=True, slots=True)
class _GenerationRecord:
    """一代结束后的快照。"""

    generation: int
    outcomes: tuple[_CandidateOutcome, ...]
    pool_scores_after: dict[str, float]
    pruned: bool
    post_prune_corr: dict[tuple[str, str], float] | None


@dataclass(frozen=True, slots=True)
class _Simulation:
    """整次多代进化的最终状态与逐代记录。"""

    library_dir: Path
    registry: Registry
    generations: tuple[_GenerationRecord, ...]


def _generation_candidates(generation: int) -> tuple[_Candidate, ...]:
    """构造第 ``generation`` 代的候选集合。

    旗舰的父代是上一代旗舰，变体的父代是本代旗舰（同方向），换皮复刻是上一代
    旗舰的等价源码，垃圾因子无父代。第 0 代没有可复刻的对象，不含复刻项。
    """
    column = CHANNEL_COLUMNS[generation]
    core_id = f"g{generation}_core"
    parent_id = f"g{generation - 1}_core" if generation > 0 else None
    candidates = [
        _Candidate(core_id, _CORE_KIND, _flagship_source(column), parent_id),
        _Candidate(
            f"g{generation}_var_a",
            _VARIANT_KIND,
            _variant_source(column, 0.6),
            core_id,
        ),
        _Candidate(
            f"g{generation}_var_b",
            _VARIANT_KIND,
            _variant_source(column, 0.8),
            core_id,
        ),
    ]
    if generation > 0:
        candidates.append(
            _Candidate(
                f"g{generation}_replica",
                _REPLICA_KIND,
                _flagship_source(CHANNEL_COLUMNS[generation - 1]),
                f"g{generation - 1}_core",
            )
        )
    candidates.append(
        _Candidate(
            f"g{generation}_garbage",
            _GARBAGE_KIND,
            _garbage_source(),
            None,
        )
    )
    return tuple(candidates)


def _empty_registry() -> Registry:
    """空 registry，作为进化起点。"""
    return Registry(version=REGISTRY_VERSION, factors=())


def _score_of(evaluation: FactorEvaluation) -> float | None:
    """取 ``score.json`` 口径的折扣后分数 ``quality × corr_discount``。"""
    if not evaluation.ok:
        return None
    return float(factor_module._score_payload(evaluation)["score"])


def _outcome(
    generation: int,
    candidate: _Candidate,
    evaluation: FactorEvaluation,
    score: float | None,
) -> _CandidateOutcome:
    """把评估结果拍成观测记录。"""
    metrics = evaluation.metrics
    quality = metrics.get("quality")
    max_corr = metrics.get("max_corr")
    corr_discount = metrics.get("corr_discount")
    return _CandidateOutcome(
        generation=generation,
        candidate_id=candidate.candidate_id,
        kind=candidate.kind,
        ok=evaluation.ok,
        gate_passed=evaluation.gate_passed,
        score=score,
        quality=quality if isinstance(quality, float) else None,
        max_corr=max_corr if isinstance(max_corr, float) else None,
        corr_discount=corr_discount if isinstance(corr_discount, float) else None,
    )


def _entry(
    candidate: _Candidate,
    evaluation: FactorEvaluation,
    generation: int,
    library_dir: Path,
) -> FactorEntry:
    """由过门控的评估结果构造 registry 条目，metrics 保留折扣拆解。"""
    metrics = evaluation.metrics
    score = _score_of(evaluation)
    assert score is not None
    return FactorEntry.from_dict(
        {
            "factor_id": candidate.candidate_id,
            "hypothesis": f"{candidate.candidate_id} 的测试假设",
            "code_path": f"{library_dir.name}/{candidate.candidate_id}.py",
            "metrics": {
                "rank_ic": metrics["rank_ic_mean"],
                "icir": metrics["icir"],
                "max_corr": metrics["max_corr"],
                "quality": metrics["quality"],
                "corr_discount": metrics["corr_discount"],
                "score": score,
            },
            "direction": {
                "signal_source": "price",
                "time_scale": "short",
                "mechanism": "momentum",
            },
            "lineage": {
                "op": "seed" if candidate.parent_id is None else "mutation",
                "parents": [] if candidate.parent_id is None else [candidate.parent_id],
                "run_id": "sim-m3",
                "generation": generation,
            },
            "status": STATUS_POOL,
        }
    )


def _run_simulation(root: Path) -> _Simulation:
    """跑完整的多代进化仿真并返回逐代记录。"""
    data = _make_bars()
    factor_input = data.select(list(FACTOR_INPUT_COLUMNS))
    library_dir = root / LIBRARY_DIRNAME
    library_dir.mkdir(parents=True, exist_ok=True)
    registry = _empty_registry()
    save_registry(registry, library_dir)

    generations: list[_GenerationRecord] = []
    for generation in range(N_GENERATIONS):
        # 本代全部候选对「本代开始时」的库状态评估，再统一登记过门控者。
        outcomes: list[_CandidateOutcome] = []
        accepted: list[tuple[_Candidate, FactorEvaluation]] = []
        for candidate in _generation_candidates(generation):
            path = _write_factor(library_dir, candidate.candidate_id, candidate.source)
            evaluation = evaluate_factor(path, data, factor_library_dir=library_dir)
            outcomes.append(
                _outcome(generation, candidate, evaluation, _score_of(evaluation))
            )
            if evaluation.ok and evaluation.gate_passed:
                accepted.append((candidate, evaluation))

        for candidate, evaluation in accepted:
            registry = register_factor(
                registry,
                _entry(candidate, evaluation, generation, library_dir),
            )
        save_registry(registry, library_dir)

        pruned = generation % PRUNE_EVERY == PRUNE_EVERY - 1
        post_prune_corr: dict[tuple[str, str], float] | None = None
        if pruned:
            values = load_library_values(library_dir, factor_input)
            result = prune(registry, values)
            registry = result.registry
            save_registry(registry, library_dir)
            pool_ids = {entry.factor_id for entry in pool_factors(registry)}
            post_prune_corr = build_corr_matrix(
                {fid: value for fid, value in values.items() if fid in pool_ids}
            )

        pool = pool_factors(registry)
        generations.append(
            _GenerationRecord(
                generation=generation,
                outcomes=tuple(outcomes),
                pool_scores_after={
                    entry.factor_id: float(entry.metrics["score"]) for entry in pool
                },
                pruned=pruned,
                post_prune_corr=post_prune_corr,
            )
        )

    return _Simulation(
        library_dir=library_dir,
        registry=registry,
        generations=tuple(generations),
    )


@pytest.fixture(scope="module")
def simulation(tmp_path_factory: pytest.TempPathFactory) -> _Simulation:
    """整次仿真只跑一次，四个断言共享同一份结果。"""
    return _run_simulation(tmp_path_factory.mktemp("evolution_m3"))


# ---------------------------------------------------------------------------
# 断言一：质量单调 + 折扣生效
# ---------------------------------------------------------------------------


def test_pool_top_score_is_non_decreasing(simulation: _Simulation) -> None:
    """pool 最优质量分随代数单调不降，且确实在提升。"""
    maxima = [
        max(record.pool_scores_after.values()) for record in simulation.generations
    ]

    assert all(later >= earlier for earlier, later in zip(maxima, maxima[1:]))
    assert maxima[-1] > maxima[0]


def test_flagship_score_increases_across_generations(simulation: _Simulation) -> None:
    """每代旗舰的质量分严格高于上一代，是质量提升的来源。"""
    flagship_scores = [
        record.pool_scores_after[f"g{record.generation}_core"]
        for record in simulation.generations
    ]

    assert all(later > earlier for earlier, later in zip(flagship_scores, flagship_scores[1:]))


def test_replica_score_is_discounted_below_quality(simulation: _Simulation) -> None:
    """换皮复刻被共线性折扣压到远低于其 quality，折扣真实生效。"""
    replicas = [
        outcome
        for record in simulation.generations
        for outcome in record.outcomes
        if outcome.kind == _REPLICA_KIND
    ]

    assert len(replicas) == N_GENERATIONS - 1
    for outcome in replicas:
        assert outcome.ok is True
        assert outcome.gate_passed is False
        assert outcome.quality is not None and outcome.quality > 0.0
        assert outcome.max_corr is not None and outcome.max_corr > MAX_CORR_REJECT
        assert outcome.corr_discount is not None and outcome.corr_discount < 0.05
        assert outcome.score is not None
        assert outcome.score < 0.1 * outcome.quality


# ---------------------------------------------------------------------------
# 断言二：相关性受控
# ---------------------------------------------------------------------------


def test_pool_pairwise_corr_within_threshold_after_each_prune(
    simulation: _Simulation,
) -> None:
    """每次整库后，pool 内两两 |corr| 全部不超过 MAX_CORR_REJECT。"""
    pruned_records = [
        record for record in simulation.generations if record.pruned
    ]

    assert len(pruned_records) == math.ceil(N_GENERATIONS / PRUNE_EVERY)
    for record in pruned_records:
        assert record.post_prune_corr is not None
        for corr in record.post_prune_corr.values():
            assert abs(corr) <= MAX_CORR_REJECT


def test_final_pool_keeps_single_member_per_family(simulation: _Simulation) -> None:
    """最终 pool 里每个同族只剩旗舰一个。"""
    final_pool = pool_factors(simulation.registry)

    assert len(final_pool) == N_GENERATIONS
    by_family: dict[int, list[str]] = {}
    for entry in final_pool:
        by_family.setdefault(entry.lineage.generation, []).append(entry.factor_id)

    assert sorted(by_family) == list(range(N_GENERATIONS))
    for generation, ids in by_family.items():
        assert ids == [f"g{generation}_core"]


# ---------------------------------------------------------------------------
# 断言三：血统链完整
# ---------------------------------------------------------------------------


def test_registry_persisted_matches_in_memory(simulation: _Simulation) -> None:
    """落盘 registry 与内存一致，仿真是可复算的。"""
    assert load_registry(simulation.library_dir) == simulation.registry


def test_lineage_parents_exist_and_generation_non_decreasing(
    simulation: _Simulation,
) -> None:
    """代际因子 lineage.parents 指回存在的 factor_id，generation 单调不降。"""
    registry = simulation.registry
    ids = {entry.factor_id for entry in registry.factors}
    generations = [entry.lineage.generation for entry in registry.factors]

    assert generations == sorted(generations)
    for entry in registry.factors:
        if entry.lineage.op == "seed":
            assert entry.lineage.parents == ()
            assert entry.lineage.generation == 0
            continue
        assert entry.lineage.op == "mutation"
        assert entry.lineage.parents
        for parent in entry.lineage.parents:
            assert parent in ids
            assert parent != entry.factor_id


def test_flagship_lineage_chains_back(simulation: _Simulation) -> None:
    """旗舰的父代串成一条代际链，generation 与代次对齐。"""
    registry = simulation.registry

    assert registry.get("g0_core").lineage.parents == ()  # type: ignore[union-attr]
    for generation in range(1, N_GENERATIONS):
        entry = registry.get(f"g{generation}_core")
        assert entry is not None
        assert entry.lineage.op == "mutation"
        assert entry.lineage.parents == (f"g{generation - 1}_core",)
        assert entry.lineage.generation == generation


def test_graveyard_entries_keep_full_lineage(simulation: _Simulation) -> None:
    """整库降级进 graveyard 的因子保留完整血统与指标。"""
    registry = simulation.registry
    ids = {entry.factor_id for entry in registry.factors}
    graveyard = [
        entry for entry in registry.factors if entry.status == STATUS_GRAVEYARD
    ]

    assert len(graveyard) == N_GENERATIONS * 2
    for entry in graveyard:
        assert entry.lineage.op == "mutation"
        assert entry.lineage.generation >= 0
        assert entry.lineage.parents
        for parent in entry.lineage.parents:
            assert parent in ids
        assert entry.metrics.get("score") is not None
        assert entry.metrics.get("rank_ic") is not None
        assert entry.status == STATUS_GRAVEYARD


# ---------------------------------------------------------------------------
# 断言四：奖励信号一致
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("generation", range(N_GENERATIONS))
def test_rejected_scores_stay_below_accepted_floor(
    simulation: _Simulation, generation: int
) -> None:
    """同代被拒候选的 score 全部低于过门控者的最低 score。"""
    record = simulation.generations[generation]
    accepted = [
        outcome.score
        for outcome in record.outcomes
        if outcome.gate_passed and outcome.score is not None
    ]
    rejected = [
        outcome.score
        for outcome in record.outcomes
        if not outcome.gate_passed and outcome.score is not None
    ]

    assert accepted
    assert rejected
    assert max(rejected) < min(accepted)


def test_no_high_score_rejected_candidate(simulation: _Simulation) -> None:
    """不存在「分数高过同代入库线却被拒」的隐藏规则。"""
    for record in simulation.generations:
        accepted = [
            outcome.score
            for outcome in record.outcomes
            if outcome.gate_passed and outcome.score is not None
        ]
        assert accepted
        entry_line = min(accepted)
        violators = [
            outcome.candidate_id
            for outcome in record.outcomes
            if not outcome.gate_passed
            and outcome.score is not None
            and outcome.score >= entry_line
        ]
        assert violators == []
