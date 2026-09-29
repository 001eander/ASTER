"""对拍样本与原型参考输出生成脚本（开发用，不进测试路径）。

从 ``reference-portfolio-optimization.py``（pandas 原型）抽出约束构建、可行性预检、
放松阶梯与求解循环，对固定合成样本跑出参考权重与放松轮数，落盘
``tests/data/enhanced_opt_fixture/expected.json``。测试在 ``tests/test_enhanced_optimizer.py``
里只读这份 JSON，不导入 pandas / 原型。

原型硬编码 ECOS（本机未安装），脚本改用与本仓库一致的 CLARABEL 求解，
可行性预检同样用 CLARABEL 重实现，其余约束数学逐行取自原型。

用法：
    uv run python tests/data/generate_enhanced_opt_fixture.py
"""
from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FIXTURE_DIR = HERE / "enhanced_opt_fixture"
INPUTS_PATH = FIXTURE_DIR / "inputs.json"
EXPECTED_PATH = FIXTURE_DIR / "expected.json"
PROTOTYPE_PATH = Path(r"C:\Users\陶唐\.opencode\plan\reference-portfolio-optimization.py")

FACTORS = ["beta", "momentum", "nlsize", "reverse", "sigma", "turnover"]
DATE = "20250102"


def _load_prototype():
    spec = importlib.util.spec_from_file_location("reference_portfolio_optimization", PROTOTYPE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def build_inputs() -> dict:
    rng = np.random.default_rng(20250102)
    n, n_members = 40, 30
    instruments = [f"{600000 + i:06d}.SH" for i in range(n)]

    bench_raw = np.zeros(n)
    bench_raw[:n_members] = rng.dirichlet(np.ones(n_members))
    members = bench_raw > 0

    alpha = rng.standard_normal(n)
    industry = [f"I{rng.integers(0, 5)}" for _ in range(n)]
    float_mv = np.exp(rng.normal(24.0, 1.2, size=n))
    style = {f: rng.standard_normal(n) for f in FACTORS}
    # 上期权重：0.7 基准 + 0.3 随机，投影回可行带内约需 0.17 换手，
    # 用于在"relaxed"样本里触发换手放松阶梯（风格/行业/市值轮次无效果）。
    bench_norm = bench_raw / bench_raw.sum()
    w_prev = 0.7 * bench_norm + 0.3 * rng.dirichlet(np.ones(n))
    w_prev = w_prev / w_prev.sum()
    # 远离可行域的上期权重（随机），用于不可行样本。
    w_prev_far = rng.dirichlet(np.ones(n))

    return {
        "instruments": instruments,
        "members": members.tolist(),
        "bench_weights": bench_raw.tolist(),
        "alpha": alpha.tolist(),
        "industry": industry,
        "float_mv": float_mv.tolist(),
        "style": {f: v.tolist() for f, v in style.items()},
        "w_prev": w_prev.tolist(),
        "w_prev_far": w_prev_far.tolist(),
        "covariance": _covariance(rng, n).tolist(),
    }


def _covariance(rng: np.random.Generator, n: int) -> np.ndarray:
    b = rng.standard_normal((n, n))
    return (b @ b.T) / n * 1.0e-4


# 每组样本的约束阈值；键名对应模块配置。
DEFAULT_PARAMS = {
    "initial_industry_exposure": 0.005,
    "initial_market_value_exposure": 0.005,
    "initial_market_value_exposure_type": "std",
    "initial_cover_rate_lower_bound": 0.5,
    "initial_turn_over_rate_upper_bound": 0.2,
    "initial_style_factors_exposure": 0.005,
    "industry_exposure_relaxation_step": 0.001,
    "market_value_exposure_relaxation_step": 0.001,
    "turn_over_rate_upper_bound_relaxation_step": 0.01,
    "style_factors_exposure_relaxation_step": 0.001,
}

SAMPLES = [
    {"name": "optimal", "params": {"initial_turn_over_rate_upper_bound": 2.0}},
    {"name": "relaxed", "params": {"initial_turn_over_rate_upper_bound": 0.1}},
    {"name": "tight_cover",
     "params": {"initial_cover_rate_lower_bound": 0.95,
                "initial_turn_over_rate_upper_bound": 2.0}},
    {"name": "infeasible",
     "params": {"initial_turn_over_rate_upper_bound": 0.0},
     "prev": "w_prev_far"},
]


def run_prototype(module, inputs: dict, params: dict, solver: str, prev_key: str = "w_prev") -> dict:
    n = len(inputs["instruments"])
    instruments = inputs["instruments"]
    members = np.array(inputs["members"], dtype=bool)
    bench_raw = np.array(inputs["bench_weights"], dtype=float)
    industry = inputs["industry"]
    float_mv = np.array(inputs["float_mv"], dtype=float)
    style = {f: np.array(v, dtype=float) for f, v in inputs["style"].items()}
    alpha = np.array(inputs["alpha"], dtype=float)
    w_prev = np.array(inputs[prev_key], dtype=float)
    cov = np.array(inputs["covariance"], dtype=float)

    s_temp = pd.DataFrame(
        {
            "TICKER": instruments,
            "DATE": DATE,
            "pred_return": alpha,
            "CITIC_CODE": industry,
            "float_mv": float_mv,
            **{f: style[f] for f in FACTORS},
        }
    ).sort_values("TICKER").reset_index(drop=True)

    member_idx = np.where(members)[0]
    index_weight_temp = pd.DataFrame(
        {
            "TICKER": [instruments[i] for i in member_idx],
            "DATE": DATE,
            "weight": [bench_raw[i] for i in member_idx],
            "CITIC_CODE": [industry[i] for i in member_idx],
            "float_mv": [float_mv[i] for i in member_idx],
            **{f: [style[f][i] for i in member_idx] for f in FACTORS},
        }
    ).sort_values("TICKER").reset_index(drop=True)

    s_temp = s_temp.merge(index_weight_temp[["TICKER", "DATE", "weight"]], on=["TICKER", "DATE"], how="left")
    s_temp["weight"] = s_temp["weight"].fillna(0.0)
    if s_temp["weight"].sum() == 0:
        s_temp["weight"] = 0.0
    else:
        s_temp["weight"] = s_temp["weight"] / s_temp["weight"].sum()
    s_temp["stock_weight_upper_bound"] = np.where(
        s_temp["weight"] == 0.0, 0.005, np.minimum(s_temp["weight"] + 0.005, 1.0)
    )
    s_temp["stock_weight_lower_bound"] = np.where(
        s_temp["weight"] == 0.0, 0.0, np.maximum(s_temp["weight"] - 0.005, 0.0)
    )

    with tempfile.TemporaryDirectory() as tmp:
        full_params = {**DEFAULT_PARAMS, **params, "root_directory": tmp,
                       "style_factors_list": FACTORS, "objective_function_information": "朴素得分最优",
                       "initial_industry_exposure_type": "ratio",
                       "initial_style_factors_exposure_type": "std",
                       "start_date": DATE, "end_date": DATE}
        opt = module.portfolio_optimization(**full_params)
        opt.update_mode = 0
        opt.stock_score = s_temp.copy()
        opt.index_weight = index_weight_temp.copy()
    opt.check_feasibility = lambda cons: _check_feasible(cons, solver)

    objective_function, w, v, lamda = opt.generate_objective_function(DATE, None)
    thresholds = {
        "industry_exposure": DEFAULT_PARAMS["initial_industry_exposure"],
        "market_value_exposure": DEFAULT_PARAMS["initial_market_value_exposure"],
        "cover_rate_lower_bound": DEFAULT_PARAMS["initial_cover_rate_lower_bound"],
        "turn_over_rate_upper_bound": DEFAULT_PARAMS["initial_turn_over_rate_upper_bound"],
        "style_factors_exposure": DEFAULT_PARAMS["initial_style_factors_exposure"],
    }
    thresholds.update(
        {
            "industry_exposure": params.get("initial_industry_exposure", 0.005),
            "market_value_exposure": params.get("initial_market_value_exposure", 0.005),
            "cover_rate_lower_bound": params.get("initial_cover_rate_lower_bound", 0.5),
            "turn_over_rate_upper_bound": params.get("initial_turn_over_rate_upper_bound", 0.2),
            "style_factors_exposure": params.get("initial_style_factors_exposure", 0.005),
        }
    )

    constraints = opt.add_constraints(
        date=DATE, s_temp=s_temp.copy(), index_weight_temp=index_weight_temp.copy(),
        w=w, wL1=w_prev, v=v, lamda=lamda, **thresholds,
    )

    w.value = s_temp["weight"].to_numpy()
    if_feasible = False
    relaxed = dict(thresholds)
    status = 0
    relax_count = 0
    for relax_count in range(60):
        if not if_feasible:
            if_feasible = opt.check_feasibility(constraints)
        if if_feasible:
            problem = cp.Problem(objective_function, constraints)
            try:
                problem.solve(solver=solver)
                if w.value is not None:
                    value = np.asarray(w.value).flatten()
                    if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] and not np.isnan(value).any():
                        status = 1
                        break
            except cp.SolverError:
                pass
        relaxed = opt.relax_constraints_threshold(relaxed, relax_count)
        constraints = opt.add_constraints(
            date=DATE, s_temp=s_temp.copy(), index_weight_temp=index_weight_temp.copy(),
            w=w, wL1=w_prev, v=v, lamda=lamda, **relaxed,
        )

    if status == 1:
        weights = np.maximum(np.asarray(w.value).flatten(), 0.0)
        weights[weights < 1e-4] = 0.0
        total = weights.sum()
        weights = weights / total if total > 0 else np.zeros(n)
    else:
        weights = np.zeros(n)
    return {
        "status": status,
        "relax_rounds": int(relax_count),
        "weights": weights.tolist(),
        "thresholds": {k: float(val) for k, val in relaxed.items()},
    }


def _check_feasible(constraints, solver: str) -> bool:
    try:
        problem = cp.Problem(cp.Minimize(0), constraints)
        problem.solve(solver=solver)
        return problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]
    except Exception:
        return False


def main() -> None:
    module = _load_prototype()
    inputs = build_inputs()
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    INPUTS_PATH.write_text(json.dumps(inputs, indent=1), encoding="utf-8")

    expected: dict[str, dict] = {}
    for sample in SAMPLES:
        result = run_prototype(
            module, inputs, sample["params"], cp.CLARABEL,
            prev_key=sample.get("prev", "w_prev"),
        )
        expected[sample["name"]] = {
            "params": sample["params"],
            "prev": sample.get("prev", "w_prev"),
            **result,
        }
        print(sample["name"], "status", result["status"], "relax", result["relax_rounds"])
    EXPECTED_PATH.write_text(json.dumps(expected, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
