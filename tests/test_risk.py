"""``quant.portfolio.risk`` 单元测试：Ledoit-Wolf 收缩协方差性质。

全部用合成数据，不触网。核对的性质：

- 收缩强度落在 ``[0, 1]``，``γ̂ = 0`` 与 ``N = 1`` 时退化为 0。
- 输出对称、正定，迹与样本协方差一致。
- 收缩把非对角元按 ``(1 − δ)`` 等比拉向 0（identity target）。
- 独立同分布高维样本强烈收缩、接近对角阵；真实结构明显 + 大样本时几乎不收缩。
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from quant.portfolio.risk import (
    LedoitWolfResult,
    ledoit_wolf,
    ledoit_wolf_covariance,
    ledoit_wolf_shrinkage,
)


def _panel(x: np.ndarray) -> pl.DataFrame:
    return pl.DataFrame(x, schema=[f"{600000 + i:06d}.SH" for i in range(x.shape[1])])


def _iid(n: int, t: int, *, seed: int = 0, scale: float = 0.01) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((t, n)) * scale


def _constant_corr(n: int, t: int, corr: float, *, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    common = rng.standard_normal((t, 1))
    z = np.sqrt(corr) * common + np.sqrt(1 - corr) * rng.standard_normal((t, n))
    return z * 0.01


def _sample_cov(x: np.ndarray) -> np.ndarray:
    centered = x - x.mean(axis=0, keepdims=True)
    return (centered.T @ centered) / x.shape[0]


# ---------------------------------------------------------------------------
# 基本性质
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "t", "corr"),
    [(5, 50, 0.0), (30, 5, 0.6), (80, 12, 0.0), (10, 200, 0.3), (2, 3, 0.9)],
)
def test_shrinkage_within_unit_interval(n: int, t: int, corr: float) -> None:
    for seed in range(5):
        x = _constant_corr(n, t, corr, seed=seed) if corr else _iid(n, t, seed=seed)
        delta = ledoit_wolf_shrinkage(x)
        assert 0.0 <= delta <= 1.0


def test_result_contains_covariance_and_shrinkage() -> None:
    x = _iid(6, 40, seed=1)
    result = ledoit_wolf(x)
    assert isinstance(result, LedoitWolfResult)
    assert result.covariance.shape == (6, 6)
    assert result.shrinkage == pytest.approx(ledoit_wolf_shrinkage(x))


def test_covariance_symmetric_and_positive_definite_high_dim() -> None:
    """T < N 时样本协方差奇异，收缩后仍正定。"""
    x = _iid(80, 12, seed=2)
    covariance = ledoit_wolf_covariance(x)
    assert np.allclose(covariance, covariance.T, atol=1e-12)
    assert np.linalg.eigvalsh(covariance).min() > 0.0
    # 样本协方差此时秩不足
    assert np.linalg.matrix_rank(_sample_cov(x)) < 80


def test_trace_preserved() -> None:
    """Σ̂ = (1−δ)S + δμI 中 δμN = δtr(S)，故迹恒等于样本协方差的迹。"""
    x = _iid(10, 60, seed=3)
    covariance = ledoit_wolf_covariance(x)
    assert np.trace(covariance) == pytest.approx(np.trace(_sample_cov(x)))


def test_offdiagonal_scaled_by_one_minus_delta() -> None:
    """identity target 的非对角元为 0，收缩等于把 S 的非对角元乘以 (1 − δ)。"""
    x = _constant_corr(30, 8, 0.5, seed=4)
    delta = ledoit_wolf_shrinkage(x)
    covariance = ledoit_wolf_covariance(x)
    sample = _sample_cov(x)
    off = ~np.eye(30, dtype=bool)
    assert delta > 0.0
    assert np.allclose(covariance[off], (1.0 - delta) * sample[off], atol=1e-14)


# ---------------------------------------------------------------------------
# 收缩强度的方向性
# ---------------------------------------------------------------------------


def test_iid_high_dim_shrinks_toward_diagonal() -> None:
    """独立同分布、维度高于样本量：样本非对角元多为噪声，应强收缩成近对角阵。"""
    x = _iid(80, 12, seed=5)
    delta = ledoit_wolf_shrinkage(x)
    sample = _sample_cov(x)
    covariance = ledoit_wolf_covariance(x)
    off = ~np.eye(80, dtype=bool)
    off_norm_sample = np.linalg.norm(sample[off])
    off_norm_shrunk = np.linalg.norm(covariance[off])
    assert delta > 0.6
    assert off_norm_shrunk < 0.5 * off_norm_sample


def test_large_sample_structured_low_shrinkage() -> None:
    """真实结构明显（异方差 + 相关）且样本充足时，identity target 误设大，几乎不收缩。"""
    rng = np.random.default_rng(6)
    t, n = 2000, 5
    common = rng.standard_normal((t, 1))
    z = np.sqrt(0.3) * common + np.sqrt(0.7) * rng.standard_normal((t, n))
    z = z * np.sqrt(np.array([1.0, 4.0, 9.0, 16.0, 25.0]))
    delta = ledoit_wolf_shrinkage(z)
    assert delta < 0.05


def test_constant_correlation_has_positive_shrinkage() -> None:
    """恒定相关 + 样本不足时仍会收缩一部分非对角元。"""
    x = _constant_corr(40, 6, 0.5, seed=7)
    delta = ledoit_wolf_shrinkage(x)
    assert delta > 0.1
    sample = _sample_cov(x)
    covariance = ledoit_wolf_covariance(x)
    off = ~np.eye(40, dtype=bool)
    assert np.linalg.norm(covariance[off]) < np.linalg.norm(sample[off])


# ---------------------------------------------------------------------------
# 退化情形
# ---------------------------------------------------------------------------


def test_zero_gamma_no_shrinkage() -> None:
    """样本协方差本身即 μI（列正交且等范数）时 γ̂ = 0，δ = 0。"""
    scale = 0.02
    x = np.zeros((4, 3))
    x[0, 0] = scale
    x[1, 1] = scale
    x[2, 2] = scale
    # assume_centered=True 保留零均值结构：S = (scale²/4) I
    delta = ledoit_wolf_shrinkage(x, assume_centered=True)
    assert delta == 0.0


def test_single_asset_no_shrinkage() -> None:
    """N = 1 时 π̂ = ρ̂，收缩无意义，δ = 0。"""
    x = _iid(1, 20, seed=8)
    assert ledoit_wolf_shrinkage(x) == 0.0


def test_assume_centered_skips_demeaning() -> None:
    """assume_centered=True 时不去均值，结果与手工同一口径一致。"""
    rng = np.random.default_rng(9)
    x = rng.standard_normal((30, 4)) + 5.0  # 非零均值
    centered_twice = x - x.mean(axis=0, keepdims=True)
    manual = ledoit_wolf_covariance(centered_twice, assume_centered=True)
    default = ledoit_wolf_covariance(x)
    assert np.allclose(manual, default)


# ---------------------------------------------------------------------------
# 输入形式与校验
# ---------------------------------------------------------------------------


def test_polars_and_numpy_agree() -> None:
    x = _iid(6, 40, seed=10)
    assert np.allclose(ledoit_wolf_covariance(x), ledoit_wolf_covariance(_panel(x)))


def test_too_few_observations_raises() -> None:
    with pytest.raises(ValueError, match="观测数"):
        ledoit_wolf_covariance(np.zeros((1, 3)))


def test_non_numeric_column_raises() -> None:
    df = pl.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
    with pytest.raises(ValueError, match="非数值列"):
        ledoit_wolf_covariance(df)


def test_non_finite_raises() -> None:
    x = _iid(3, 10)
    x[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN / Inf"):
        ledoit_wolf_covariance(x)


def test_one_dimensional_raises() -> None:
    with pytest.raises(ValueError, match="二维"):
        ledoit_wolf_covariance(np.zeros(10))


def test_unsupported_type_raises() -> None:
    with pytest.raises(TypeError):
        ledoit_wolf_covariance([[1.0, 2.0]])  # type: ignore[arg-type]
