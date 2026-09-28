"""``quant.backtest.lot`` 与 ``quant.portfolio.roundlot`` 单元测试。

覆盖各板块整手规则、向下取整、现金守恒（只留现金不透支）、输入校验。
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from quant.backtest.lot import is_valid_volume, lot_step, min_volume, round_lot_down
from quant.portfolio.roundlot import ORDERS_SCHEMA, RoundLotResult, round_weights_to_lots


# ---------------------------------------------------------------------------
# 整手规则
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("volume", "board", "expected"),
    [
        (105.0, "main", 100),
        (199.0, "main", 100),
        (100.0, "main", 100),
        (99.0, "main", 0),
        (150.7, "main", 100),
        (1000.0, "main", 1000),
        (150.0, "cyb", 100),
        (199.0, "kcb", 0),
        (199.9, "kcb", 0),
        (200.0, "kcb", 200),
        (250.0, "kcb", 250),
        (200.9, "kcb", 200),
        (99.0, "bj", 0),
        (100.0, "bj", 100),
        (137.0, "bj", 137),
        (-5.0, "main", 0),
    ],
)
def test_round_lot_down(volume: float, board: str, expected: int) -> None:
    assert round_lot_down(volume, board) == expected  # type: ignore[arg-type]


def test_lot_constants() -> None:
    assert min_volume("main") == 100
    assert min_volume("kcb") == 200
    assert min_volume("cyb") == 100
    assert min_volume("bj") == 100
    assert lot_step("kcb") == 1
    assert lot_step("main") == 100


@pytest.mark.parametrize(
    ("volume", "board", "expected"),
    [
        (0, "main", True),
        (100, "main", True),
        (150, "main", False),
        (200, "kcb", True),
        (201, "kcb", True),
        (150, "kcb", False),
        (100, "bj", True),
        (137, "bj", True),
        (99, "bj", False),
        (50, "main", False),
        (-1, "main", False),
    ],
)
def test_is_valid_volume(volume: int, board: str, expected: bool) -> None:
    assert is_valid_volume(volume, board) == expected  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 权重 → 股数
# ---------------------------------------------------------------------------


def test_orders_table_shapes_and_values() -> None:
    weights = {"600000.SH": 0.05, "688001.SH": 0.03, "300001.SZ": 0.02}
    prices = {"600000.SH": 33.0, "688001.SH": 7.0, "300001.SZ": 21.0}
    nav = 100_000.0
    result = round_weights_to_lots(weights, prices, nav)

    assert isinstance(result, RoundLotResult)
    assert result.orders.schema == ORDERS_SCHEMA
    rows = {row["instrument"]: row for row in result.orders.to_dicts()}

    assert rows["600000.SH"]["board"] == "main"
    assert rows["600000.SH"]["volume"] == 100  # 151.5 → 100
    assert rows["600000.SH"]["est_value"] == pytest.approx(3300.0)
    assert rows["688001.SH"]["board"] == "kcb"
    assert rows["688001.SH"]["volume"] == 428  # 428.57 → 428（≥200，1 股递增）
    assert rows["688001.SH"]["est_value"] == pytest.approx(2996.0)
    assert rows["300001.SZ"]["board"] == "cyb"
    assert rows["300001.SZ"]["volume"] == 0  # 95.2 不足 1 手
    assert result.invested_value == pytest.approx(6296.0)
    assert result.cash_left == pytest.approx(nav - 6296.0)


def test_cash_never_overspent() -> None:
    rng = np.random.default_rng(0)
    instruments = [
        "600000.SH",
        "600519.SH",
        "300750.SZ",
        "688981.SH",
        "000001.SZ",
        "830001.BJ",
    ]
    raw = rng.random(len(instruments))
    weights = {inst: float(w) for inst, w in zip(instruments, raw / raw.sum())}
    prices = {inst: float(p) for inst, p in zip(instruments, rng.uniform(5, 200, len(instruments)))}
    nav = 1_000_000.0

    result = round_weights_to_lots(weights, prices, nav)
    assert result.invested_value <= nav + 1e-6
    assert result.cash_left >= -1e-6
    # 每票股数都满足整手规则
    for row in result.orders.to_dicts():
        assert is_valid_volume(int(row["volume"]), row["board"])


def test_board_inferred_from_code() -> None:
    weights = {"600000.SH": 0.01, "300001.SZ": 0.01, "688001.SH": 0.01, "830001.BJ": 0.01}
    prices = {inst: 10.0 for inst in weights}
    result = round_weights_to_lots(weights, prices, 100_000.0)
    boards = {row["instrument"]: row["board"] for row in result.orders.to_dicts()}
    assert boards == {
        "600000.SH": "main",
        "300001.SZ": "cyb",
        "688001.SH": "kcb",
        "830001.BJ": "bj",
    }


def test_explicit_boards_override() -> None:
    result = round_weights_to_lots(
        {"600000.SH": 0.05},
        {"600000.SH": 10.0},
        100_000.0,
        boards={"600000.SH": "kcb"},
    )
    row = result.orders.to_dicts()[0]
    assert row["board"] == "kcb"
    assert row["volume"] == 500


def test_empty_weights_returns_empty_table() -> None:
    result = round_weights_to_lots({}, {}, 100_000.0)
    assert result.orders.height == 0
    assert result.orders.schema == ORDERS_SCHEMA
    assert result.invested_value == 0.0
    assert result.cash_left == pytest.approx(100_000.0)


# ---------------------------------------------------------------------------
# 输入校验
# ---------------------------------------------------------------------------


def test_missing_price_raises() -> None:
    with pytest.raises(ValueError, match="缺少"):
        round_weights_to_lots({"600000.SH": 0.5}, {}, 100_000.0)


def test_invalid_price_raises() -> None:
    with pytest.raises(ValueError, match="价格非法"):
        round_weights_to_lots({"600000.SH": 0.5}, {"600000.SH": 0.0}, 100_000.0)


def test_weights_over_one_raises() -> None:
    with pytest.raises(ValueError, match="超过 1"):
        round_weights_to_lots(
            {"600000.SH": 0.6, "600001.SH": 0.6},
            {"600000.SH": 10.0, "600001.SH": 10.0},
            100_000.0,
        )


def test_negative_weight_raises() -> None:
    with pytest.raises(ValueError, match="权重非法"):
        round_weights_to_lots({"600000.SH": -0.1}, {"600000.SH": 10.0}, 100_000.0)


@pytest.mark.parametrize("nav", [0.0, -1.0, float("nan")])
def test_invalid_nav_raises(nav: float) -> None:
    with pytest.raises(ValueError, match="nav"):
        round_weights_to_lots({"600000.SH": 0.5}, {"600000.SH": 10.0}, nav)
