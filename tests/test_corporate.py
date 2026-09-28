"""``quant.backtest.corporate`` 单元测试：除权日筛选、分红送转到账与明细。"""
from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from quant.backtest.account import Account, Position
from quant.backtest.corporate import (
    CorporateActionDetail,
    actions_on,
    apply_corporate_actions,
)
from quant.data.schema import CORPORATE_ACTIONS

D1 = date(2024, 3, 1)
D2 = date(2024, 3, 2)
A = "600000.SH"
B = "000001.SZ"


# ---------------------------------------------------------------------------
# 构造辅助
# ---------------------------------------------------------------------------


def build_actions(
    rows: list[tuple[date, str, float | None, float | None]],
) -> pl.DataFrame:
    """(date, instrument, cash_per_share, share_per_share) -> CORPORATE_ACTIONS。"""
    return pl.DataFrame(
        {
            "date": [row[0] for row in rows],
            "instrument": [row[1] for row in rows],
            "cash_per_share": [row[2] for row in rows],
            "share_per_share": [row[3] for row in rows],
        },
        schema=CORPORATE_ACTIONS,
    )


def held_account(volume: int = 1000, cost_basis: float = 10_000.0) -> Account:
    account = Account(cash=1_000.0)
    account.positions[A] = Position(
        instrument=A, volume=volume, sellable=volume, cost_basis=cost_basis
    )
    return account


def single_detail(details: list[CorporateActionDetail]) -> CorporateActionDetail:
    assert len(details) == 1
    return details[0]


# ---------------------------------------------------------------------------
# actions_on
# ---------------------------------------------------------------------------


def test_actions_on_filters_by_date_and_sorts() -> None:
    actions = build_actions(
        [
            (D1, B, 0.2, 0.0),
            (D2, A, 0.5, 0.0),
            (D1, A, 0.1, 0.0),
        ]
    )

    today = actions_on(actions, D1)

    assert today["instrument"].to_list() == [B, A]  # 按 instrument 字符串升序
    assert today["cash_per_share"].to_list() == [0.2, 0.1]


def test_actions_on_returns_empty_frame_when_no_action() -> None:
    actions = build_actions([(D1, A, 0.5, 0.0)])

    today = actions_on(actions, D2)

    assert today.height == 0
    assert today.columns == list(CORPORATE_ACTIONS.keys())


def test_actions_on_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="缺少列"):
        actions_on(pl.DataFrame({"date": [D1]}), D1)


# ---------------------------------------------------------------------------
# 现金分红
# ---------------------------------------------------------------------------


def test_cash_dividend_credited_to_cash() -> None:
    account = held_account(volume=1000, cost_basis=10_000.0)
    actions = build_actions([(D1, A, 0.5, 0.0)])

    details = apply_corporate_actions(account, actions, D1, {A: 10.0})

    assert account.cash == pytest.approx(1_500.0)  # 1000 + 0.5 × 1000
    position = account.position(A)
    assert position is not None
    assert position.volume == 1000  # 分红不改股数
    assert position.cost_basis == pytest.approx(10_000.0)

    detail = single_detail(details)
    assert detail.date == D1
    assert detail.instrument == A
    assert detail.cash_per_share == pytest.approx(0.5)
    assert detail.price_before == pytest.approx(10.0)
    assert detail.cash_received == pytest.approx(500.0)
    assert detail.market_value_before == pytest.approx(10_000.0)
    assert (detail.volume_before, detail.volume_after, detail.shares_added) == (
        1000,
        1000,
        0,
    )


# ---------------------------------------------------------------------------
# 送转股
# ---------------------------------------------------------------------------


def test_share_bonus_adds_volume_and_dilutes_cost() -> None:
    account = held_account(volume=1000, cost_basis=10_000.0)
    actions = build_actions([(D1, A, 0.0, 1.0)])  # 10 送 10

    details = apply_corporate_actions(account, actions, D1, {A: 10.0})

    assert account.cash == pytest.approx(1_000.0)
    position = account.position(A)
    assert position is not None
    assert position.volume == 2000
    assert position.sellable == 2000  # 送转股直接可卖
    assert position.cost_basis == pytest.approx(10_000.0)  # 成本总额不变
    assert position.avg_cost == pytest.approx(5.0)  # 每股成本摊薄

    detail = single_detail(details)
    assert detail.share_per_share == pytest.approx(1.0)
    assert (detail.volume_before, detail.volume_after, detail.shares_added) == (
        1000,
        2000,
        1000,
    )
    assert detail.cash_received == 0.0


def test_share_bonus_rounds_to_whole_shares() -> None:
    account = held_account(volume=150, cost_basis=1_500.0)
    actions = build_actions([(D1, A, 0.0, 0.3)])  # 每 10 股送 3 股

    details = apply_corporate_actions(account, actions, D1, {A: 10.0})

    position = account.position(A)
    assert position is not None
    assert position.volume == 195  # round(150 × 0.3) = 45
    assert single_detail(details).shares_added == 45


def test_cash_uses_pre_bonus_volume_when_both_applied() -> None:
    account = held_account(volume=1000, cost_basis=10_000.0)
    actions = build_actions([(D1, A, 0.5, 1.0)])

    details = apply_corporate_actions(account, actions, D1, {A: 10.0})

    # 分红按送转前的 1000 股计提，股数随后翻倍。
    assert account.cash == pytest.approx(1_500.0)
    position = account.position(A)
    assert position is not None
    assert position.volume == 2000
    detail = single_detail(details)
    assert detail.cash_received == pytest.approx(500.0)
    assert detail.shares_added == 1000


# ---------------------------------------------------------------------------
# 边界：未持有 / 空行为 / 缺值
# ---------------------------------------------------------------------------


def test_unheld_instrument_produces_no_detail() -> None:
    account = Account(cash=1_000.0)
    actions = build_actions([(D1, A, 0.5, 1.0)])

    details = apply_corporate_actions(account, actions, D1, {A: 10.0})

    assert details == []
    assert account.cash == 1_000.0
    assert account.positions == {}


def test_no_action_on_day_returns_empty() -> None:
    account = held_account()
    actions = build_actions([(D2, A, 0.5, 0.0)])

    assert apply_corporate_actions(account, actions, D1, {A: 10.0}) == []
    assert account.cash == pytest.approx(1_000.0)


def test_empty_action_table_is_noop() -> None:
    account = held_account()
    actions = build_actions([])

    assert apply_corporate_actions(account, actions, D1, {A: 10.0}) == []
    assert account.cash == pytest.approx(1_000.0)


def test_zero_and_null_amounts_produce_no_detail() -> None:
    account = held_account()
    actions = build_actions([(D1, A, None, None), (D1, A, 0.0, 0.0)])

    assert apply_corporate_actions(account, actions, D1, {A: 10.0}) == []
    assert account.cash == pytest.approx(1_000.0)


def test_missing_price_keeps_detail_but_values_at_zero() -> None:
    account = held_account()
    actions = build_actions([(D1, A, 0.5, 0.0)])

    detail = single_detail(apply_corporate_actions(account, actions, D1, {}))

    assert account.cash == pytest.approx(1_500.0)
    assert detail.price_before == 0.0
    assert detail.market_value_before == 0.0
