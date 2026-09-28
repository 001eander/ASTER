"""虚拟账户：跟单持仓的 JSON 持久化 + 实际成交回录（架构 3.5 生产侧）。

定位
----
回测与生产共用同一套会计逻辑（:class:`quant.backtest.account.Account`），本模块只在其
外面包一层「生产工作流」需要的状态：

- **持久化**：现金、持仓、历史净值序列、累计统计、已完成跑批日期，落成一份 JSON，
  崩溃后 :meth:`VirtualAccount.load` 即可续跑。
- **实际成交回录**：人工按调仓单在 T+1 下单后，把真实成交（含实际价格与费用）录进
  账户；持仓以真实成交为准，而非模拟撮合结果。
- **跑批幂等标记**：``last_pipeline_date`` 记录已完成跑批的信号日，重复跑同一天不再
  重算、不重复改账户（见 :mod:`quant.daily.pipeline`）。

文件布局::

    runs/account/<name>.json

写入用「临时文件 + 替换」的原子方式，避免中断留下半截 JSON。
"""
from __future__ import annotations

import datetime as dt
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from quant.backtest.account import Account
from quant.backtest.fee import Fee, Side

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

#: 账户目录（``runs/`` 不入 git）。
DEFAULT_ACCOUNT_DIR: Path = Path("runs/account")

#: 默认账户名。
DEFAULT_ACCOUNT_NAME: str = "default"

#: 账户文件名后缀。
ACCOUNT_SUFFIX: str = ".json"

#: 新建账户的默认起始现金（元）。
DEFAULT_INITIAL_CASH: float = 1_000_000.0

_VALID_SIDES: frozenset[str] = frozenset({"buy", "sell"})

#: 历史净值表 schema。
NAV_SCHEMA: pl.Schema = pl.Schema(
    {
        "date": pl.Date,
        "cash": pl.Float64,
        "market_value": pl.Float64,
        "nav": pl.Float64,
    }
)

#: 持仓摘要表 schema。
HOLDINGS_SCHEMA: pl.Schema = pl.Schema(
    {
        "instrument": pl.String,
        "volume": pl.Int64,
        "sellable": pl.Int64,
        "avg_cost": pl.Float64,
        "price": pl.Float64,
        "market_value": pl.Float64,
        "weight": pl.Float64,
    }
)


# ---------------------------------------------------------------------------
# 记录类型
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NavRecord:
    """某一日的账户净值快照。"""

    date: date
    cash: float
    market_value: float
    nav: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date.isoformat(),
            "cash": self.cash,
            "market_value": self.market_value,
            "nav": self.nav,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NavRecord":
        return cls(
            date=date.fromisoformat(str(data["date"])),
            cash=float(data["cash"]),
            market_value=float(data["market_value"]),
            nav=float(data["nav"]),
        )


@dataclass(frozen=True)
class ActualFill:
    """一笔人工跟单后的真实成交（元 / 股）。

    ``fee`` 为这笔成交实际付出的现金费用（佣金 + 印花税等）。``date`` 为成交日，
    回录时留空表示不记录日期。
    """

    instrument: str
    side: Side
    volume: int
    price: float
    fee: float = 0.0
    date: date | None = None

    def __post_init__(self) -> None:
        if self.side not in _VALID_SIDES:
            raise ValueError(f"未知交易方向: {self.side!r}，应为 'buy' 或 'sell'")
        if self.volume <= 0:
            raise ValueError(f"成交股数必须为正: {self.volume}")
        if self.price <= 0:
            raise ValueError(f"成交价格必须为正: {self.price}")
        if self.fee < 0:
            raise ValueError(f"费用不能为负: {self.fee}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date.isoformat() if self.date is not None else None,
            "instrument": self.instrument,
            "side": self.side,
            "volume": self.volume,
            "price": self.price,
            "fee": self.fee,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ActualFill":
        raw_date = data.get("date")
        return cls(
            instrument=str(data["instrument"]),
            side=str(data["side"]),  # type: ignore[arg-type]
            volume=int(data["volume"]),
            price=float(data["price"]),
            fee=float(data.get("fee", 0.0)),
            date=date.fromisoformat(str(raw_date)) if raw_date else None,
        )


# ---------------------------------------------------------------------------
# 虚拟账户
# ---------------------------------------------------------------------------


@dataclass
class VirtualAccount:
    """跟单虚拟账户：账户状态 + 历史净值 + 累计统计 + 回录成交。"""

    account: Account = field(default_factory=Account)
    name: str = DEFAULT_ACCOUNT_NAME
    initial_cash: float = 0.0
    nav_history: list[NavRecord] = field(default_factory=list)
    fills: list[ActualFill] = field(default_factory=list)
    last_pipeline_date: date | None = None
    created_at: str = ""
    updated_at: str = ""

    # -- 路径 ---------------------------------------------------------------

    @staticmethod
    def path_for(account_dir: Path, name: str = DEFAULT_ACCOUNT_NAME) -> Path:
        """返回账户 JSON 路径 ``<account_dir>/<name>.json``。"""
        return Path(account_dir) / f"{name}{ACCOUNT_SUFFIX}"

    # -- 查询 ---------------------------------------------------------------

    @property
    def cash(self) -> float:
        return self.account.cash

    @property
    def positions(self) -> dict[str, Any]:
        return self.account.positions

    def nav(self, prices: Mapping[str, float]) -> float:
        """按给定价格估值：现金 + 持仓市值。"""
        return self.account.nav(prices)

    def market_value(self, prices: Mapping[str, float]) -> float:
        return self.account.market_value(prices)

    def weights(self, prices: Mapping[str, float]) -> dict[str, float]:
        """当前持仓权重 ``{instrument: weight}``；总市值为 0 时返回空字典。"""
        market_value = self.account.market_value(prices)
        nav = self.account.cash + market_value
        if nav <= 0.0 or market_value <= 0.0:
            return {}
        return {
            instrument: position.volume * prices[instrument] / nav
            for instrument, position in self.account.positions.items()
            if position.volume > 0
        }

    def holdings(self, prices: Mapping[str, float]) -> pl.DataFrame:
        """持仓摘要（列见 :data:`HOLDINGS_SCHEMA`），按权重降序。"""
        market_value = self.account.market_value(prices)
        nav = self.account.cash + market_value
        rows = [
            {
                "instrument": instrument,
                "volume": position.volume,
                "sellable": position.sellable,
                "avg_cost": position.avg_cost,
                "price": float(prices[instrument]),
                "market_value": position.volume * float(prices[instrument]),
                "weight": (
                    position.volume * float(prices[instrument]) / nav
                    if nav > 0
                    else 0.0
                ),
            }
            for instrument, position in self.account.positions.items()
            if position.volume > 0
        ]
        if not rows:
            return pl.DataFrame(schema=HOLDINGS_SCHEMA)
        return pl.DataFrame(rows, schema=HOLDINGS_SCHEMA).sort(
            "weight", descending=True
        )

    # -- 净值与成交 ---------------------------------------------------------

    def record_nav(self, day: date, prices: Mapping[str, float]) -> NavRecord:
        """按 ``prices`` 在 ``day`` 记一条净值快照；同日已有记录则原样返回。"""
        existing = self.nav_history[-1] if self.nav_history else None
        if existing is not None and existing.date == day:
            return existing
        record = NavRecord(
            date=day,
            cash=self.account.cash,
            market_value=self.account.market_value(prices),
            nav=self.account.nav(prices),
        )
        self.nav_history.append(record)
        return record

    def apply_actual_fills(self, fills: Sequence[ActualFill]) -> None:
        """回录真实成交，直接调整现金与持仓（真实账户以回录为准）。

        先处理卖单再处理买单，让卖出回款可用于当日买入；同一批内保持原有相对顺序。
        现金不足、可卖不足沿用 :class:`quant.backtest.account` 的异常口径。
        """
        ordered = sorted(fills, key=lambda fill: 0 if fill.side == "sell" else 1)
        for fill in ordered:
            fee = Fee(commission=fill.fee)
            if fill.side == "buy":
                self.account.buy(fill.instrument, fill.volume, fill.price, fee)
            else:
                self.account.sell(fill.instrument, fill.volume, fill.price, fee)
            self.fills.append(fill)

    def settle_new_day(self) -> None:
        """昨日买入解冻（次日开盘前调用）。"""
        self.account.settle_new_day()

    # -- 统计 ---------------------------------------------------------------

    @property
    def latest_nav(self) -> float:
        """最近一条净值；无记录时为初始现金。"""
        return self.nav_history[-1].nav if self.nav_history else self.initial_cash

    def stats(self) -> dict[str, float | int | str | None]:
        """累计统计：总收益、最大回撤、记录天数与最新净值。"""
        navs = [record.nav for record in self.nav_history]
        total_return = (
            navs[-1] / self.initial_cash - 1.0
            if navs and self.initial_cash > 0
            else 0.0
        )
        peak = float("-inf")
        max_drawdown = 0.0
        for value in navs:
            peak = max(peak, value)
            if peak > 0:
                max_drawdown = min(max_drawdown, value / peak - 1.0)
        return {
            "initial_cash": self.initial_cash,
            "latest_nav": self.latest_nav,
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "n_days": len(navs),
            "last_pipeline_date": (
                self.last_pipeline_date.isoformat()
                if self.last_pipeline_date is not None
                else None
            ),
        }

    # -- 持久化 -------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "initial_cash": self.initial_cash,
            "account": self.account.to_dict(),
            "nav_history": [record.to_dict() for record in self.nav_history],
            "fills": [fill.to_dict() for fill in self.fills],
            "last_pipeline_date": (
                self.last_pipeline_date.isoformat()
                if self.last_pipeline_date is not None
                else None
            ),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VirtualAccount":
        raw_last = data.get("last_pipeline_date")
        return cls(
            account=Account.from_dict(data["account"]),
            name=str(data.get("name", DEFAULT_ACCOUNT_NAME)),
            initial_cash=float(data.get("initial_cash", 0.0)),
            nav_history=[
                NavRecord.from_dict(item) for item in data.get("nav_history", [])
            ],
            fills=[ActualFill.from_dict(item) for item in data.get("fills", [])],
            last_pipeline_date=(
                date.fromisoformat(str(raw_last)) if raw_last else None
            ),
            created_at=str(data.get("created_at", "")),
            updated_at=str(data.get("updated_at", "")),
        )

    def save(self, path: Path) -> Path:
        """原子写入 JSON，返回写入路径。"""
        path = Path(path)
        if not self.created_at:
            self.created_at = _now()
        self.updated_at = _now()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, path)
        return path

    @classmethod
    def load(cls, path: Path) -> "VirtualAccount":
        """从 :meth:`save` 写入的 JSON 恢复账户。"""
        path = Path(path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"账户文件格式非法：{path}")
        return cls.from_dict(raw)

    @classmethod
    def create(
        cls,
        *,
        name: str = DEFAULT_ACCOUNT_NAME,
        initial_cash: float = DEFAULT_INITIAL_CASH,
    ) -> "VirtualAccount":
        """新建空账户（仅现金）。"""
        if initial_cash < 0:
            raise ValueError(f"初始资金不能为负: {initial_cash}")
        return cls(
            account=Account(cash=initial_cash),
            name=name,
            initial_cash=initial_cash,
            created_at=_now(),
        )

    @classmethod
    def load_or_create(
        cls,
        path: Path,
        *,
        name: str = DEFAULT_ACCOUNT_NAME,
        initial_cash: float = DEFAULT_INITIAL_CASH,
    ) -> "VirtualAccount":
        """文件存在则加载，否则新建（不落盘，由调用方决定何时 save）。"""
        path = Path(path)
        if path.exists():
            return cls.load(path)
        return cls.create(name=name, initial_cash=initial_cash)


def _now() -> str:
    return dt.datetime.now().replace(microsecond=0).isoformat()


__all__ = [
    "ACCOUNT_SUFFIX",
    "ActualFill",
    "DEFAULT_ACCOUNT_DIR",
    "DEFAULT_ACCOUNT_NAME",
    "DEFAULT_INITIAL_CASH",
    "HOLDINGS_SCHEMA",
    "NAV_SCHEMA",
    "NavRecord",
    "VirtualAccount",
]
