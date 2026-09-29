"""策略配置：把散落的策略参数收敛为一份 JSON（issue #70）。

配置形态（单文件 JSON，不引入 yaml 依赖）::

    {
      "strategy": "index_enhanced",
      "universe": "zz1000",
      "benchmark": "000852",
      "top_k": 50,
      "rebalance_freq": "M",
      "optimize": {
        "stock_band": 0.005,
        "cover_rate_min": 0.5,
        "turnover_max": 0.2
      }
    }

字段
----
- ``strategy``：``stock_selection`` 或 ``index_enhanced``，必填。
- ``universe``：命名池名或自定义池文件路径；缺省全市场。
- ``benchmark``：基准指数代码；``index_enhanced`` 必填（指增锚定基准）。
- ``top_k``：打分进入组合的候选数，缺省 :data:`DEFAULT_TOP_K`。
- ``rebalance_freq``：``D`` / ``W`` / ``M``，缺省 ``D``。
- ``optimize``：优化器参数子对象，键必须是所选策略优化器的字段。

校验集中在 :func:`parse_strategy_config`；非法字段、缺必填、枚举值外即抛
:class:`StrategyConfigError`。优化器构造由 :func:`build_optimizer` 统一完成，
``index_enhanced`` 的 ``frequency`` 取 ``rebalance_freq``，不接受在 ``optimize``
里重复指定。
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from quant.portfolio.enhanced import EnhancedOptimizer
from quant.portfolio.optimizer import PortfolioOptimizer

# ---------------------------------------------------------------------------
# 配置（枚举与缺省值集中在此）
# ---------------------------------------------------------------------------

#: 策略枚举。
STRATEGY_STOCK_SELECTION: str = "stock_selection"
STRATEGY_INDEX_ENHANCED: str = "index_enhanced"
STRATEGIES: tuple[str, ...] = (STRATEGY_STOCK_SELECTION, STRATEGY_INDEX_ENHANCED)

#: 调仓频率枚举。
REBALANCE_FREQS: tuple[str, ...] = ("D", "W", "M")

#: 缺省策略、候选数与调仓频率（缺省即现状：全市场量化选股）。
DEFAULT_STRATEGY: str = STRATEGY_STOCK_SELECTION
DEFAULT_TOP_K: int = 50
DEFAULT_REBALANCE_FREQ: str = "D"

#: 顶层允许字段与必填字段。
_TOP_LEVEL_FIELDS: frozenset[str] = frozenset(
    {"strategy", "universe", "benchmark", "top_k", "rebalance_freq", "optimize"}
)
_REQUIRED_FIELDS: tuple[str, ...] = ("strategy",)

#: ``index_enhanced`` 的 ``optimize`` 中禁止出现的字段（由 ``rebalance_freq`` 统一给出）。
_ENHANCED_RESERVED_FIELDS: frozenset[str] = frozenset({"frequency"})


class StrategyConfigError(ValueError):
    """策略配置不合法（未知字段、缺必填、枚举值外、参数不在优化器字段内）。"""


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StrategyConfig:
    """一份已校验的策略配置。

    Attributes
    ----------
    strategy:
        ``stock_selection`` 或 ``index_enhanced``。
    universe:
        命名池名或自定义池路径；``None`` 表示全市场。
    benchmark:
        基准指数代码；``index_enhanced`` 必填。
    top_k:
        打分进入组合的候选数。
    rebalance_freq:
        ``D`` / ``W`` / ``M``。
    optimize:
        优化器参数字典，键为所选优化器的字段。
    """

    strategy: str = DEFAULT_STRATEGY
    universe: str | None = None
    benchmark: str | None = None
    top_k: int = DEFAULT_TOP_K
    rebalance_freq: str = DEFAULT_REBALANCE_FREQ
    optimize: dict[str, Any] = field(default_factory=dict)

    @property
    def is_index_enhanced(self) -> bool:
        return self.strategy == STRATEGY_INDEX_ENHANCED

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "universe": self.universe,
            "benchmark": self.benchmark,
            "top_k": self.top_k,
            "rebalance_freq": self.rebalance_freq,
            "optimize": dict(self.optimize),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "StrategyConfig":
        return parse_strategy_config(raw)


# ---------------------------------------------------------------------------
# 校验
# ---------------------------------------------------------------------------


def _require_str(value: Any, name: str, *, allow_none: bool) -> str | None:
    if value is None:
        if allow_none:
            return None
        raise StrategyConfigError(f"字段 {name!r} 不能为空")
    if not isinstance(value, str) or not value.strip():
        raise StrategyConfigError(f"字段 {name!r} 必须是非空字符串：{value!r}")
    return value


def _require_int(value: Any, name: str) -> int:
    # bool 是 int 的子类，显式排除，避免 true 被当作 1。
    if isinstance(value, bool) or not isinstance(value, int):
        raise StrategyConfigError(f"字段 {name!r} 必须是整数：{value!r}")
    if value <= 0:
        raise StrategyConfigError(f"字段 {name!r} 必须为正整数：{value!r}")
    return value


def parse_strategy_config(raw: Mapping[str, Any]) -> StrategyConfig:
    """校验原始字典并返回 :class:`StrategyConfig`。

    未知顶层字段、缺必填、枚举值外、``optimize`` 键不在目标优化器字段内均报错。
    """
    if not isinstance(raw, Mapping):
        raise StrategyConfigError(f"策略配置顶层应为对象：{type(raw).__name__}")

    unknown = sorted(set(raw) - _TOP_LEVEL_FIELDS)
    if unknown:
        raise StrategyConfigError(
            f"策略配置含未知字段：{unknown}；允许字段 {sorted(_TOP_LEVEL_FIELDS)}"
        )
    missing = [name for name in _REQUIRED_FIELDS if name not in raw]
    if missing:
        raise StrategyConfigError(f"策略配置缺少必填字段：{missing}")

    strategy = _require_str(raw["strategy"], "strategy", allow_none=False)
    if strategy not in STRATEGIES:
        raise StrategyConfigError(
            f"strategy 必须是 {list(STRATEGIES)} 之一：{strategy!r}"
        )
    assert strategy is not None  # for type checkers

    universe = _require_str(raw.get("universe"), "universe", allow_none=True)
    benchmark = _require_str(raw.get("benchmark"), "benchmark", allow_none=True)

    top_k = _require_int(raw.get("top_k", DEFAULT_TOP_K), "top_k")

    rebalance_freq = _require_str(
        raw.get("rebalance_freq", DEFAULT_REBALANCE_FREQ),
        "rebalance_freq",
        allow_none=False,
    )
    if rebalance_freq not in REBALANCE_FREQS:
        raise StrategyConfigError(
            f"rebalance_freq 必须是 {list(REBALANCE_FREQS)} 之一：{rebalance_freq!r}"
        )
    assert rebalance_freq is not None

    raw_optimize = raw.get("optimize", {})
    if not isinstance(raw_optimize, Mapping):
        raise StrategyConfigError(
            f"optimize 应为对象：{type(raw_optimize).__name__}"
        )
    optimize = dict(raw_optimize)

    config = StrategyConfig(
        strategy=strategy,
        universe=universe,
        benchmark=benchmark,
        top_k=top_k,
        rebalance_freq=rebalance_freq,
        optimize=optimize,
    )

    if config.is_index_enhanced:
        if benchmark is None:
            raise StrategyConfigError("index_enhanced 策略必须给出 benchmark")
        reserved = sorted(set(optimize) & _ENHANCED_RESERVED_FIELDS)
        if reserved:
            raise StrategyConfigError(
                f"index_enhanced 的 optimize 不应含 {reserved}（由 rebalance_freq 统一给出）"
            )
        allowed = set(EnhancedOptimizer.__dataclass_fields__) - _ENHANCED_RESERVED_FIELDS
    else:
        allowed = set(PortfolioOptimizer.__dataclass_fields__)

    unknown_params = sorted(set(optimize) - allowed)
    if unknown_params:
        raise StrategyConfigError(
            f"{strategy} 的 optimize 含未知参数：{unknown_params}；允许 {sorted(allowed)}"
        )

    # 实例化一次触发优化器自身的取值域校验（w_max / cover_rate_min 等）。
    build_optimizer(config)
    return config


def load_strategy_config(path: str | Path) -> StrategyConfig:
    """读取策略配置 JSON；文件缺失或非对象顶层时报错。"""
    path = Path(path)
    if not path.exists():
        raise StrategyConfigError(f"策略配置文件不存在：{path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return parse_strategy_config(raw)


# ---------------------------------------------------------------------------
# 优化器构造
# ---------------------------------------------------------------------------


def build_stock_optimizer(config: StrategyConfig) -> PortfolioOptimizer:
    """按 ``optimize`` 构造 :class:`PortfolioOptimizer`（stock_selection）。"""
    return PortfolioOptimizer(**config.optimize)


def build_enhanced_optimizer(config: StrategyConfig) -> EnhancedOptimizer:
    """按 ``optimize`` 构造 :class:`EnhancedOptimizer`，``frequency`` 取 ``rebalance_freq``。"""
    return EnhancedOptimizer(**config.optimize, frequency=config.rebalance_freq)


def build_optimizer(
    config: StrategyConfig,
) -> PortfolioOptimizer | EnhancedOptimizer:
    """按 ``strategy`` 分派构造优化器。"""
    if config.is_index_enhanced:
        return build_enhanced_optimizer(config)
    return build_stock_optimizer(config)


# ---------------------------------------------------------------------------
# 调仓日判定
# ---------------------------------------------------------------------------


def is_rebalance_day(
    open_days: Sequence[date], day: date, rebalance_freq: str
) -> bool:
    """判断 ``day`` 是否为调仓日（按交易日历的开市日序列）。

    - ``D``：每个开市日都调仓；
    - ``W``：某 ISO 周的首个开市日；
    - ``M``：某自然月的首个开市日。

    ``day`` 不在 ``open_days`` 中时按首个大于它的判断无从谈起，直接返回交易日
    边界前一天的判定（这里保守返回 ``False``，由调用方保证 ``day`` 是开市日）。
    """
    if rebalance_freq == "D":
        return True
    days = sorted({d for d in open_days if d <= day})
    if not days or days[-1] != day:
        return False
    if len(days) == 1:
        return True
    previous = days[-2]
    if rebalance_freq == "W":
        return day.isocalendar()[:2] != previous.isocalendar()[:2]
    if rebalance_freq == "M":
        return (day.year, day.month) != (previous.year, previous.month)
    raise StrategyConfigError(
        f"rebalance_freq 必须是 {list(REBALANCE_FREQS)} 之一：{rebalance_freq!r}"
    )


__all__ = [
    "DEFAULT_REBALANCE_FREQ",
    "DEFAULT_STRATEGY",
    "DEFAULT_TOP_K",
    "REBALANCE_FREQS",
    "STRATEGIES",
    "STRATEGY_INDEX_ENHANCED",
    "STRATEGY_STOCK_SELECTION",
    "StrategyConfig",
    "StrategyConfigError",
    "build_enhanced_optimizer",
    "build_optimizer",
    "build_stock_optimizer",
    "is_rebalance_day",
    "load_strategy_config",
    "parse_strategy_config",
]
