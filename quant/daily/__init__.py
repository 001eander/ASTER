"""每日生产跑批（架构 3.5 生产侧）。

- :mod:`quant.daily.virtual_account`：跟单虚拟账户，JSON 持久化 + 实际成交回录；
- :mod:`quant.daily.pipeline`：T 日收盘后的完整跑批（重算因子 → 模型打分 →
  组合优化 → 整手取整 → 调仓单调仓单落盘）。

账户会计逻辑复用 ``quant.backtest.account``，与回测同一套实现。
"""
from quant.daily.pipeline import DailyReport, run_daily
from quant.daily.virtual_account import (
    DEFAULT_ACCOUNT_NAME,
    ActualFill,
    NavRecord,
    VirtualAccount,
)

__all__ = [
    "ActualFill",
    "DailyReport",
    "DEFAULT_ACCOUNT_NAME",
    "NavRecord",
    "VirtualAccount",
    "run_daily",
]
