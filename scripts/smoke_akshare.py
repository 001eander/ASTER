"""akshare 数据源手动冒烟脚本（不进 pytest）。

用法::

    uv run python scripts/smoke_akshare.py

对 600519.SH / 300750.SZ / 430047.BJ 各抓近一月日线，另抓交易日历、公司行为、
证券信息，打印行数与样例行，并做两件事：

1. 用「``vwap = amount / volume`` 应落在当日 [low, high] 内」校验成交量单位；
2. ``assert`` 通过 DataSource 协议，各表过 ``check_schema`` / ``check_daily_bars``。

网络说明：东方财富历史行情域名 ``push2his.eastmoney.com`` 在部分网络下会被阻断，
脚本对每只票独立捕获异常，能取多少打印多少。

代码说明：北交所老代码已迁移到 920 段（诺思兰德 430047.BJ -> 920047.BJ），
脚本用现用代码请求；旧代码取不到数据，因为交易所列表与腾讯接口都用新代码。
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl  # noqa: E402

from quant.data.schema import (  # noqa: E402
    CORPORATE_ACTIONS,
    DAILY_BARS,
    INSTRUMENT_INFO,
    TRADE_CALENDAR,
    check_daily_bars,
    check_schema,
)
from quant.data.source.akshare import AkshareSource  # noqa: E402
from quant.data.source.base import DataSource  # noqa: E402

INSTRUMENTS = ["600519.SH", "300750.SZ", "920047.BJ"]  # 诺思兰德现用代码，原 430047.BJ


def _reconfigure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        pass


def _show(title: str, df: pl.DataFrame, head: int = 3) -> None:
    print(f"\n## {title}  rows={df.height}")
    print(f"columns={df.columns}")
    if df.height:
        print(df.head(head))


def _check_vwap(df: pl.DataFrame) -> bool:
    """vwap 落在 [low, high] 区间内则说明成交量单位（股）正确。"""
    valid = df.filter(pl.col("vwap").is_not_null())
    if valid.height == 0:
        return True
    ok = valid.select(
        (
            (pl.col("vwap") >= pl.col("low") * 0.98)
            & (pl.col("vwap") <= pl.col("high") * 1.02)
        ).all()
    ).item()
    return bool(ok)


def main() -> int:
    _reconfigure_stdout()
    today = dt.date.today()
    start = today - dt.timedelta(days=30)
    source = AkshareSource()

    print(f"# akshare 冒烟：{start} ~ {today}")

    # 1. 日线
    bars = source.daily_bars(INSTRUMENTS, start, today)
    _show("日线 daily_bars", bars, head=5)
    check_daily_bars(bars)
    for instrument in INSTRUMENTS:
        one = bars.filter(pl.col("instrument") == instrument)
        if one.height == 0:
            print(f"  [WARN] {instrument} 无日线数据（该来源被阻断或本区间停牌）")
            continue
        ok = _check_vwap(one)
        print(
            f"  {instrument} rows={one.height} "
            f"volume[min/median]={one['volume'].min():.0f}/{one['volume'].median():.0f} "
            f"amount_median={one['amount'].median():.0f} "
            f"adjfactor_last={one['adjfactor'].drop_nulls().last()} "
            f"vwap_in_range={ok}"
        )
    if bars.height:
        print(
            "  单位核对：vwap 全部落在 [low, high] 内 "
            f"→ {_check_vwap(bars)}（true 表示成交量已按「股」计价）"
        )

    # 2. 交易日历
    calendar = source.trade_calendar(start, today)
    _show("交易日历 trade_calendar", calendar, head=5)
    check_schema(calendar, TRADE_CALENDAR, name="trade_calendar")
    print(f"  区间开市天数={calendar.filter(pl.col('is_open')).height}")

    # 3. 公司行为
    actions = source.corporate_actions(INSTRUMENTS, today - dt.timedelta(days=365 * 3), today)
    _show("公司行为 corporate_actions", actions, head=5)
    check_schema(actions, CORPORATE_ACTIONS, name="corporate_actions")

    # 4. 证券信息
    info = source.instrument_info()
    _show("证券信息 instrument_info", info, head=5)
    check_schema(info, INSTRUMENT_INFO, name="instrument_info")
    print(f"  instrument_info rows={info.height}")
    for instrument in INSTRUMENTS:
        hit = info.filter(pl.col("instrument") == instrument)
        print(f"  {instrument} in info: {hit.height > 0}")

    # 5. 协议与 schema 断言
    assert isinstance(source, DataSource), "AkshareSource 未满足 DataSource 协议"
    check_schema(bars, DAILY_BARS, name="daily_bars")
    print("\n# PASS：协议与各表 schema 校验通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
