"""每日生产跑批 CLI：数据更新后的「因子 → 模型 → 组合 → 调仓单」一体化。

在 ``scripts/daily_update.py`` 之后运行：T 日收盘，本地缓存已推进到最近开市日，
本脚本重算因子、加载已训练模型打分、凸优化出目标持仓、整手取整后 diff 出调仓单。

用法::

    uv run python scripts/run_daily.py
    uv run python scripts/run_daily.py --date 2026-09-25 --account-name default
    uv run python scripts/run_daily.py --data-dir data --model-dir runs/automl/baseline
    uv run python scripts/run_daily.py --dry-run

落盘（``--dry-run`` 时跳过）::

    runs/orders/YYYY-MM-DD.parquet / .csv   # 调仓单
    runs/reports/YYYY-MM-DD.json            # 跑批报告
    runs/account/<name>.json                # 虚拟账户状态

不触网；行情来自 ``data/`` 缓存，模型来自 ``--model-dir``。
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant.automl.trainer import DEFAULT_MODEL_DIR  # noqa: E402
from quant.daily.pipeline import (  # noqa: E402
    DEFAULT_FACTOR_LIBRARY_DIR,
    DEFAULT_ORDERS_DIR,
    DEFAULT_REPORTS_DIR,
    DailyReport,
    run_daily,
)
from quant.daily.strategy import (  # noqa: E402
    StrategyConfig,
    StrategyConfigError,
    load_strategy_config,
)
from quant.daily.virtual_account import (  # noqa: E402
    DEFAULT_ACCOUNT_DIR,
    DEFAULT_ACCOUNT_NAME,
    DEFAULT_INITIAL_CASH,
)

logger = logging.getLogger("run_daily")

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

DEFAULT_DATA_DIR: str = "data"


def _parse_date(text: str) -> date:
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"日期格式应为 YYYY-MM-DD：{text!r}") from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="每日生产跑批：生成调仓单")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="缓存目录，默认 data/")
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help=f"已训练模型目录，默认 {DEFAULT_MODEL_DIR}",
    )
    parser.add_argument(
        "--account-name", default=DEFAULT_ACCOUNT_NAME, help="虚拟账户名，默认 default"
    )
    parser.add_argument("--date", type=_parse_date, default=None, help="信号日，默认最近开市日")
    parser.add_argument(
        "--factor-library",
        default=str(DEFAULT_FACTOR_LIBRARY_DIR),
        help=f"因子库目录，默认 {DEFAULT_FACTOR_LIBRARY_DIR}",
    )
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=DEFAULT_INITIAL_CASH,
        help=f"新建账户的起始现金，默认 {DEFAULT_INITIAL_CASH:.0f}",
    )
    parser.add_argument(
        "--strategy-config",
        default=None,
        help="策略配置 JSON 路径；缺省全市场量化选股（现状）",
    )
    parser.add_argument("--dry-run", action="store_true", help="只计算并打印，不落盘")
    return parser


def _print_summary(report: DailyReport, elapsed: float) -> None:
    print("\n# 每日跑批完成")
    print(f"信号日：{report.date.isoformat()}    账户：{report.account_name}")
    print(
        f"策略：{report.strategy}    池：{report.universe or '全市场'}    "
        f"基准：{report.benchmark or '（未指定）'}    调仓频率：{report.rebalance_freq}"
    )
    print(
        f"净值 {report.nav:.2f} = 现金 {report.cash:.2f} + 持仓市值 "
        f"{report.market_value:.2f}"
    )
    if report.already_ran:
        print("提示：该信号日已跑过，本次跳过（幂等）")
        return
    print(
        f"候选 {report.n_candidates} 只   换手 {report.turnover:.4f}   "
        f"订单 {report.orders.height} 笔   预过滤 {report.filtered.height} 笔"
    )
    print(
        f"敞口：现金 {report.exposure.get('cash_ratio', 0.0):.2%}，"
        f"持仓 {report.exposure.get('position_ratio', 0.0):.2%}，"
        f"持仓只数 {int(report.exposure.get('n_positions', 0.0))}"
    )
    print("打分 top10：")
    for instrument, score in report.top_scores:
        print(f"  {instrument}  {score:+.4f}")
    if report.orders.height:
        print("调仓单：")
        for row in report.orders.iter_rows(named=True):
            print(
                f"  {row['side']:<4} {row['instrument']}  {row['volume']} 股  "
                f"参考价 {row['ref_price']:.3f}  约 {row['est_amount']:.0f} 元"
            )
    else:
        print("调仓单：无")
    if report.filtered.height:
        print("预过滤（T 日停牌 / 涨跌停，仅预估）：")
        for row in report.filtered.iter_rows(named=True):
            print(f"  {row['side']:<4} {row['instrument']}  {row['reason']}")
    for note in report.notes:
        print(f"注意：{note}")
    if report.orders_path is not None:
        print(f"调仓单：{report.orders_path} / {report.orders_csv_path}")
    if report.report_path is not None:
        print(f"报告：{report.report_path}")
    print(f"耗时 {elapsed:.1f}s")


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        pass

    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    strategy_config: StrategyConfig | None = None
    if args.strategy_config:
        try:
            strategy_config = load_strategy_config(args.strategy_config)
        except StrategyConfigError as exc:
            print(f"策略配置不合法：{exc}", file=sys.stderr)
            return 2
        logger.info(
            "策略配置：strategy=%s universe=%s benchmark=%s top_k=%d rebalance_freq=%s",
            strategy_config.strategy,
            strategy_config.universe,
            strategy_config.benchmark,
            strategy_config.top_k,
            strategy_config.rebalance_freq,
        )

    started = time.monotonic()
    report = run_daily(
        Path(args.data_dir),
        Path(args.model_dir),
        account_name=args.account_name,
        ref_date=args.date,
        factor_library_dir=Path(args.factor_library),
        account_dir=DEFAULT_ACCOUNT_DIR,
        orders_dir=DEFAULT_ORDERS_DIR,
        reports_dir=DEFAULT_REPORTS_DIR,
        initial_cash=args.initial_cash,
        strategy_config=strategy_config,
        dry_run=args.dry_run,
    )
    _print_summary(report, time.monotonic() - started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
