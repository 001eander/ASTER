"""数据体检 CLI：跑批前串联校验，失败以非 0 退出码中止。

用法::

    # 全量体检（首抓后）
    uv run python scripts/validate_data.py --data-dir data --full

    # 日常跑批：只看最近 60 个开市日
    uv run python scripts/validate_data.py --data-dir data --lookback-days 60

退出码：0 表示无 error 级 issue（``report.ok``），1 表示存在 error。
warning 只打印不阻塞。不触网。
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant.data.validate import (  # noqa: E402
    ValidationReport,
    validate,
)

#: 默认数据目录。
DEFAULT_DATA_DIR: str = "data"

logger = logging.getLogger("validate_data")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="对 data/ 缓存做完整性 / 异常值 / 复权体检")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="缓存目录，默认 data/")
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=None,
        help="只体检最近这么多个开市日（日常跑批用）",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="全量体检（与 --lookback-days 互斥；两者都不给时也是全量）",
    )
    return parser


def _print_report(report: ValidationReport) -> None:
    for issue in report.issues:
        logger.log(
            logging.ERROR if issue.severity == "error" else logging.WARNING,
            "[%s] %s（%d）%s",
            issue.severity,
            issue.check,
            issue.count,
            issue.message,
        )
    if report.ok:
        logger.info(
            "体检通过：%d 行 / %d 只证券，%d 条 warning",
            report.checked_rows,
            report.checked_instruments,
            len(report.warnings),
        )
    else:
        logger.error(
            "体检未通过：%d 条 error，%d 条 warning",
            len(report.errors),
            len(report.warnings),
        )


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        pass

    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.full and args.lookback_days is not None:
        parser.error("--full 与 --lookback-days 不能同时使用")

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    report = validate(
        Path(args.data_dir),
        lookback_days=args.lookback_days,
    )
    _print_report(report)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
