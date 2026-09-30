"""因子入库：把过门控的内循环产物复制进因子库并登记 registry。

设计意图
--------
M3 的四层防御（源头分流 / 相关性折扣 / 门控 / 定期整库）铺完后，入库这个动作仍是
人工搬运：内循环产物停在 ``runs/<ts>/eb/solutions/`` 与 ``best/``，没有任何代码把
过门控因子的 ``factor.py`` 落进 ``factor_library/`` 并登记 registry。本模块补上这一段，
按 issue #109 的决策做成「人工确认后执行一条命令」的半自动流程：

1. 读 ``score.json``，``details.gate_passed`` 不为 ``true`` 直接拒绝。门控即评分口径，
   奖励信号一致性是仓库红线（AGENTS.md 量化纪律第 4 条）。
2. 校验 ``factor_id`` 不与 registry 现有条目重复、目标 ``code_path`` 不存在。
3. 复制因子源码到 ``factor_library/<factor_id>.py``。
4. 调 :func:`quant.factor_lib.registry.register_factor` 追加条目，``metrics`` 从
   ``score.json`` 的 ``details.metrics`` 回填（``rank_ic_mean`` → ``rank_ic``），
   ``lineage`` 记 op / 父代 / run_id / generation，``status`` 固定为 pool；写回走
   :func:`quant.factor_lib.registry.save_registry` 的原子写。

写入顺序是先复制源码再保存 registry；保存失败时回滚刚复制的源码，避免留下孤儿文件。
流程结束时只打印摘要，不做 git 提交，``factor_library/`` 的提交由人工确认后完成。

CLI 见 ``scripts/promote_factor.py``。退出码：0 成功，1 未过门控，2 与既有库冲突，
3 输入文件或参数非法，4 registry 缺失或非法。
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from quant.factor_lib.registry import (
    load_registry,
    register_factor,
    save_registry,
)
from quant.factor_lib.schema import (
    KNOWN_LINEAGE_OPS,
    STATUS_POOL,
    Direction,
    FactorEntry,
    FactorLibError,
    Lineage,
)

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

#: 内循环产物中的因子源码文件名。
FACTOR_SOURCE_NAME: str = "factor.py"

#: 评估产物中的评分文件名。
SCORE_FILENAME: str = "score.json"

#: 入库后 ``status`` 固定为 pool。
PROMOTE_STATUS: str = STATUS_POOL

#: registry 指标键 ← ``score.json`` 的 ``details.metrics`` 键。
METRIC_SOURCE_KEYS: dict[str, str] = {
    "rank_ic": "rank_ic_mean",
    "icir": "icir",
    "max_corr": "max_corr",
}

#: 合法 ``factor_id``：字母或下划线开头，后接字母数字下划线。同时也是文件名 stem。
FACTOR_ID_PATTERN: re.Pattern[str] = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

#: CLI 默认因子库目录。
DEFAULT_FACTOR_LIBRARY_DIR: str = "factor_library"

#: 退出码：入库成功。
EXIT_OK: int = 0

#: 退出码：``score.json`` 未过门控。
EXIT_GATE_REJECTED: int = 1

#: 退出码：factor_id 或目标 code_path 与既有库冲突。
EXIT_CONFLICT: int = 2

#: 退出码：输入文件或参数非法。
EXIT_INPUT_ERROR: int = 3

#: 退出码：registry 缺失或非法。
EXIT_REGISTRY_ERROR: int = 4


# ---------------------------------------------------------------------------
# 异常
# ---------------------------------------------------------------------------


class PromoteError(ValueError):
    """入库被拒：输入非法、未过门控或与既有库冲突。"""


class GateNotPassed(PromoteError):
    """``score.json`` 的 ``details.gate_passed`` 不为 true。"""


class PromoteConflict(PromoteError):
    """``factor_id`` 已登记，或目标 ``code_path`` 已被占用。"""


class PromoteInputError(PromoteError):
    """参数、因子源码或 ``score.json`` 结构非法。"""


# ---------------------------------------------------------------------------
# 请求与结果
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PromoteRequest:
    """一次入库请求：内循环产物路径 + 元数据 + 目标因子库目录。"""

    factor_path: Path
    score_path: Path
    factor_id: str
    hypothesis: str
    direction: Direction
    lineage: Lineage
    library_dir: Path


@dataclass(frozen=True, slots=True)
class PromoteResult:
    """入库结果：落位路径与最终写入 registry 的条目。"""

    entry: FactorEntry
    destination: Path
    registry_file: Path

    @property
    def factor_id(self) -> str:
        return self.entry.factor_id


# ---------------------------------------------------------------------------
# score.json 读取与门控
# ---------------------------------------------------------------------------


def _read_json_object(path: Path, *, where: str) -> dict[str, Any]:
    """读取 JSON 对象，文件缺失 / 解析失败 / 顶层非对象都抛错。"""
    if not path.is_file():
        raise PromoteInputError(f"{where} 不存在：{path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PromoteInputError(f"{where} 读取或解析失败：{path}（{exc}）") from exc
    if not isinstance(raw, dict):
        raise PromoteInputError(f"{where} 顶层应为对象：{path}")
    return raw


def _metric_value(metrics: dict[str, Any], key: str, *, where: str) -> float | None:
    """取一个指示值，缺失或 null 记 None，非有限数值抛错。"""
    value = metrics.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PromoteInputError(
            f"{where} 指标 {key!r} 应为数值或 null，实际 {type(value).__name__}"
        )
    number = float(value)
    if not math.isfinite(number):
        raise PromoteInputError(f"{where} 指标 {key!r} 应为有限数值，实际 {value!r}")
    return number


def read_gate_metrics(score_path: str | Path) -> dict[str, float | None]:
    """读 ``score.json``，返回回填 registry 用的指标。

    ``details.gate_passed`` 不为 ``true`` 时抛 :class:`GateNotPassed`；
    ``details`` / ``details.metrics`` 缺失或指标非数值时抛 :class:`PromoteInputError`。
    """
    path = Path(score_path)
    payload = _read_json_object(path, where=SCORE_FILENAME)
    details = payload.get("details")
    if not isinstance(details, dict):
        raise PromoteInputError(f"{SCORE_FILENAME} 缺少 details 对象：{path}")
    if details.get("gate_passed") is not True:
        raise GateNotPassed(
            f"{SCORE_FILENAME} 未过门控（gate_passed={details.get('gate_passed')!r}），拒绝入库：{path}"
        )
    metrics = details.get("metrics")
    if not isinstance(metrics, dict):
        raise PromoteInputError(f"{SCORE_FILENAME} 缺少 details.metrics 对象：{path}")
    return {
        target: _metric_value(metrics, source, where=SCORE_FILENAME)
        for target, source in METRIC_SOURCE_KEYS.items()
    }


# ---------------------------------------------------------------------------
# 入库
# ---------------------------------------------------------------------------


def _validate_factor_id(factor_id: str) -> None:
    """``factor_id`` 兼作文件名 stem，限制字符集以防越出因子库目录。"""
    if not factor_id:
        raise PromoteInputError("factor_id 不能为空")
    if FACTOR_ID_PATTERN.match(factor_id) is None:
        raise PromoteInputError(
            f"factor_id 只允许字母、数字、下划线且不以数字开头：{factor_id!r}"
        )


def _code_path(library_dir: Path, factor_id: str) -> str:
    """registry 中的 ``code_path``，相对仓库根：``factor_library/<id>.py``。"""
    return f"{library_dir.name}/{factor_id}.py"


def promote_factor(request: PromoteRequest) -> PromoteResult:
    """执行一次入库：门控校验 → 冲突校验 → 复制源码 → 登记 registry。

    拒绝（未过门控 / 冲突 / 输入非法）时不修改因子库；registry 保存失败时回滚已复制
    的源码。``factor_id`` 重复与目标文件已存在都抛 :class:`PromoteConflict`。
    """
    library_dir = Path(request.library_dir)
    factor_path = Path(request.factor_path)

    _validate_factor_id(request.factor_id)
    if not request.hypothesis or not request.hypothesis.strip():
        raise PromoteInputError("hypothesis 不能为空")
    if not factor_path.is_file():
        raise PromoteInputError(f"因子源码不存在：{factor_path}")

    metrics = read_gate_metrics(request.score_path)

    try:
        entry = FactorEntry.from_dict(
            {
                "factor_id": request.factor_id,
                "hypothesis": request.hypothesis,
                "code_path": _code_path(library_dir, request.factor_id),
                "metrics": metrics,
                "direction": request.direction.to_dict(),
                "lineage": request.lineage.to_dict(),
                "status": PROMOTE_STATUS,
            },
            where=f"待入库因子 {request.factor_id}",
        )
    except FactorLibError as exc:
        raise PromoteInputError(f"待入库因子元数据非法：{exc}") from exc

    registry = load_registry(library_dir)
    if registry.get(request.factor_id) is not None:
        raise PromoteConflict(f"factor_id 已登记在 registry：{request.factor_id}")

    destination = library_dir / f"{request.factor_id}.py"
    if destination.exists():
        raise PromoteConflict(f"目标源码文件已存在：{destination}")

    updated = register_factor(registry, entry)
    shutil.copyfile(factor_path, destination)
    try:
        registry_file = save_registry(updated, library_dir)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return PromoteResult(
        entry=entry, destination=destination, registry_file=registry_file
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="promote_factor.py",
        description="把过门控的内循环因子登记进 factor_library（人工确认的半自动入库）",
    )
    parser.add_argument("--factor", required=True, help="内循环产物 factor.py 路径")
    parser.add_argument("--score", required=True, help="评估产物 score.json 路径")
    parser.add_argument("--factor-id", required=True, help="因子唯一标识，兼作文件名 stem")
    parser.add_argument("--hypothesis", required=True, help="经济假设一句话")
    parser.add_argument(
        "--signal-source", required=True, help="方向标签：price / volume / price_volume 等"
    )
    parser.add_argument(
        "--time-scale", required=True, help="方向标签：short / medium / long"
    )
    parser.add_argument(
        "--mechanism", required=True, help="方向标签：经济机制，如 momentum / reversal"
    )
    parser.add_argument(
        "--op",
        required=True,
        choices=list(KNOWN_LINEAGE_OPS),
        help="血统操作：seed / mutation / crossover",
    )
    parser.add_argument(
        "--parent",
        action="append",
        default=[],
        dest="parents",
        help="父因子 id，可重复；种子因子留空",
    )
    parser.add_argument("--run-id", default=None, help="产出该因子的内循环 run id")
    parser.add_argument(
        "--generation", type=int, default=0, help="演化代数，非负整数，默认 0"
    )
    parser.add_argument(
        "--factor-library-dir",
        default=DEFAULT_FACTOR_LIBRARY_DIR,
        help=f"因子库目录（含 registry.json），默认 {DEFAULT_FACTOR_LIBRARY_DIR}/",
    )
    return parser


def _require_text(value: str, *, flag: str) -> None:
    if not value or not value.strip():
        raise PromoteInputError(f"--{flag} 不能为空")


def _build_request(args: argparse.Namespace) -> PromoteRequest:
    """把 CLI 参数校验并组装成 :class:`PromoteRequest`。"""
    _validate_factor_id(args.factor_id)
    _require_text(args.hypothesis, flag="hypothesis")
    _require_text(args.signal_source, flag="signal-source")
    _require_text(args.time_scale, flag="time-scale")
    _require_text(args.mechanism, flag="mechanism")
    if args.generation < 0:
        raise PromoteInputError(f"--generation 不应为负：{args.generation}")
    for parent in args.parents:
        _validate_factor_id(parent)
    run_id = args.run_id or None
    return PromoteRequest(
        factor_path=Path(args.factor),
        score_path=Path(args.score),
        factor_id=args.factor_id,
        hypothesis=args.hypothesis,
        direction=Direction(
            signal_source=args.signal_source,
            time_scale=args.time_scale,
            mechanism=args.mechanism,
        ),
        lineage=Lineage(
            op=args.op,
            parents=tuple(args.parents),
            run_id=run_id,
            generation=args.generation,
        ),
        library_dir=Path(args.factor_library_dir),
    )


def _fmt_metric(value: float | None) -> str:
    return "null" if value is None else f"{value:.4f}"


def _print_summary(result: PromoteResult) -> None:
    entry = result.entry
    metrics = entry.metrics
    lineage = entry.lineage
    print(f"已登记因子 {entry.factor_id}")
    print(f"  源码      {result.destination}")
    print(f"  code_path {entry.code_path}")
    print(f"  status    {entry.status}")
    print(
        "  metrics   "
        f"rank_ic={_fmt_metric(metrics.get('rank_ic'))} "
        f"icir={_fmt_metric(metrics.get('icir'))} "
        f"max_corr={_fmt_metric(metrics.get('max_corr'))}"
    )
    print(
        "  lineage   "
        f"op={lineage.op} parents={list(lineage.parents)} "
        f"run_id={lineage.run_id} generation={lineage.generation}"
    )
    print(f"  registry  {result.registry_file}")
    print("确认无误后提交：git add factor_library/ && git commit")


def _reconfigure_stdout() -> None:
    """Windows 控制台下把 stdout 切到 UTF-8，失败则忽略。"""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        pass


def main(argv: list[str] | None = None) -> int:
    """CLI 入口，返回退出码（见模块文档）。"""
    _reconfigure_stdout()
    args = _build_parser().parse_args(argv)

    try:
        request = _build_request(args)
    except PromoteInputError as exc:
        print(f"参数非法：{exc}", file=sys.stderr)
        return EXIT_INPUT_ERROR

    try:
        result = promote_factor(request)
    except GateNotPassed as exc:
        print(f"拒绝入库：{exc}", file=sys.stderr)
        return EXIT_GATE_REJECTED
    except PromoteConflict as exc:
        print(f"拒绝入库：{exc}", file=sys.stderr)
        return EXIT_CONFLICT
    except PromoteInputError as exc:
        print(f"输入非法：{exc}", file=sys.stderr)
        return EXIT_INPUT_ERROR
    except FactorLibError as exc:
        print(f"registry 不可用：{exc}", file=sys.stderr)
        return EXIT_REGISTRY_ERROR

    _print_summary(result)
    return EXIT_OK


__all__ = [
    "DEFAULT_FACTOR_LIBRARY_DIR",
    "EXIT_CONFLICT",
    "EXIT_GATE_REJECTED",
    "EXIT_INPUT_ERROR",
    "EXIT_OK",
    "EXIT_REGISTRY_ERROR",
    "FACTOR_ID_PATTERN",
    "FACTOR_SOURCE_NAME",
    "METRIC_SOURCE_KEYS",
    "PROMOTE_STATUS",
    "SCORE_FILENAME",
    "GateNotPassed",
    "PromoteConflict",
    "PromoteError",
    "PromoteInputError",
    "PromoteRequest",
    "PromoteResult",
    "main",
    "promote_factor",
    "read_gate_metrics",
]


if __name__ == "__main__":
    raise SystemExit(main())
