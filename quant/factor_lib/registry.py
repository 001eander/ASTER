"""``registry.json`` 读写：加载、原子保存、pool 过滤与注册。

设计意图
--------
把「因子集」从目录扫描改为显式注册表后，读写入口只有这一个模块：

- :func:`load_registry` 严格解析，注册表缺失或格式非法立即抛错，不静默回退；
  需要「存在才加载」的调用方先用 :func:`registry_path` 判断。
- :func:`save_registry` 用临时文件 + :func:`os.replace` 原子替换，避免中途崩溃
  留下半截 JSON。
- :func:`pool_factors` 提供建模侧唯一入口：只有 ``status == "pool"`` 的因子
  参与建模，graveyard 因子留在库里但不进入数据集。
- :func:`register_factor` 是纯函数：返回追加条目后的新 :class:`Registry`，
  ``factor_id`` 与已有条目冲突即抛 :class:`FactorLibError`；落盘由调用方决定。
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from quant.factor_lib.schema import (
    STATUS_POOL,
    FactorEntry,
    FactorLibError,
    Registry,
)

#: 注册表文件名（相对因子库目录）。
REGISTRY_FILENAME: str = "registry.json"


def registry_path(directory: str | Path) -> Path:
    """返回因子库目录下 registry 文件的路径（不判断是否存在）。"""
    return Path(directory) / REGISTRY_FILENAME


def load_registry(directory: str | Path) -> Registry:
    """读取 ``<directory>/registry.json`` 并严格解析。

    文件缺失、JSON 非法或结构不符均抛 :class:`FactorLibError`。
    """
    path = registry_path(directory)
    if not path.is_file():
        raise FactorLibError(f"registry 文件不存在：{path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FactorLibError(f"registry JSON 解析失败：{path}（{exc}）") from exc
    return Registry.from_dict(raw, where=str(path))


def save_registry(registry: Registry, directory: str | Path) -> Path:
    """原子写入 ``<directory>/registry.json``，返回落盘路径。

    先写同目录下的临时文件并 ``fsync``，再 :func:`os.replace` 覆盖目标；
    任一步失败都会清理临时文件，目标文件保持原样。
    """
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    path = registry_path(dir_path)
    payload = (
        json.dumps(registry.to_dict(), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n"
    )

    fd, tmp_name = tempfile.mkstemp(
        dir=str(dir_path), prefix=f".{REGISTRY_FILENAME}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return path


def pool_factors(registry: Registry) -> tuple[FactorEntry, ...]:
    """返回 ``status == "pool"`` 的条目，顺序与 registry 一致。"""
    return tuple(entry for entry in registry.factors if entry.status == STATUS_POOL)


def register_factor(registry: Registry, entry: FactorEntry) -> Registry:
    """返回追加 ``entry`` 后的新 :class:`Registry`。

    ``entry.factor_id`` 与已有条目冲突时抛 :class:`FactorLibError`。
    """
    if registry.get(entry.factor_id) is not None:
        raise FactorLibError(f"factor_id 已存在：{entry.factor_id}")
    return Registry(version=registry.version, factors=registry.factors + (entry,))


__all__ = [
    "REGISTRY_FILENAME",
    "load_registry",
    "pool_factors",
    "register_factor",
    "registry_path",
    "save_registry",
]
