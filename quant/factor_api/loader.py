"""因子加载器：从任意路径动态加载因子 ``.py`` 并校验 compute 契约。"""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
import sys
from pathlib import Path

from quant.factor_api.spec import FactorCompute

#: 因子的入口函数名。
FACTOR_FUNCTION_NAME: str = "compute"


class FactorLoadError(RuntimeError):
    """因子文件无法加载为可用的 compute 函数。"""


def _module_name(path: Path) -> str:
    """按文件路径哈希生成唯一模块名，避免多次加载同名文件相互覆盖。"""
    digest = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:16]
    return f"_aster_factor_{digest}"


def load_factor(path: Path) -> FactorCompute:
    """从 ``path`` 动态加载因子文件，返回其 ``compute`` 函数。

    校验点：文件存在且为 ``.py``、模块内定义 ``compute``、可调用、
    ``inspect.signature`` 为单参数。任一条不符抛 :class:`FactorLoadError`，
    错误信息指明具体原因。
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FactorLoadError(f"因子文件不存在：{file_path}")
    if file_path.suffix != ".py":
        raise FactorLoadError(f"因子文件必须是 .py：{file_path}")

    module_name = _module_name(file_path)
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise FactorLoadError(f"无法为因子文件创建导入规格：{file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # 因子文件顶层代码报错
        sys.modules.pop(module_name, None)
        raise FactorLoadError(
            f"导入因子文件失败：{file_path}（{type(exc).__name__}: {exc}）"
        ) from exc

    compute = getattr(module, FACTOR_FUNCTION_NAME, None)
    if compute is None:
        raise FactorLoadError(f"因子文件未定义 {FACTOR_FUNCTION_NAME}：{file_path}")
    if not callable(compute):
        raise FactorLoadError(f"因子文件的 {FACTOR_FUNCTION_NAME} 不可调用：{file_path}")

    parameters = list(inspect.signature(compute).parameters.values())
    if len(parameters) != 1:
        raise FactorLoadError(
            f"因子 {FACTOR_FUNCTION_NAME} 必须接受恰好 1 个参数，"
            f"实际 {len(parameters)} 个：{file_path}"
        )
    return compute
