"""因子库注册表：schema 定义、严格校验与 ``registry.json`` 读写。

对外入口：

- :mod:`quant.factor_lib.schema`：frozen 值对象与解析校验。
- :mod:`quant.factor_lib.registry`：加载、原子保存、pool 过滤、注册。
- :mod:`quant.factor_lib.correlation`：新因子对库内 pool 因子的行为相关性查重。

生产侧 :func:`quant.daily.pipeline.discover_factors` 从此模块读取因子集，
只加载 ``status == "pool"`` 的条目。
"""
from quant.factor_lib.correlation import (
    CORR_METHOD,
    CorrelationReport,
    cross_section_corr,
    load_library_values,
    max_library_corr,
)
from quant.factor_lib.registry import (
    REGISTRY_FILENAME,
    load_registry,
    pool_factors,
    register_factor,
    registry_path,
    save_registry,
)
from quant.factor_lib.schema import (
    KNOWN_LINEAGE_OPS,
    REGISTRY_VERSION,
    STATUS_GRAVEYARD,
    STATUS_POOL,
    VALID_STATUSES,
    Direction,
    FactorEntry,
    FactorLibError,
    Lineage,
    Registry,
)

__all__ = [
    "CORR_METHOD",
    "KNOWN_LINEAGE_OPS",
    "REGISTRY_FILENAME",
    "REGISTRY_VERSION",
    "STATUS_GRAVEYARD",
    "STATUS_POOL",
    "VALID_STATUSES",
    "CorrelationReport",
    "Direction",
    "FactorEntry",
    "FactorLibError",
    "Lineage",
    "Registry",
    "cross_section_corr",
    "load_library_values",
    "load_registry",
    "max_library_corr",
    "pool_factors",
    "register_factor",
    "registry_path",
    "save_registry",
]
