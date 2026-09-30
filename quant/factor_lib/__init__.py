"""因子库注册表：schema 定义、严格校验与 ``registry.json`` 读写。

对外入口：

- :mod:`quant.factor_lib.schema`：frozen 值对象与解析校验。
- :mod:`quant.factor_lib.registry`：加载、原子保存、pool 过滤、注册。
- :mod:`quant.factor_lib.correlation`：新因子对库内 pool 因子的行为相关性查重。
- :mod:`quant.factor_lib.prune`：定期整库——冗余聚簇降级与 pool 容量上限。
- :mod:`quant.factor_lib.promote`：内循环产物入库——过门控因子的源码落位与登记。

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
from quant.factor_lib.promote import (
    DEFAULT_FACTOR_LIBRARY_DIR,
    FACTOR_ID_PATTERN,
    FACTOR_SOURCE_NAME,
    METRIC_SOURCE_KEYS,
    PROMOTE_STATUS,
    SCORE_FILENAME,
    GateNotPassed,
    PromoteConflict,
    PromoteError,
    PromoteInputError,
    PromoteRequest,
    PromoteResult,
    promote_factor,
    read_gate_metrics,
)
from quant.factor_lib.prune import (
    CAPACITY_RATIO,
    REASON_CAPACITY,
    REASON_CLUSTER,
    PruneDemotion,
    PrunePlan,
    PruneResult,
    apply_prune,
    build_corr_matrix,
    capacity_limit,
    cluster_corr_threshold,
    plan_prune,
    prune,
    quality_key,
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
    "CAPACITY_RATIO",
    "CORR_METHOD",
    "DEFAULT_FACTOR_LIBRARY_DIR",
    "FACTOR_ID_PATTERN",
    "FACTOR_SOURCE_NAME",
    "KNOWN_LINEAGE_OPS",
    "METRIC_SOURCE_KEYS",
    "PROMOTE_STATUS",
    "REGISTRY_FILENAME",
    "REGISTRY_VERSION",
    "REASON_CAPACITY",
    "REASON_CLUSTER",
    "SCORE_FILENAME",
    "STATUS_GRAVEYARD",
    "STATUS_POOL",
    "VALID_STATUSES",
    "CorrelationReport",
    "Direction",
    "FactorEntry",
    "FactorLibError",
    "GateNotPassed",
    "Lineage",
    "PromoteConflict",
    "PromoteError",
    "PromoteInputError",
    "PromoteRequest",
    "PromoteResult",
    "PruneDemotion",
    "PrunePlan",
    "PruneResult",
    "Registry",
    "apply_prune",
    "build_corr_matrix",
    "capacity_limit",
    "cluster_corr_threshold",
    "cross_section_corr",
    "load_library_values",
    "load_registry",
    "max_library_corr",
    "plan_prune",
    "pool_factors",
    "promote_factor",
    "prune",
    "quality_key",
    "read_gate_metrics",
    "register_factor",
    "registry_path",
    "save_registry",
]
