"""因子接口规范：契约常量、动态加载器、schema 校验、复杂度度量与截断重算检测。

因子文件的接口契约见 :mod:`quant.factor_api.spec`。
"""
from quant.factor_api.complexity import (
    ComplexityReport,
    measure_complexity,
)
from quant.factor_api.loader import (
    FACTOR_FUNCTION_NAME,
    FactorCompute,
    FactorLoadError,
    load_factor,
)
from quant.factor_api.schema import validate_input, validate_output
from quant.factor_api.spec import (
    FACTOR_INPUT_COLUMNS,
    FACTOR_INPUT_SCHEMA,
    FACTOR_OUTPUT_KEY_COLUMNS,
    FACTOR_OUTPUT_SCHEMA,
)
from quant.factor_api.truncation import (
    DEFAULT_WARMUP_DAYS,
    TruncationFailure,
    TruncationResult,
    check_truncation,
)

__all__ = [
    "DEFAULT_WARMUP_DAYS",
    "FACTOR_FUNCTION_NAME",
    "FACTOR_INPUT_COLUMNS",
    "FACTOR_INPUT_SCHEMA",
    "FACTOR_OUTPUT_KEY_COLUMNS",
    "FACTOR_OUTPUT_SCHEMA",
    "ComplexityReport",
    "FactorCompute",
    "FactorLoadError",
    "TruncationFailure",
    "TruncationResult",
    "check_truncation",
    "load_factor",
    "measure_complexity",
    "validate_input",
    "validate_output",
]
