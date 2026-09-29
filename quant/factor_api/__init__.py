"""因子接口规范：契约常量、动态加载器与输入输出 schema 校验。

因子文件的接口契约见 :mod:`quant.factor_api.spec`。
"""
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

__all__ = [
    "FACTOR_FUNCTION_NAME",
    "FACTOR_INPUT_COLUMNS",
    "FACTOR_INPUT_SCHEMA",
    "FACTOR_OUTPUT_KEY_COLUMNS",
    "FACTOR_OUTPUT_SCHEMA",
    "FactorCompute",
    "FactorLoadError",
    "load_factor",
    "validate_input",
    "validate_output",
]
