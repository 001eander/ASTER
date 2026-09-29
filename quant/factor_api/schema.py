"""因子输入输出 schema 校验：复用既有 ``check_schema``，只补充输出键去重。"""
from __future__ import annotations

import polars as pl

from quant.data.schema import SchemaError, check_schema
from quant.factor_api.spec import (
    FACTOR_INPUT_SCHEMA,
    FACTOR_OUTPUT_KEY_COLUMNS,
    FACTOR_OUTPUT_SCHEMA,
)


def validate_input(df: pl.DataFrame) -> None:
    """校验因子输入：恰好 10 列，列名、顺序、dtype 均需符合契约。

    不符即抛 :class:`quant.data.schema.SchemaError`。
    """
    check_schema(df, FACTOR_INPUT_SCHEMA, name="因子输入")


def validate_output(df: pl.DataFrame) -> None:
    """校验因子输出：恰好 ``(date, instrument, value)`` 三列且无重复键。

    列名、顺序、dtype 由 ``check_schema`` 校验；另检查 ``(date, instrument)``
    不重复。行序不作要求。不符即抛 :class:`quant.data.schema.SchemaError`。
    """
    check_schema(df, FACTOR_OUTPUT_SCHEMA, name="因子输出")
    key = list(FACTOR_OUTPUT_KEY_COLUMNS)
    if df.select(key).is_duplicated().any():
        raise SchemaError(
            f"因子输出存在重复的 ({', '.join(FACTOR_OUTPUT_KEY_COLUMNS)})"
        )
