from typing import Any
from etl.generic import Step
import polars as pl


class FillBlank(Step):
    """
    Fill blank cells with per-column defaults.

    "Blank" means null, and for string columns also an empty or whitespace-only
    string. Non-string columns treat only null as blank. Unlike FillNulls this
    step also catches "" / "  ", and can be limited to rows matching 'when'.

        :param values: {column: default} - the default written into blank cells.
        :param when: optional Polars boolean expression; a blank cell is filled only
                     on rows where it is true (others keep their blank value).

        Example:
            FillBlank({"currency": "KZT", "amount": 0})
            FillBlank({"currency": "KZT"}, when=pl.col("bank") == "acme")

    """

    def __init__(self, values: dict[str, Any], when: pl.Expr | None = None):
        self.values = values
        self.when = when

    async def apply(self, df: pl.DataFrame, data=None):
        exprs = []
        for column, default in self.values.items():
            col = pl.col(column)
            blank = col.is_null()
            if df.schema[column] == pl.String:
                blank = blank | (col.str.strip_chars() == "") # "" and "  " count as blank
            if self.when is not None:
                blank = blank & self.when
            exprs.append(pl.when(blank).then(pl.lit(default)).otherwise(col).alias(column))
        return df.with_columns(exprs)

    def __repr__(self):
        return f"FillBlank(values={self.values}, when={self.when is not None})"