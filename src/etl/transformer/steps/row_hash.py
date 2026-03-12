from etl.transformer.steps.step import Step
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


class RowHash(Step):
    """
    Generates a SHA-256 hash of all columns in a row.

    Automatically excludes the following columns from the hash calculation to avoid circular dependencies and ensure stability:
      - ``row_hash``   (the result itself)
      - ``_loaded_at`` (meta field for load timestamp)
      - any columns from ``exclude``

    Result: a new column ``row_hash`` (string, 64-character hex).

    Example::

        Pipeline([
            ...,
            RowHash(),                        # hash all columns
            AddColumn("_loaded_at", F.current_timestamp()),
        ])
    """

    _META = frozenset({"row_hash", "_loaded_at"})

    def __init__(self, exclude: list[str] | None = None, separator: str = "||") -> None:
        self.exclude = set(exclude or [])
        self.separator = separator

    def apply(self, df: DataFrame) -> DataFrame:
        skip = self._META | self.exclude
        cols = [c for c in df.columns if c not in skip]

        concat_expr = F.concat_ws(
            self.separator,
            *[F.coalesce(F.col(c).cast("string"), F.lit("__null__")) for c in cols],
        )
        return df.withColumn("row_hash", F.sha2(concat_expr, 256))

    def __repr__(self) -> str:
        return f"RowHash(exclude={sorted(self.exclude)!r}, separator={self.separator!r})"