import polars as pl, polars_hash # noqa: F401 - registers the .chash namespace
from etl.generic.step import Step

class RowHash(Step):
    """
    SHA-256 fingerprint of all row values, written to 'row_hash'.
    Used by the loader to insert only new/changed rows.

    Excludes 'row_hash', '_loaded_at', and any columns in 'exclude'.

    Example:
        Pipeline([..., RowHash(), AddColumn("_loaded_at", pl.lit(datetime.now()))])
    """

    _META = frozenset({"row_hash", "_loaded_at"})

    def __init__(self, exclude: list[str] | None = None, separator: str = "||") -> None:
        self.exclude = set(exclude or [])
        self.separator = separator

    async def apply(self, df: pl.DataFrame, data = None):
        skip = self._META | self.exclude
        cols = sorted(column for column in df.columns if column not in skip)
        fingerprint = pl.concat_str(
            [pl.col(column).cast(pl.String).fill_null("__null__") for column in cols],
            separator=self.separator
        )

        return df.with_columns(fingerprint.chash.sha2_256().alias("row_hash"))

    def __repr__(self) -> str:
        return f"RowHash(exclude={sorted(self.exclude)!r}, separator={self.separator!r})"