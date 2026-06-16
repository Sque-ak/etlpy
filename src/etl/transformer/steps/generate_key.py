from etl.generic.step import Step
import polars as pl, polars_hash # noqa: F401 - import registers the .chash / .nchash namespaces

class GenerateKey(Step):
    """
    Generate a key column from one or more source columns.

    mode:
        "hash"        - SHA-256 hex of the joined columns (stable, 64 chars).
        "hash_int"    - stable 64-bit int (xxhash64), safe for integer keys at scale.
        "sequential"  - 1, 2, 3, ... ordered by order_by (or current order).

    :param columns: source column(s) for hash modes.
    :param key_name: name of the generated column (default "pk").
    :param mode: "hash" | "hash_int" | "sequential".
    :param separator: joiner between column values before hashing.
    :param order_by: ordering column(s) for sequential mode.
    """

    def __init__(
        self,
        columns = None,
        key_name = "pk",
        mode = "hash",
        separator = "||",
        order_by = None,
    ) -> None:
        self.columns = [columns] if isinstance(columns, str) else list(columns or [])
        self.key_name = key_name
        self.mode = mode
        self.separator = separator
        self.order_by = [order_by] if isinstance(order_by, str) else list(order_by or [])

    async def apply(self, df: pl.DataFrame, data = None):
        if self.mode == "sequential":
            if self.order_by:
                df = df.sort(self.order_by)
            return df.with_row_index(name=self.key_name, offset=1)
        if self.mode == "hash_int":
            return df.with_columns(self._concat().nchash.xxhash64().alias(self.key_name))
        return df.with_columns(self._concat().chash.sha2_256().alias(self.key_name))

    def _concat(self) -> pl.Expr:
        return pl.concat_str(
            [pl.col(c).cast(pl.String) for c in self.columns],
            separator=self.separator,
            ignore_nulls=True,
        )

    def __repr__(self) -> str:
        return f"GenerateKey(columns={self.columns!r}, key_name={self.key_name!r})"

