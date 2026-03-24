from etl.transformer.steps.step import Step
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window


class GenerateKey(Step):
    """
    Generate a unique primary key column based on one or more source columns.

    Supports three modes controlled by the ``mode`` parameter:

    *hash (default)
        SHA-256 hash of the concatenated source columns — deterministic and
        reproducible across runs. Returns a 64-char hex string.

    *hash_int
        Deterministic 32-bit integer hash (MurmurHash3) of the source columns.
        Suitable for ClickHouse Int32 primary keys.

    *sequential
        Auto-incrementing integer IDs (1, 2, 3, …). The order is defined by
        ``order_by`` columns if provided, otherwise the current DataFrame order
        is used.

    :param columns: Column name(s) used as key source (for ``mode="hash"``
        and ``mode="hash_int"``). Ignored when ``mode="sequential"``.
    :param key_name: Name of the generated PK column (default: ``"pk"``).
    :param mode: ``"hash"``, ``"hash_int"``, or ``"sequential"`` (default: ``"hash"``).
    :param separator: Separator between column values before hashing (default: ``"||"``).
        Only used in hash / hash_int modes.
    :param order_by: Column(s) to order rows before assigning sequential IDs.
        Only used in sequential mode. ``None`` keeps the current order.

    Example (hash)::

        GenerateKey(columns=["company_id", "date"], key_name="pk")

        [pk]          [company_id] [date]       [amount]
        [a1b2c3...]   [1]          [2026-01-01] [100]
        [d4e5f6...]   [2]          [2026-01-01] [200]

    Example (hash_int — for ClickHouse Int32)::

        GenerateKey(columns=["account_id", "date"], key_name="pk", mode="hash_int")

        [pk]          [account_id] [date]       [amount]
        [1923847562]  [1]          [2026-01-01] [100]
        [-438291004]  [2]          [2026-01-01] [200]

    Example (sequential)::

        GenerateKey(key_name="id", mode="sequential", order_by="date")

        [id] [company_id] [date]       [amount]
        [1]  [1]          [2026-01-01] [100]
        [2]  [2]          [2026-01-01] [200]
        [3]  [3]          [2026-01-02] [300]
    """

    def __init__(
        self,
        columns: str | list[str] | None = None,
        key_name: str = "pk",
        mode: str = "hash",
        separator: str = "||",
        order_by: str | list[str] | None = None,
    ) -> None:
        self.columns = (
            [columns] if isinstance(columns, str) else list(columns)
            if columns is not None else []
        )
        self.key_name = key_name
        self.mode = mode
        self.separator = separator
        self.order_by = (
            [order_by] if isinstance(order_by, str) else list(order_by)
            if order_by is not None else []
        )

    def apply(self, df: DataFrame) -> DataFrame:
        if self.mode == "sequential":
            return self._apply_sequential(df)
        if self.mode == "hash_int":
            return self._apply_hash_int(df)
        return self._apply_hash(df)

    # Hash mode
    def _apply_hash(self, df: DataFrame) -> DataFrame:
        concat_expr = F.concat_ws(
            self.separator,
            *[F.col(c).cast("string") for c in self.columns],
        )
        hash_expr = F.sha2(concat_expr, 256)

        return df.withColumn(self.key_name, hash_expr).select(
            self.key_name, *[c for c in df.columns]
        )

    # Hash-int mode (MurmurHash3 → Int32)
    def _apply_hash_int(self, df: DataFrame) -> DataFrame:
        hash_expr = F.hash(
            *[F.col(c).cast("string") for c in self.columns]
        )
        return df.withColumn(self.key_name, hash_expr).select(
            self.key_name, *[c for c in df.columns]
        )

    # Sequential mode  (1, 2, 3, …)
    def _apply_sequential(self, df: DataFrame) -> DataFrame:
        if self.order_by:
            order_cols = [F.col(c) for c in self.order_by]
            window = Window.orderBy(*order_cols)
        else:
            window = Window.orderBy(F.monotonically_increasing_id())

        result = df.withColumn(self.key_name, F.row_number().over(window))
        return result.select(self.key_name, *[c for c in df.columns])

    def __repr__(self) -> str:
        return (
            f"GenerateKey(columns={self.columns!r}, "
            f"key_name={self.key_name!r})"
        )
