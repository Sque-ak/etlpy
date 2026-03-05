from __future__ import annotations
from pyspark.sql import DataFrame
from etl.transformer.steps.step import Step


class Join(Step):
    """
    Join two DataFrames on one or more columns.

    The primary DataFrame (passed to ``apply()``) is the **left** side;
    the ``other`` DataFrame supplied at construction time is the **right** side.

    Parameters
    ----------
    other : DataFrame
        The right-side DataFrame to join with.
    on : str | list[str]
        Column name(s) used as the join key.
        - A single string is treated as a column present in both DataFrames.
        - A list of strings is treated as multiple join keys.
    how : str, default ``"inner"``
        Join type — any value accepted by PySpark:
        ``"inner"``, ``"left"``, ``"right"``, ``"outer"``,
        ``"left_semi"``, ``"left_anti"``, ``"cross"``, etc.
    select : list[str] | None, default ``None``
        Columns to pick from the **right** DataFrame (in addition to the key
        columns which are always used for matching).  When ``None`` every
        column of ``other`` is included in the result.
    prefix : str | None, default ``None``
        If set, all columns coming from the right DataFrame (except the join
        keys) are prefixed with this string — handy for avoiding name
        collisions, e.g. ``prefix="company_"`` → ``company_name``.

    Example
    -------
    >>> from etl.transformer.steps import Join
    >>>
    >>> companies = spark.read.parquet("stg/companies")
    >>> transactions = spark.read.parquet("stg/transactions")
    >>>
    >>> step = Join(
    ...     other=companies,
    ...     on="company_id",
    ...     select=["company_name", "bin"],
    ...     how="left",
    ... )
    >>> result = step.apply(transactions)
    """

    def __init__(
        self,
        other: DataFrame,
        on: str | list[str],
        how: str = "inner",
        select: list[str] | None = None,
        prefix: str | None = None,
    ) -> None:
        self.other = other
        self.on = [on] if isinstance(on, str) else list(on)
        self.how = how
        self.select = select
        self.prefix = prefix

    # ------------------------------------------------------------------
    # Step interface
    # ------------------------------------------------------------------
    def apply(self, df: DataFrame) -> DataFrame:
        right = self._prepare_right()
        result = df.join(right, on=self.on, how=self.how)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _prepare_right(self) -> DataFrame:
        """Optionally select columns and apply prefix to the right DataFrame."""
        right = self.other

        # --- select only requested columns (+ join keys) ---------------
        if self.select is not None:
            keep = list(dict.fromkeys(self.on + self.select))  # deduplicate, preserve order
            right = right.select(*keep)

        # --- prefix non-key columns ------------------------------------
        if self.prefix:
            for col_name in right.columns:
                if col_name not in self.on:
                    right = right.withColumnRenamed(
                        col_name, f"{self.prefix}{col_name}"
                    )

        return right

    def __repr__(self) -> str:
        return (
            f"Join(on={self.on!r}, how={self.how!r}, "
            f"select={self.select!r}, prefix={self.prefix!r})"
        )
