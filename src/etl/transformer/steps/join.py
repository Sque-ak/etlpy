from __future__ import annotations
from polars import DataFrame
from etl.generic.step import Step
from etl.generic.pipeline import Pipeline

class Join(Step):
    """
    Join the incoming DataFrame (left) with another DataFrame (right).

    Parameters
    ----------
    other : DataFrame
        The right-side DataFrame to join with.
    on : str | list[str]
        Join key column name(s), present in both frames.
    how : str, default "inner"
        Polars join type: "inner", "left", "right", "full", "semi", "anti", "cross".
        Polars names differ from pyspark: "outer" -> "full",
        "left_semi" -> "semi", "left_anti" -> "anti".
    select : list[str] | None, default None
        Columns to keep from the right frame (join keys are always kept).
        None keeps every column of 'other'.
    prefix : str | None, default None
        Prefix for the right frame's non-key columns, to avoid collisions,
        e.g. prefix="company_" -> "company_name".

    Example:
        Join(other=companies, on="company_id", select=["company_name", "bin"], how="left")
    """

    def __init__(
        self,
        other: DataFrame | Pipeline,
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

    async def apply(self, df: DataFrame, data = None):
        right = self.other
        
        if isinstance(right, Pipeline):
            right = await right.run()

        return df.join(self._prepare_right(right), on=self.on, how=self.how)


    def _prepare_right(self, right: DataFrame) -> DataFrame:
        """Select requested columns and prefix the right frame's non-key columns."""

        if self.select is not None:
            keep = list(dict.fromkeys(self.on + self.select))  # deduplicate, preserve order
            right = right.select(keep)

        if self.prefix:
            right = right.rename(
                {column: f"{self.prefix}{column}" for column in right.columns if column not in self.on}
            )

        return right

    def __repr__(self) -> str:
        return (
            f"Join(on={self.on!r}, how={self.how!r}, "
            f"select={self.select!r}, prefix={self.prefix!r})"
        )
