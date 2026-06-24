from __future__ import annotations
import polars as pl
from etl.generic.step import Step
from etl.generic.pipeline import Pipeline

class Union(Step):
    """
    Union the incoming DataFrame (left) with another DataFrame (right).

        :param other: pipeline or dataframe with we want union 
        :param how: Specify the stacking strategy using the how parameter (e.g., how="vertical", how="diagonal", or how="horizontal") depending on your schema alignment

    Example:
        Union(other=companies)
    """

    def __init__(
        self,
        other: pl.DataFrame | Pipeline,
        how: str | None = "vertical"
    ) -> None:
        self.other, self.how = other, how

    async def apply(self, df: pl.DataFrame, data = None):
        right = self.other
        
        if isinstance(right, Pipeline):
            right = await right.run()

        return pl.concat([df, right], how=self.how)


    def __repr__(self) -> str:
        return (f"Union()")
