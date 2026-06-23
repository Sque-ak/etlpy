from __future__ import annotations
import polars as pl
from etl.generic.step import Step
from etl.generic.pipeline import Pipeline

class Union(Step):
    """
    Union the incoming DataFrame (left) with another DataFrame (right).

        :param other: pipeline or dataframe with we want union 

    Example:
        Union(other=companies)
    """

    def __init__(
        self,
        other: pl.DataFrame | Pipeline,
    ) -> None:
        self.other = other

    async def apply(self, df: pl.DataFrame, data = None):
        right = self.other
        
        if isinstance(right, Pipeline):
            right = await right.run()

        return pl.concat([df, right])


    def __repr__(self) -> str:
        return (f"Union()")
