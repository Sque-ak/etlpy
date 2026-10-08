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
        other: pl.DataFrame | Pipeline | list[pl.DataFrame],
        how: str | None = "vertical"
    ) -> None:
        self.other, self.how = other, how

    async def apply(self, df: pl.DataFrame, data = None):
        right = self.other
        
        if isinstance(right, Pipeline):
            right = await right.run()


        others = list(right) if isinstance(right, list) else [right]
        frames = [f for f in [df, *others] if f is not None and not f.is_empty()]
        if not frames:
            return df if df is not None else pl.DataFrame()
        
        return pl.concat(frames, how=self.how)


    def __repr__(self) -> str:
        return (f"Union()")
