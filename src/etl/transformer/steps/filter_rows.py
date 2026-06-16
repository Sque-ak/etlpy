from etl.generic.step import Step
from polars import DataFrame

class FilterRows(Step):
    """
        Keep rows matching a Polars boolean expression.

        :param condition: a Polars expression, e.g. pl.col("age") > 30

        Example:
            FilterRows(pl.col("age") > 30)
    """


    def __init__(self, condition: str):
        self.condition = condition

    async def apply(self, df: DataFrame, data=None) -> DataFrame:
        return df.filter(self.condition)
        
    def __repr__(self) -> str:
        return f"FilterRows(condition='{self.condition}')"