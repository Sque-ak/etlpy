from etl.generic.step import Step
from polars import DataFrame
from typing import Callable

class Lambda(Step):
    """
        Apply a custom function to the DataFrame (escape hatch for one-off Polars logic).

        :param func: callable taking a DataFrame and returning a DataFrame.

        Example:
            Lambda(lambda df: df.with_columns((pl.col('age') + 10).alias('age_plus_10')))
    """

    def __init__(self, func: Callable[[DataFrame], DataFrame]):
        self.func = func

    async def apply(self, df: DataFrame, data = None):
        return self.func(df)
    
    def __repr__(self) -> str:
        return f"Lambda(func={self.func})"