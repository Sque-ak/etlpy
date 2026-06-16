from etl.generic.step import Step
from polars import DataFrame

class DropNulls(Step):
    """
        Drop rows with null values in specified columns.

        :param subset: List of column names to check for null values. If None, all columns are checked.
        
        Example:
            
            [id] [name]  [email]
            [1]  [Alice] [a@m.r]
            [2]  [Bob]   [null]
            [3]  [null]  [null]
            
            DropNulls(subset=['name', 'email']) # will drop 2 and 3

    """

    def __init__(self, subset: list[str] | None = None):
        self.subset = subset

    async def apply(self, df: DataFrame, data = None) -> DataFrame:
        return df.drop_nulls(subset=self.subset)
        
    def __repr__(self) -> str:
        return f"DropNulls(subset={self.subset}')"