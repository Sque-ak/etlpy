from etl.generic.step import Step
from polars import DataFrame

class DropColumns(Step):
    """
        Drop specified columns from the DataFrame.

        :param columns: List of column names to drop.
        :param exclude: If true then drop exclude.

        Example:
            
            [id] [name]  [email]
            [1]  [Alice] [a@m.r]
            [2]  [Bob]   [b@m.r]
            [3]  [Charlie] [c@m.r]
            
            DropColumns(columns=['email']) # will drop the 'email' column from the DataFrame.
            DropColumns(columns=['id', 'name'], exclude=True) # keep only id, name

    """

    def __init__(self, columns: list[str], exclude: bool = False):
        self.columns = columns
        self.exclude = exclude

    async def apply(self, df: DataFrame, data = None):

        if self.exclude:
            return df.select(self.columns)

        return df.drop(*self.columns)
        
    def __repr__(self) -> str:
        kind = "keep" if self.exclude else "columns"
        return f"DropColumns({kind}={self.columns})"