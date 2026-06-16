from etl.generic.step import Step
from polars import DataFrame

class DropDuplicates(Step):
    """
        Drop duplicate rows from the DataFrame based on specified columns.

        :param subset: List of column names to consider for identifying duplicates. If None, all columns are considered.
        
        Example:
            
            [id] [name]  [email]
            [1]  [Alice] [a@m.r]
            [2]  [Bob]   [b@m.r]
            [3]  [Alice] [a@m.r]
            
            DropDuplicates(subset=['name', 'email']) # will drop row 3 as it is a duplicate of row 1 based on 'name' and 'email'.

    """

    def __init__(self, subset: list[str] | None = None):
        self.subset = subset

    async def apply(self, df: DataFrame, data=None) -> DataFrame:
        return df.unique(subset=self.subset, keep="first", maintain_order=True)

        
    def __repr__(self) -> str:
        return f"DropDuplicates(subset={self.subset})"