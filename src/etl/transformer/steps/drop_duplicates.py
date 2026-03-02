from etl.transformer.steps.step import Step
from pyspark.sql import DataFrame

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

    def apply(self, df: DataFrame) -> DataFrame:
        """
        Apply the DropDuplicates transformation to the DataFrame.

        :param df: Input DataFrame to transform
        :return: Transformed DataFrame with duplicate rows dropped
        """
        if self.subset is not None:
            return df.dropDuplicates(subset=self.subset)
        else:
            return df.dropDuplicates()
        
    def __repr__(self) -> str:
        return f"DropDuplicates(subset={self.subset})"