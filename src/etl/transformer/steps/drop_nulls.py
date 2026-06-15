from etl.transformer.steps.step import Step
from pyspark.sql import DataFrame

class DropNulls(Step):
    """
        Drop rows with null values in specified columns.

        :param subset: List of column names to check for null values. If None, all columns are checked.
        :param how: 'any' to drop rows with any nulls, 'all' to drop rows with all nulls in the subset.
        
        Example:
            
            [id] [name]  [email]
            [1]  [Alice] [a@m.r]
            [2]  [Bob]   [null]
            [3]  [null]  [null]
            
            DropNulls(subset=['email'], how='any') # will drop row 2.
            DropNulls(subset=['name', 'email'], how='all') # will drop row 3.

    """

    def __init__(self, subset: list[str] | None = None, how: str = "any"):
        self.subset = subset
        self.how = how

    def apply(self, df: DataFrame) -> DataFrame:
        """
        Apply the DropNulls transformation to the DataFrame.

        :param df: Input DataFrame to transform
        :return: Transformed DataFrame with null values dropped
        """
        if self.subset is not None:
            return df.dropna(subset=self.subset, how=self.how)
        else:
            return df.dropna(how=self.how)
        
    def __repr__(self) -> str:
        return f"DropNulls(subset={self.subset}, how='{self.how}')"