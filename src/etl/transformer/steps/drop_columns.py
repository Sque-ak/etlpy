from etl.transformer.steps.step import Step
from pyspark.sql import DataFrame

class DropColumns(Step):
    """
        Drop specified columns from the DataFrame.

        :param columns: List of column names to drop.
        
        Example:
            
            [id] [name]  [email]
            [1]  [Alice] [a@m.r]
            [2]  [Bob]   [b@m.r]
            [3]  [Charlie] [c@m.r]
            
            DropColumns(columns=['email']) # will drop the 'email' column from the DataFrame.

    """

    def __init__(self, columns: list[str]):
        self.columns = columns

    def apply(self, df: DataFrame) -> DataFrame:
        """
        Apply the DropColumns transformation to the DataFrame.

        :param df: Input DataFrame to transform
        :return: Transformed DataFrame with specified columns dropped
        """
        return df.drop(*self.columns)
        
    def __repr__(self) -> str:
        return f"DropColumns(columns={self.columns})"