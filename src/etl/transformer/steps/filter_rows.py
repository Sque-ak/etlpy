from etl.transformer.steps.step import Step
from pyspark.sql import DataFrame

class FilterRows(Step):
    """
        Filter rows in the DataFrame based on a specified condition.

        :param condition: A string representing the filter condition, using Spark SQL syntax.
        
        Example:
            
            [id] [name]    [age]
            [1]  [Alice]   [30]
            [2]  [Bob]     [25]
            [3]  [Charlie] [35]
            
            FilterRows(condition='age > 30') # will keep only row 3.

    """

    def __init__(self, condition: str):
        self.condition = condition

    def apply(self, df: DataFrame) -> DataFrame:
        """
        Apply the FilterRows transformation to the DataFrame.

        :param df: Input DataFrame to transform
        :return: Transformed DataFrame with rows filtered based on the condition
        """
        return df.filter(self.condition)
        
    def __repr__(self) -> str:
        return f"FilterRows(condition='{self.condition}')"