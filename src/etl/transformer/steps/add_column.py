from etl.transformer.steps.step import Step
from pyspark.sql import DataFrame

class AddColumn(Step):
    """
        Add a new column to the DataFrame with a specified name and value.

        :param column_name: Name of the new column to add.
        :param value: Value to assign to the new column for all rows.
        
        Example:
            
            [id] [name]
            [1]  [Alice]
            [2]  [Bob]
            [3]  [Charlie]
            
            AddColumn(column_name='age', value=30) # will add a new column 'age' with value 30 for all rows.

    """

    def __init__(self, column_name: str, value):
        self.column_name = column_name
        self.value = value

    def apply(self, df: DataFrame) -> DataFrame:
        """
        Apply the AddColumn transformation to the DataFrame.

        :param df: Input DataFrame to transform
        :return: Transformed DataFrame with the new column added
        """
        return df.withColumn(self.column_name, self.value)
        
    def __repr__(self) -> str:
        return f"AddColumn(column_name='{self.column_name}', value={self.value})"