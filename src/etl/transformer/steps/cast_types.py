from etl.transformer.steps.step import Step
from pyspark.sql import DataFrame

class CastTypes(Step):
    """
        Cast specified columns to given data types.

        :param schema: Dictionary mapping column names to target data types.
               
               like: {"amount": "double", "date": "timestamp", "id": "integer"}
        
        Example:
            
            [id] [name]    [age]
            [1]  [Alice]   [30]
            [2]  [Bob]     [25]
            [3]  [Charlie] [35]
            
            CastTypes(schema={'age': 'integer'}) 
            # will cast the 'age' column to integer type.

    """

    def __init__(self, schema: dict[str, str]):
        self.schema = schema

    def apply(self, df: DataFrame) -> DataFrame:
        """
        Apply the CastTypes transformation to the DataFrame.

        :param df: Input DataFrame to transform
        :return: Transformed DataFrame with specified columns cast to target data types
        """
        for column, dtype in self.schema.items():
            df = df.withColumn(column, df[column].cast(dtype))
        return df
        
    def __repr__(self) -> str:
        return f"CastTypes(schema={self.schema})"