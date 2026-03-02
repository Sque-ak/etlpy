from etl.transformer.steps.step import Step
from pyspark.sql import DataFrame
from typing import Callable

class Lambda(Step):
    """
        Apply a custom transformation function to the DataFrame.

        :param func: A function that takes a DataFrame as input and returns a transformed DataFrame.
        
        Example:
            
            [id] [name]  [age]
            [1]  [Alice] [30]
            [2]  [Bob]   [25]
            [3]  [Charlie] [35]
            
            Lambda(func=lambda df: df.withColumn('age_plus_10', df['age'] + 10)) 
            # will add a new column 'age_plus_10' with age values increased by 10.

    """

    def __init__(self, func: Callable[[DataFrame], DataFrame]):
        self.func = func

    def apply(self, df: DataFrame) -> DataFrame:
        """
        Apply the Lambda transformation to the DataFrame.

        :param df: Input DataFrame to transform
        :return: Transformed DataFrame resulting from the custom function
        """
        return self.func(df)
    
    def __repr__(self) -> str:
        return f"LambdaStep(func={self.func})"