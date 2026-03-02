from etl.transformer.steps.step import Step
from pyspark.sql import DataFrame

class FillNulls(Step):
    """
        Fill null values in specified columns with a given value.

        :param value: The value to replace nulls with. Can be a scalar or a dictionary mapping column names to values.
        :param columns: List of column names to fill null values in. If None, all columns are filled.
        
        Example:
            
            [id] [name]  [email]
            [1]  [Alice] [a@m.r]
            [2]  [Bob]   [null]
            [3]  [null]  [null]
            
            FillNulls(value='unknown', columns=['name', 'email']) # will fill nulls in 'name' and 'email' with 'unknown'.
            FillNulls(value=0, columns=['id']) # will fill nulls in 'id' with 0.

    """

    def __init__(self, value=None, columns: list[str] | None = None):
        self.value = value  
        self.columns = columns

    def apply(self, df: DataFrame) -> DataFrame:
        """
        Apply the FillNulls transformation to the DataFrame.

        :param df: Input DataFrame to transform
        :return: Transformed DataFrame with null values filled
        """        
        if self.columns is not None:
            return df.fillna(self.value, subset=self.columns)
        else:
            return df.fillna(self.value)
        
    def __repr__(self) -> str:
        return f"FillNulls(value={self.value}, columns={self.columns})"