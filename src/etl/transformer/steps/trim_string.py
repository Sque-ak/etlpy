from etl.transformer.steps.step import Step
from pyspark.sql import DataFrame
from pyspark.sql.functions import trim, col
from pyspark.sql.types import StringType

class TrimString(Step):
    """
        Trim leading and trailing whitespace from string columns in the DataFrame.

        :param columns: Optional list of column names to trim. If None, all string columns will be trimmed.

        Example:
            
            [id] [name]  
            [1]  [ Alice ] 
            [2]  [ Bob ]   
            [3]  [ Charlie ] 
            
            TrimString() # will trim the whitespace from the 'name' column.

    """

    def __init__(self, columns: list[str] | None = None):
        self.columns = columns

    def apply(self, df: DataFrame) -> DataFrame:
        """
        Apply the TrimString transformation to the DataFrame.

        :param df: Input DataFrame to transform
        :return: Transformed DataFrame with string columns trimmed
        """
        cols = self.columns
        if not cols:
            cols = [f.name for f in df.schema.fields if isinstance(f.dataType, StringType)]

        result = df
        for c in cols:
            result = result.withColumn(c, trim(col(c)))
        return result
    
    def __repr__(self) -> str:
        return f"TrimString(columns={self.columns})"