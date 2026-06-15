from etl.transformer.steps.step import Step
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

class NormalizeNumeric(Step):
    """
        Normalize numeric columns in the DataFrame using min-max scaling.

        :param columns: List of numeric column names to normalize.
        
        Example:
            
            [id] [age] [income]
            [1]  [30]  [50000]
            [2]  [25]  [60000]
            [3]  [35]  [55000]
            
            NormalizeNumeric(columns=['age', 'income']) # will normalize the 'age' and 'income' columns using min-max scaling.

    """

    def __init__(self, columns: list[str], method: str = "minmax"):
        self.columns = columns
        self.method = method

    def apply(self, df: DataFrame) -> DataFrame:
        """
        Apply the NormalizeNumeric transformation to the DataFrame.

        :param df: Input DataFrame to transform
        :return: Transformed DataFrame with numeric columns normalized
        """

        result = df
        for col_name in self.columns:
            if self.method == "minmax":
                stats = result.agg(
                    F.min(col_name).alias("min_val"),
                    F.max(col_name).alias("max_val"),
                ).collect()[0]
                min_val, max_val = stats["min_val"], stats["max_val"]
                if max_val != min_val:
                    result = result.withColumn(
                        col_name,
                        (F.col(col_name) - F.lit(min_val)) / F.lit(max_val - min_val),
                    )
            elif self.method == "zscore":
                stats = result.agg(
                    F.mean(col_name).alias("mean_val"),
                    F.stddev(col_name).alias("std_val"),
                ).collect()[0]
                mean_val, std_val = stats["mean_val"], stats["std_val"]
                if std_val and std_val != 0:
                    result = result.withColumn(
                        col_name,
                        (F.col(col_name) - F.lit(mean_val)) / F.lit(std_val),
                    )
        return result

    def __repr__(self) -> str:
        return f"NormalizeNumeric(columns={self.columns}, method='{self.method}')"