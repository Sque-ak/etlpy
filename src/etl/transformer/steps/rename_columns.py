from etl.transformer.steps.step import Step
from pyspark.sql import DataFrame

class RenameColumns(Step):
    """
        Rename columns in the DataFrame based on a provided mapping.

        :param columns_mapping: Dictionary mapping old column names to new column names.
        
        Example:
            
            [id] [name]  [email]
            [1]  [Alice] [a@m.r]
            [2]  [Bob]   [b@m.r]
            [3]  [Charlie] [c@m.r]
            
            RenameColumns(columns_mapping={'name': 'full_name', 'email': 'contact_email'}) 
            # will rename 'name' to 'full_name' and 'email' to 'contact_email'.

    """

    def __init__(self, columns_mapping: dict[str, str]):
        self.columns_mapping = columns_mapping

    def apply(self, df: DataFrame) -> DataFrame:
        """
        Apply the RenameColumns transformation to the DataFrame.

        :param df: Input DataFrame to transform
        :return: Transformed DataFrame with columns renamed
        """
        for old, new in self.columns_mapping.items():
            df = df.withColumnRenamed(old, new)
        return df
        
    def __repr__(self) -> str:
        return f"RenameColumns(columns_mapping={self.columns_mapping})"