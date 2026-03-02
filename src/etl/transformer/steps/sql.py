from etl.transformer.steps.step import Step
from pyspark.sql import DataFrame

class SQL(Step):
    """
        Apply a SQL query to the DataFrame.

        :param query: A string representing the SQL query to execute, using Spark SQL syntax.
        :param view_name: Optional name for the temporary view created from the DataFrame (default is "source").

        Example:
            
            [id] [name]    [age]
            [1]  [Alice]   [30]
            [2]  [Bob]     [25]
            [3]  [Charlie] [35]
            
            SQL(query='SELECT name, age FROM source WHERE age > 30') # will return a DataFrame with rows where age > 30 and only the 'name' and 'age' columns.

    """

    def __init__(self, query: str, view_name: str = "source"):
        self.query = query
        self.view_name = view_name

    def apply(self, df: DataFrame) -> DataFrame:
        """
        Apply the SQL transformation to the DataFrame.

        :param df: Input DataFrame to transform
        :return: Transformed DataFrame resulting from the SQL query
        """
        df.createOrReplaceTempView(self.view_name)
        return df.sparkSession.sql(self.query)
    
    def __repr__(self) -> str:
        return f"SQLStep(query='{self.query[:50]}...', view_name='{self.view_name}')"