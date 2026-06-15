from etl.transformer.steps import Step
from pyspark.sql import DataFrame, functions as F


class ClearText(Step):
    """ Clean text fields by removing special characters (\\r\\n, quotes) and trimming whitespace.

        :param columns: list of columns to clean

        Example:

            [id] [name]           [email]
            [1]  [ Alice ]        [a@m.r]
            [2]  [Bob\\r\\nSmith] [b@m.r]
            [3]  [Charlie]        ["c@m.r"]

            CleanText(columns=["name", "email"])

            [id] [name]      [email]
            [1]  [Alice]     [a@m.r]
            [2]  [Bob Smith] [b@m.r]
            [3]  [Charlie]   [c@m.r]

    """

    def __init__(self, columns: list[str]):
        super().__init__()
        self.columns = columns

    def apply(self, df: DataFrame) -> DataFrame:
        for col_name in self.columns:
            col = F.col(col_name)
            cleaned = F.trim(F.regexp_replace(
                F.regexp_replace(F.coalesce(col, F.lit("")), r'[\r\n]+', ' '),
                r'[\\\"]+', ''
            ))
            df = df.withColumn(col_name, cleaned)
        return df