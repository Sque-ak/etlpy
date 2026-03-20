from etl.transformer.steps import Step
from pyspark.sql import DataFrame, functions as F


class ExtractEntities(Step):
    """Extract entities from multiple sets of columns and combine them using UNION.

    Parameters:
        sources:  list of mappings {target_col: source_col}
        defaults: dict {target_col: default_value} replaces NULL and empty
                  strings with the given default for each specified column
    """

    def __init__(
        self,
        sources: list[dict[str, str]],
        defaults: dict[str, object] | None = None,
    ):
        super().__init__()
        self.sources = sources
        self.defaults = defaults or {}

    def apply(self, df: DataFrame) -> DataFrame:
        dfs = []
        for mapping in self.sources:
            cols = [F.col(src).alias(tgt) for tgt, src in mapping.items()]
            dfs.append(df.select(*cols))

        result = dfs[0]
        for d in dfs[1:]:
            result = result.union(d)

        for col_name, default in self.defaults.items():
            condition = F.col(col_name).isNull()
            if isinstance(default, str):
                condition = condition | (F.trim(F.col(col_name)) == "")
            result = result.withColumn(
                col_name,
                F.when(condition, F.lit(default)).otherwise(F.col(col_name)),
            )
        return result