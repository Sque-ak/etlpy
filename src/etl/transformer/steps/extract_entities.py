from etl.generic.step import Step
import polars as pl


class ExtractEntities(Step):
    """Extract entities from multiple sets of columns and combine them using UNION.
    Scheme the tables must be same.

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

    async def apply(self, df: pl.DataFrame, data=None):
        parts = [
            df.select([pl.col(source).alias(target) for target, source in mapping.items()])
            for mapping in self.sources
        ]
        result = pl.concat(parts, how="vertical")

        exprs = []
        for col_name, default in self.defaults.items():
            col = pl.col(col_name)
            blank = col.is_null()
            if isinstance(default, str):
                blank = blank | (col.cast(pl.String).str.strip_chars() == "") # null OR blank string
            exprs.append(pl.when(blank).then(pl.lit(default)).otherwise(col).alias(col_name))
        if exprs:
            result = result.with_columns(exprs)
        return result

    def __repr__(self):
        return f"ExtractEntities(sources={self.sources}, defaults={self.defaults})"
