from etl.generic.step import Step
import polars as pl


class NormalizeNumeric(Step):
    """
        Normalize numeric columns with min-max scaling or z-score.

        :param columns: numeric column names to normalize.
        :param method: "minmax" (default) -> (x - min) / (max - min);
                       "zscore" -> (x - mean) / std.
        Constant columns (zero denominator) are left unchanged.

        Example:
            NormalizeNumeric(columns=['age', 'income'])
    """

    def __init__(self, columns: list[str], method: str = "minmax"):
        self.columns = columns
        self.method = method

    async def apply(self, df: pl.DataFrame, data = None):
        exprs = []
        for column in self.columns:
            x = pl.col(column)
            if self.method == "zscore":
                mu, sd = x.mean(), x.std()
                expr = pl.when(sd != 0).then((x - mu) / sd).otherwise(x)
            else:
                lo, hi = x.min(), x.max()
                expr = pl.when(hi != lo).then((x - lo) / (hi - lo)).otherwise(x)
            exprs.append(expr.alias(column))
        return df.with_columns(exprs)

    def __repr__(self) -> str:
        return f"NormalizeNumeric(columns={self.columns}, method='{self.method}')"