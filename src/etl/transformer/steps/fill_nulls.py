from etl.generic.step import Step
import polars as pl

class FillNulls(Step):
    """
        Fill null values with a given value.

        :param value: scalar, or a dict {column: value} for per-column values.
        :param columns: columns to fill (when value is a scalar). None = all columns.

        Example:
            FillNulls(value='unknown', columns=['name', 'email'])
            FillNulls(value=0, columns=['id'])
            FillNulls(value={'name': 'unknown', 'amount': 0})
    """

    def __init__(self, value=None, columns: list[str] | None = None):
        self.value = value  
        self.columns = columns

    async def apply(self, df: pl.DataFrame, data=None):
        if isinstance(self.value, dict):
            return df.with_columns([pl.col(column).fill_null(value) for column, value in self.value.items()])
        if self.columns is not None:
            return df.with_columns([pl.col(column).fill_null(self.value) for column in self.columns])
        return df.fill_null(self.value)
        
    def __repr__(self) -> str:
        return f"FillNulls(value={self.value}, columns={self.columns})"