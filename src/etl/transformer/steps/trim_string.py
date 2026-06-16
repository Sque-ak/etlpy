import polars as pl
from etl.generic.step import Step

class TrimString(Step):
    """
        Trim leading/trailing whitespace from string columns.

        :param columns: columns to trim; None = all string columns.

        Example:
            TrimString()                 # all string columns
            TrimString(['name'])         # only 'name'
    """

    def __init__(self, columns: list[str] | None = None):
        self.columns = columns

    async def apply(self, df: pl.DataFrame, data=None):
        target = pl.col(self.columns) if self.columns else pl.col(pl.String)
        return df.with_columns(target.str.strip_chars())

    def __repr__(self) -> str:
        return f"TrimString(columns={self.columns})"
