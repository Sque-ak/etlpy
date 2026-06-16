from etl.generic.step import Step
import polars as pl


class ClearText(Step):
    """ Clean text fields by removing special characters (\\r\\n, quotes) and trimming whitespace.

        :param columns: list of columns to clean

        Example:

            [id] [name]           [email]
            [1]  [ Alice ]        [a@m.r]
            [2]  [Bob\\r\\nSmith] [b@m.r]
            [3]  [Charlie]        ["c@m.r"]

            CleanText(columns=["name", "email"]) or ClearText("*")

            [id] [name]      [email]
            [1]  [Alice]     [a@m.r]
            [2]  [Bob Smith] [b@m.r]
            [3]  [Charlie]   [c@m.r]

    """

    def __init__(self, columns: list[str] | None = None):
        self.columns = columns

    async def apply(self, df: pl.DataFrame, data = None):
        target = pl.col(pl.String) if self.columns == None else pl.col(self.columns)
        return df.with_columns(
            target.fill_null("")
                  .str.replace_all(r"[\r\n]+", " ")
                  .str.replace_all(r'[\\"]+', "")
                  .str.strip_chars()
        )

    def __repr__(self):
        return f"Cleartext(columns={self.columns})"