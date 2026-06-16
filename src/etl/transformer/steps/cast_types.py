from etl.generic.step import Step
from polars import DataFrame

class CastTypes(Step):
    """
    Cast specified columns to given Polars dtypes.

    schema: {column: polars dtype}, e.g. {"age": pl.Int32, "amount": pl.Float64, "date": pl.Date}

    Example:
        CastTypes({"age": pl.Int32})
    """


    def __init__(self, schema: dict[str, str]):
        self.schema = schema

    async def apply(self, df: DataFrame, data = None):
        return df.cast(self.schema)
        
    def __repr__(self) -> str:
        return f"CastTypes(schema={self.schema})"