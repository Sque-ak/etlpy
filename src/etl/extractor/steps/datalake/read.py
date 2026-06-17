import asyncio
from etl.generic.step import Step
from etl.loader.storage import read, Layer
from polars import DataFrame


class Read(Step):
    """
    Read a parquet file from a data lake layer and make it the pipeline df.

    Source step of a load pipeline: the incoming df is ignored, the read frame
    is returned and flows downstream.

        :param layer: lake layer to read from (e.g. Storage.Layer.FACT).
        :param name: file stem; ".parquet" is appended.
        :param date: date partition; None = today.
    """

    def __init__(self, layer: Layer, name: str, date: str | None = None):
        self.layer, self.name, self.date = layer, name, date

    async def apply(self, df: DataFrame=None, data=None):
        return await asyncio.to_thread(read, self.layer, f"{self.name}.parquet", self.date)
    
    def __repr__(self):
        return f"Read(layer={self.layer}, name={self.name!r}, date={self.date!r})"