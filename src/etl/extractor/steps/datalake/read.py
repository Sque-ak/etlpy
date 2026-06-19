import asyncio
from etl.generic.step import Step, StopPipeline
from etl.loader.storage import read, Layer, Mode
from polars import DataFrame


class Read(Step):
    """
    Read a parquet file from a data lake layer and make it the pipeline df.

    Source step of a load pipeline: the incoming df is ignored, the read frame
    is returned and flows downstream.

        :param layer: lake layer to read from (e.g. Storage.Layer.FACT).
        :param name: file stem; ".parquet" is appended.
        :param date: date partition; None = today.
        :param mode: read from static, or date
        :param missing_ok: if is true then pipeline must be continue.
    """

    def __init__(self, layer: Layer, name: str, date: str | None = None, mode: Mode | None = None , missing_ok: bool = False):
        self.layer, self.name, self.date, self.mode, self.missing_ok = layer, name, date, mode, missing_ok

    async def apply(self, df: DataFrame=None, data=None):
        try:
            return await asyncio.to_thread(read, self.layer, f"{self.name}.parquet", self.date, self.mode)
        except FileNotFoundError:
            if self.missing_ok:
                print(f" [read] file not found, returning empty df: {self.layer}/{self.name}.parquet")
                return DataFrame()
            raise StopPipeline(message="The file wasn't found.")


    def __repr__(self):
        return f"Read(layer={self.layer}, name={self.name!r}, date={self.date!r})"