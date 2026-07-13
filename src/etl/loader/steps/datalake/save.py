from __future__ import annotations
import asyncio

from polars import DataFrame

from etl.loader.storage import write, Mode, Layer
from etl.generic.context import Data
from etl.generic.step import Step, StopPipeline


class Save(Step):
    """
    Idempotently save the incoming DataFrame, then pass it through unchanged.

    :param name: file stem; ".parquet" is appended.
    :param layer: target lake layer (e.g. Storage.Layer.RAW).
    :param date: date partition to write into; None = today (pass Airflow's ds for backfills).
    :param mode: mode of save, like static file or date file.
    :param missing_ok: if df hasn't exist - safe stop pipeline or error.
    """

    def __init__(self, name: str, layer: Layer = "raw", date: str | None = None, mode: Mode | None = None, missing_ok: bool = False):
        self.layer, self.name, self.date, self.mode, self.missing_ok = layer, name, date, mode, missing_ok

    async def apply(self, df: DataFrame, data: Data):

        if df is None:
            if self.missing_ok == True:
                raise StopPipeline(message="dataframe is None")
            else:
                raise ValueError(f"dataframe is None")
        
        
        await asyncio.to_thread(write, self.layer, df, f"{self.name}.parquet", self.date, mode=self.mode, overwrite=True)
        return df

    def __repr__(self):
        return f"Save(name={self.name!r}, layer={self.layer!r}, date={self.date!r})"
