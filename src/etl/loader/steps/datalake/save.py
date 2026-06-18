from __future__ import annotations
import asyncio

from polars import DataFrame

from etl.loader.storage import write, Mode, Layer
from etl.generic.context import Data
from etl.generic.step import Step


class Save(Step):
    """
    Idempotently save the incoming DataFrame, then pass it through unchanged.

    :param name: file stem; ".parquet" is appended.
    :param layer: target lake layer (e.g. Storage.Layer.RAW).
    :param date: date partition to write into; None = today (pass Airflow's ds for backfills).
    :param mode: mode of save, like static file or date file.
    """

    def __init__(self, name: str, layer: Layer = "raw", date: str | None = None, mode: Mode | None = None):
        self.layer, self.name, self.date, self.mode = layer, name, date, mode

    async def apply(self, df: DataFrame, data: Data):
        if df is None:
            return df
        await asyncio.to_thread(write, self.layer, df, f"{self.name}.parquet", self.date, mode=self.mode, overwrite=True)
        return df

    def __repr__(self):
        return f"Save(name={self.name!r}, layer={self.layer!r}, date={self.date!r})"
