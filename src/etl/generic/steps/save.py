from __future__ import annotations
import asyncio

from polars import DataFrame

from etl.generic import storage
from etl.generic.context import Data
from etl.generic.step import Step


class Save(Step):
    """Idempotently save the incoming DataFrame, then pass it through unchanged."""

    def __init__(self, name: str, layer: str = "raw"):
        self.name = name
        self.layer = layer

    async def apply(self, df: DataFrame | None, data: Data) -> DataFrame | None:
        if df is None:
            return df
        await asyncio.to_thread(
            storage.write, self.layer, df, f"{self.name}.parquet", overwrite=True
        )
        return df

    def __repr__(self):
        return f"Save(name={self.name!r}, layer={self.layer!r})"
