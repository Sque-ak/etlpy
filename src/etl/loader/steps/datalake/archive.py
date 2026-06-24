import asyncio
from etl.generic.step import Step, StopPipeline
from etl.loader.storage import Layer, archive_file, Mode
from polars import DataFrame

class Archive(Step):
    """
    Move a source file from the data lake to the archive layer.

    Usually the last step of a load pipeline: once the data is in ClickHouse,
    archive the parquet so the lake stays clean. The df is passed through unchanged.

    :param layer: source layer of the file (e.g. Storage.Layer.FACT).
    :param name: file stem; ".parquet" is appended.
    :param date: date file, you can use "" for remove date in end file, or None for set today date.
    :param mode: you can archive Static file Mode.STATIC or date files Mode.DATE
    :param missing_ok: if is true then pipeline must be continue.
    """
                
    def __init__(self, layer:Layer, name:str, date:str | None = None, mode: Mode | None = None, missing_ok: bool = True):
        self.layer, self.name, self.date, self.mode, self.missing_ok = layer, name, date, mode, missing_ok

    async def apply(self, df:DataFrame, data = None): 
        try:
            await asyncio.to_thread(archive_file, self.layer, f"{self.name}.parquet", self.date, self.mode)
        except FileNotFoundError:
            if self.missing_ok:
                print(f" [archive] file not found, returning empty df: {self.layer}/{self.name}.parquet")
                return df
            raise StopPipeline(message="The file wasn't found.")
    
    def __repr__(self) -> str:
        return f"Archive(layer={self.layer}, name={self.name!r})"