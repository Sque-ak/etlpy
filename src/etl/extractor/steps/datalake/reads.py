from etl.generic import Step, StopPipeline
from etl.loader.storage import path, Mode, Layer
import asyncio, polars as pl


class Reads(Step):
    """
    Read every parquet file in one lake folder and concat them into a single df.
    Useful for datasets split across many files, e.g. manual_activity/<date>.parquet.

    Files are matched by `pattern` inside the folder and optionally filtered by an
    inclusive date range compared against the file stem (name without extension).
    Empty result: return an empty df when `missing_ok`, otherwise stop the pipeline.

        :param layer: lake layer to read from (e.g. Storage.Layer.STG).
        :param name: folder name inside the layer.
        :param mode: STATIC (static subdir) or DATE (dated subdir).
        :param date_from: lower bound "YYYY-MM-DD", inclusive, matched on file stem.
        :param date_to: upper bound "YYYY-MM-DD", inclusive.
        :param pattern: glob for files inside the folder (default "*.parquet").
        :param missing_ok: if True, return an empty df instead of raising StopPipeline.
    """

    def __init__(self, layer, name, mode=Mode.STATIC, date_from=None, date_to=None,
                 pattern="*.parquet", missing_ok=False):
        self.layer, self.name, self.mode = layer, name, mode
        self.date_from, self.date_to = date_from, date_to
        self.pattern, self.missing_ok = pattern, missing_ok

    async def apply(self, df, data = None):
        folder = path(self.layer, self.name, mode=self.mode)
        files = sorted(folder.glob(self.pattern)) if folder.exists() else []

        if self.date_from:
            files = [f for f in files if f.stem >= self.date_from]
        
        if self.date_to:
            files = [f for f in files if f.stem <= self.date_to]

        if not files:
            if self.missing_ok:
                print(f" No files: {folder}")
                return pl.DataFrame()
            raise StopPipeline(message=f"No files found: {folder}")
        
        return await asyncio.to_thread(
            lambda: pl.concat([pl.read_parquet(file) for file in files], how="vertical_relaxed")
            )

    def __repr__(self):
        return f"Reads(name={self.name!r}, from={self.date_from}, to={self.date_to})"