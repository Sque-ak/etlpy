import asyncio
import polars as pl

from etl.generic.step import Step


class Delta(Step):
    """
    Keep only new or changed rows by comparing 'row_hash' against ClickHouse.

    A row is kept when its keys is absent (new) or its row_hash differs.
    Reads the client from data["ch"]; needs a 'row_hash' column (RowHash in transformer).

        :param table: target table
        :param keys: key column(s) identifying a row across loads
    """

    def __init__(self, table: str, keys: list[str] | str):
        self.table = table
        self.keys = [keys] if isinstance(keys, str) else keys

    async def apply(self, df: pl.DataFrame, data=None):
        ch = data["ch"]
        if df is None or df.is_empty():
            return df
        if not await asyncio.to_thread(ch.command, f"EXISTS TABLE {self.table}"):
            return df # new table all rows are new
        
    
        existing = await self._existing_hashes(ch)
        if existing.is_empty():
            return df
        
        existing = existing.cast({key: df.schema[key] for key in self.keys}) # match key dtypes
        delta = (
            df.join(existing, on=self.keys, how="left", suffix="_old")
            .filter(pl.col("row_hash_old").is_null() | (pl.col("row_hash") != pl.col("row_hash_old")))
            .drop("row_hash_old")
        )

        if (skipped := df.height - delta.height) > 0:
            print(f"Delta: skipped {skipped} unchanged rows for {self.table}")
        
        return delta

    async def _existing_hashes(self, ch) -> pl.DataFrame:
        keys = ", ".join(f"`{key}`" for key in self.keys)
        sql = f"SELECT {keys}, `row_hash` FROM {self.table} FINAL"
        return pl.from_arrow(await asyncio.to_thread(ch.query_arrow, sql))
        
    def __repr__(self) -> str:
        return f"Delta(table={self.table!r}, keys={self.keys!r})"