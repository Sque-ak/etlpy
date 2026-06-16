import asyncio
from etl.generic.step import Step
from polars import DataFrame

class Insert(Step):
    """
    Insert the incoming DataFrame into a ClickHouse table via Arrow.

    Reads the ClickHouse client from data["ch"] (put there by ConnectClickHouse).
    
        :param table: target ClickHouse table name.
    """
    
    def __init__(self, table: str):
        self.table = table

    async def apply(self, df: DataFrame, data = None):
        if df is None or df.is_empty():
            return df
        
        await asyncio.to_thread(data["ch"].insert_arrow, self.table, df.to_arrow())
        return df
    
    def __repr__(self):
        return f"Insert(table={self.table!r})"