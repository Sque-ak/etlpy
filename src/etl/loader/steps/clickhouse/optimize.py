from etl.generic import Step
import asyncio

class Optimize(Step):
    """
    Force a merge on a ClickHouse table (OPTIMIZE TABLE ... FINAL) to collapse
    ReplacingMergeTree duplicates. Reads the client from data["ch"].

    Expensive on large tables - run after a load or on a schedule, not per row.

        :param table: target table.
    """

    def __init__(self, table:str):
        self.table = table

    async def apply(self, df, data = None):
        await asyncio.to_thread(data["ch"].command, f"OPTIMIZE TABLE {self.table} FINAL")
        return df
    
    def __repr__(self):
        return f"Optimize(table={self.table!r})"