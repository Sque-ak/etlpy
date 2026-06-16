import asyncio, polars as pl
from etl.generic.step import Step

class Read(Step):
    """
    Run a SQL query against ClickHouse and make the result the pipeline df.

    Source step: reads the client from data["ch"] (set by ConnectClickHouse),
    runs the query, returns a Polars DataFrame. The incoming df is ignored.

        :param query: SQL SELECT to execute.
    """

    def __init__(self, query: str):
        self.query = query

    async def apply(self, df=None, data=None) -> pl.DataFrame:
        table = await asyncio.to_thread(data["ch"].query_arrow, self.query)
        return pl.from_arrow(table)
    
    def __repr__(self):
        return f"Read(query={self.query!r})"