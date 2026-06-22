import asyncio, polars as pl
from etl.generic.step import Step

class Read(Step):
    '''
    Run a SQL query against ClickHouse and make the result the pipeline df.

    Source step: reads the client from data["ch"] (set by ConnectClickHouse),
    runs the query, returns a Polars DataFrame. The incoming df is ignored.

        :param query: SQL SELECT to execute.
        :param parameters: SQL parameters

        Example:
        Read(
            query="""
                SELECT *
                FROM transactions
                WHERE txn_date >= {start:Date}
                AND bank = {bank:String}
            """,
            parameters={"start": "2026-06-17", "bank": "acme"},
        )
    '''

    def __init__(self, query: str, parameters: dict | None = None):
        self.query, self.parameters = query, parameters

    async def apply(self, df=None, data=None) -> pl.DataFrame:
        table = await asyncio.to_thread(data["ch"].query_arrow, self.query, parameters=self.parameters)
        return pl.from_arrow(table)
    
    def __repr__(self):
        return f"Read(query={self.query!r}, parameters={self.parameters})"