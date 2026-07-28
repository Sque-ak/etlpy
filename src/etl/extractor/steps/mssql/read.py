import polars as pl
from etl.generic.step import Step


class Read(Step):
    """
    Run a SQL query against SQL Server and make the result the pipeline df.

    Reads the connection from data["mssql"] (set by Connect). Incoming df ignored.

        :param query: T-SQL SELECT with ? placeholders for parameters.
        :param parameters: values bound to the placeholders (injection-safe).

        Example:
        Read(
            query="SELECT * FROM transactions WHERE txn_date >= ? AND bank = ?",
            parameters=["2026-06-17", "acme"],
        )
    """

    def __init__(self, query: str, parameters: list | None = None):
        self.query = query
        self.parameters = parameters or []

    async def apply(self, df=None, data=None) -> pl.DataFrame:
        async with data["mssql"].cursor() as cur:
            await cur.execute(self.query, *self.parameters)
            rows = await cur.fetchall()
            cols = [d[0] for d in cur.description]
        return pl.DataFrame([tuple(r) for r in rows], schema=cols, orient="row", infer_schema_length=None)

    def __repr__(self):
        return f"Read(query={self.query!r}, parameters={self.parameters})"
