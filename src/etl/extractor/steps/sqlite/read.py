import polars as pl
from etl.generic.step import Step


class Read(Step):
    """
    Run a SQL query against SQLite and make the result the pipeline df.
    Reads the connection from data["sqlite"] (set by Connect).

        :param query: SQL SELECT with ? placeholders.
        :param parameters: values bound to the placeholders (injection-safe).
    """

    def __init__(self, query: str, parameters: list | None = None):
        self.query = query
        self.parameters = parameters or []

    async def apply(self, df=None, data=None) -> pl.DataFrame:
        async with data["sqlite"].execute(self.query, self.parameters) as cursor:
            rows = await cursor.fetchall()
            cols = [d[0] for d in cursor.description]
        return pl.DataFrame([tuple(r) for r in rows], schema=cols, orient="row", infer_schema_length=None)

    def __repr__(self):
        return f"Read(query={self.query!r}, parameters={self.parameters})"