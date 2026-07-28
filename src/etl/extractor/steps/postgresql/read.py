import polars as pl
from etl.generic.step import Step


class Read(Step):
    """
    Run a SQL query against PostgreSQL and make the result the pipeline df.

    Source step: reads the connection from data["pg"] (set by Connect),
    runs the query, returns a Polars DataFrame. The incoming df is ignored.

        :param query: SQL SELECT with $1, $2 placeholders.
        :param parameters: values bound to the placeholders (injection-safe).

        Example:
        Read(
            query="SELECT * FROM transactions WHERE txn_date >= $1 AND bank = $2",
            parameters=["2026-06-17", "acme"],
        )
    """

    def __init__(self, query: str, parameters: list | None = None):
        self.query = query
        self.parameters = parameters or []

    async def apply(self, df=None, data=None) -> pl.DataFrame:
        rows = await data["pg"].fetch(self.query, *self.parameters)
        return pl.DataFrame([dict(r) for r in rows], infer_schema_length=None)

    def __repr__(self):
        return f"Read(query={self.query!r}, parameters={self.parameters})"
