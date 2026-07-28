from etl.generic.step import Step
from polars import DataFrame


class Connect(Step):
    """
    Open a PostgreSQL connection and put it into data["pg"] for downstream Read.

        :param host/port/database/user/password: asyncpg connection params.
        :param kwargs: extra args forwarded to asyncpg.connect (ssl, timeout, ...).
    """

    def __init__(self, host="localhost", port=5432, database="postgres",
                 user="postgres", password="", **kwargs):
        self.config = dict(host=host, port=port, database=database,
                           user=user, password=password, **kwargs)

    async def apply(self, df: DataFrame = None, data=None):
        import asyncpg
        data["pg"] = await asyncpg.connect(**self.config)
        return df

    def __repr__(self):
        return f"ConnectPostgres(host={self.config['host']!r}, database={self.config['database']!r})"
