import asyncio
from etl.generic.step import Step
from polars import DataFrame


class Connect(Step):
    """
    Open a ClickHouse connection and put the client into data["ch"] for
    downstream steps (EnsureTable, Delta, Insert).

    Usually the first step of a load pipeline.

        :param host/port/database/username/password/secure: clickhouse-connect params.
        :param kwargs: extra arguments forwarded to clickhouse_connect.get_client.
    """

    def __init__(self, host="localhost", port=8123, database="default",
                 username="default", password="", secure=False, **kwargs):
        self.config = dict(host=host, port=port, database=database,
                           username=username, password=password, secure=secure, **kwargs)

    async def apply(self, df: DataFrame, data=None):
        import clickhouse_connect
        data["ch"] = await asyncio.to_thread(clickhouse_connect.get_client, **self.config)
        return df
    
    def __repr__(self):
        return f"ConnectClickHouse(host={self.config['host']!r}, database={self.config['database']!r})"