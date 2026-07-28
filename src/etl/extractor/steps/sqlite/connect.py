from etl.generic.step import Step
from polars import DataFrame


class Connect(Step):
    """
    Open a SQLite connection and put it into data["sqlite"].

    WARNING: SQLite is meant for tests and small local datasets. It is a
    single-file, single-writer engine and is NOT suitable for storing large
    data - use ClickHouse or Postgres for that. Great as an in-memory test
    backend (path=":memory:").

        :param path: database file path, or ":memory:" for an in-memory DB.
        :param kwargs: extra args forwarded to aiosqlite.connect.
    """

    def __init__(self, path: str = ":memory:", **kwargs):
        self.path, self.kwargs = path, kwargs

    async def apply(self, df: DataFrame = None, data=None):
        import aiosqlite
        data["sqlite"] = await aiosqlite.connect(self.path, **self.kwargs)
        return df

    def __repr__(self):
        return f"ConnectSQLite(path={self.path!r})"