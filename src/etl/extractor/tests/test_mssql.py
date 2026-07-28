import polars as pl
from etl.generic import Data
from etl.extractor.steps.mssql import Connect, Read


class _FakeCursor:
    def __init__(self, rows, description):
        self._rows, self.description, self.executed = rows, description, None
    async def execute(self, sql, *params): self.executed = (sql, params)
    async def fetchall(self): return self._rows
    async def __aenter__(self): return self
    async def __aexit__(self, *a): pass

class _FakeMSSQL:
    def __init__(self, rows, description):
        self._rows, self._desc, self.last_cursor = rows, description, None
    def cursor(self):
        self.last_cursor = _FakeCursor(self._rows, self._desc)
        return self.last_cursor


async def test_mssql_read_returns_dataframe():
    db = _FakeMSSQL(rows=[(1, "x"), (2, "y")], description=[("a",), ("b",)])
    out = await Read("SELECT a, b FROM t").apply(None, Data(mssql=db))
    assert out.columns == ["a", "b"]
    assert out["a"].to_list() == [1, 2]

async def test_mssql_read_binds_parameters():
    db = _FakeMSSQL(rows=[], description=[("a",)])
    await Read("SELECT * FROM t WHERE x = ?", parameters=[5]).apply(None, Data(mssql=db))
    assert db.last_cursor.executed == ("SELECT * FROM t WHERE x = ?", (5,))

async def test_mssql_connect(monkeypatch):
    fake = object()
    async def fake_connect(dsn=None, **kw): return fake
    monkeypatch.setattr("aioodbc.connect", fake_connect)
    data = Data()
    await Connect(host="h", database="d").apply(None, data)
    assert data["mssql"] is fake

def test_mssql_repr():
    assert repr(Connect(host="h", database="d"))
    assert repr(Read("SELECT 1"))
