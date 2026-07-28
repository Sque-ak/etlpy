import polars as pl
import pytest
from etl.generic import Data
from etl.extractor.steps.postgresql import Connect, Read


class _FakePG:
    def __init__(self, rows): self.rows, self.last = rows, None
    async def fetch(self, query, *params):
        self.last = (query, params)
        return self.rows


async def test_pg_read_returns_dataframe():
    out = await Read("SELECT a FROM t").apply(None, Data(pg=_FakePG([{"a": 1}, {"a": 2}])))
    assert out["a"].to_list() == [1, 2]

async def test_pg_read_binds_parameters():
    pg = _FakePG([])
    await Read("SELECT * FROM t WHERE x = $1", parameters=[5]).apply(None, Data(pg=pg))
    assert pg.last == ("SELECT * FROM t WHERE x = $1", (5,))    

async def test_pg_connect(monkeypatch):
    fake = object()
    async def fake_connect(**cfg): return fake
    monkeypatch.setattr("asyncpg.connect", fake_connect)
    data = Data()
    await Connect(host="h").apply(None, data)
    assert data["pg"] is fake

def test_pg_repr():
    assert repr(Connect(host="h", database="d"))
    assert repr(Read("SELECT 1"))
