import polars as pl
from etl.generic import Data
from etl.loader.steps.mssql import EnsureTable, Insert


class _FakeCursor:
    def __init__(self): self.executed, self.many = None, None
    async def execute(self, sql, *p): self.executed = sql
    async def executemany(self, sql, rows): self.many = (sql, list(rows))
    async def __aenter__(self): return self
    async def __aexit__(self, *a): pass

class _FakeMSSQL:
    def __init__(self): self.cur = _FakeCursor()
    def cursor(self): return self.cur


async def test_mssql_ensuretable_ddl():
    db = _FakeMSSQL()
    df = pl.DataFrame({"pk": [1], "amount": [1.5], "name": ["x"]})
    await EnsureTable("t", keys=["pk"]).apply(df, Data(mssql=db))
    ddl = db.cur.executed
    assert "IF OBJECT_ID(N't', N'U') IS NULL" in ddl
    assert "[pk] BIGINT NOT NULL" in ddl
    assert "[amount] FLOAT" in ddl
    assert "UNIQUE ([pk])" in ddl

async def test_mssql_insert_merge():
    db = _FakeMSSQL()
    await Insert("t", keys=["pk"]).apply(pl.DataFrame({"pk": [1], "v": [2]}), Data(mssql=db))
    sql, rows = db.cur.many
    assert "MERGE INTO t WITH (HOLDLOCK)" in sql
    assert "WHEN MATCHED THEN UPDATE SET" in sql
    assert "WHEN NOT MATCHED THEN INSERT" in sql
    assert sql.rstrip().endswith(";")
    assert rows == [(1, 2)]

async def test_mssql_insert_only_keys_no_update():
    db = _FakeMSSQL()
    await Insert("t", keys=["pk"]).apply(pl.DataFrame({"pk": [1]}), Data(mssql=db))
    assert "WHEN MATCHED" not in db.cur.many[0]

async def test_mssql_insert_skips_empty():
    db = _FakeMSSQL()
    await Insert("t", keys=["pk"]).apply(pl.DataFrame(), Data(mssql=db))
    assert db.cur.many is None

def test_mssql_loader_repr():
    assert repr(EnsureTable("t", keys=["pk"]))
    assert repr(Insert("t", keys=["pk"]))