import polars as pl
import pytest
from etl.generic import Data
from etl.loader.steps.postgresql import EnsureTable, Insert


class _FakePG:
    def __init__(self): self.executed, self.many = [], None
    async def execute(self, sql): self.executed.append(sql)
    async def executemany(self, sql, args): self.many = (sql, list(args))


async def test_pg_ensuretable_ddl():
    pg = _FakePG()
    df = pl.DataFrame({"pk": [1], "amount": [1.5], "name": ["x"]})
    await EnsureTable("t", keys=["pk"]).apply(df, Data(pg=pg))
    ddl = pg.executed[0]
    assert "CREATE TABLE IF NOT EXISTS t" in ddl
    assert '"pk" BIGINT NOT NULL' in ddl
    assert '"amount" DOUBLE PRECISION' in ddl
    assert "UNIQUE" in ddl

async def test_pg_insert_upsert():
    pg = _FakePG()
    await Insert("t", keys=["pk"]).apply(pl.DataFrame({"pk": [1], "v": [2]}), Data(pg=pg))
    sql, args = pg.many
    assert "ON CONFLICT" in sql and "DO UPDATE SET" in sql
    assert args == [(1, 2)]

async def test_pg_insert_do_nothing_all_keys():
    pg = _FakePG()
    await Insert("t", keys=["pk"]).apply(pl.DataFrame({"pk": [1]}), Data(pg=pg))
    assert "DO NOTHING" in pg.many[0]

async def test_pg_insert_skips_empty():
    pg = _FakePG()
    await Insert("t", keys=["pk"]).apply(pl.DataFrame(), Data(pg=pg))
    assert pg.many is None

def test_pg_loader_repr():
    assert repr(EnsureTable("t", keys=["pk"]))
    assert repr(Insert("t", keys=["pk"]))
