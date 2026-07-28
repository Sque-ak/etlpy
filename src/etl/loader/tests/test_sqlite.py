import polars as pl
from etl.generic import Data
from etl.extractor.steps.sqlite import Connect, Read
from etl.loader.steps.sqlite import EnsureTable, Insert


async def test_sqlite_roundtrip():
    data = Data()
    df = pl.DataFrame({"pk": [1, 2], "v": ["a", "b"]})
    await Connect(":memory:").apply(None, data)
    await EnsureTable("t", keys=["pk"]).apply(df, data)
    await Insert("t", keys=["pk"]).apply(df, data)
    out = await Read('SELECT pk, v FROM "t" ORDER BY pk').apply(None, data)
    assert out["pk"].to_list() == [1, 2]
    assert out["v"].to_list() == ["a", "b"]

async def test_sqlite_upsert_idempotent():
    data = Data()
    await Connect(":memory:").apply(None, data)
    df = pl.DataFrame({"pk": [1], "v": ["a"]})
    await EnsureTable("t", keys=["pk"]).apply(df, data)
    await Insert("t", keys=["pk"]).apply(df, data)
    await Insert("t", keys=["pk"]).apply(df.with_columns(pl.lit("b").alias("v")), data)  
    out = await Read('SELECT pk, v FROM "t"').apply(None, data)
    assert out.height == 1                
    assert out["v"].to_list() == ["b"]     

async def test_sqlite_insert_skips_empty():
    out = await Insert("t", keys=["pk"]).apply(pl.DataFrame(), Data())
    assert out.is_empty()        

def test_sqlite_repr():
    assert repr(Connect(":memory:"))
    assert repr(Read("SELECT 1"))
    assert repr(EnsureTable("t", keys=["pk"]))
    assert repr(Insert("t", keys=["pk"]))
