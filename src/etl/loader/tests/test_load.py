import pytest
from etl.generic import Pipeline, Data
from etl.loader import Storage
from etl.loader.steps.datalake import Save, Archive
from etl.loader.steps.clickhouse import Connect, EnsureTable, Delta, Insert
from etl.extractor.steps.datalake import Read
from etl.transformer.steps import GenerateKey, RowHash
from polars import DataFrame
import polars as pl


@pytest.mark.integration
async def test_load_idempotent(tmp_path, monkeypatch, ch_conn):
    monkeypatch.setenv("LAKE_DATA_DIR", str(tmp_path))

    df = DataFrame({"id": [1, 2, 3], "amount": [100, 200, 300]})
    await Pipeline([
        GenerateKey(columns=["id"], key_name="pk"),
        RowHash(),
        Save(name="tx", layer="fact"),
    ], dataframe=df).run()

    def load():
        return Pipeline([
            Connect(**ch_conn),
            Read(layer="fact", name="tx"),
            EnsureTable("fact_tx", engine="ReplacingMergeTree", order_by=["pk"]),
            Delta("fact_tx", keys=["pk"]),       # only new or changed
            Insert("fact_tx"),
        ])

    await load().run()       # first: insert 3
    await load().run()       # second: the same row_hash to Delta return 0 and insert 0

    import clickhouse_connect
    client = clickhouse_connect.get_client(**ch_conn)
    assert client.command("SELECT count() FROM fact_tx") == 3 # not 6

@pytest.mark.integration
async def test_delta_raises_on_mergetree(tmp_path, monkeypatch, ch_conn):
    """Delta uses FINAL; on a plain MergeTree it must raise loudly, not silently pass."""
    monkeypatch.setenv("LAKE_DATA_DIR", str(tmp_path))

    df = DataFrame({"id": [1, 2, 3], "amount": [100, 200, 300]})
    await Pipeline([
        GenerateKey(columns=["id"], key_name="pk"),
        RowHash(),
        Save(name="tx_mt", layer="fact"),
    ], dataframe=df).run()

    load = Pipeline([
        Connect(**ch_conn),
        Read(layer="fact", name="tx_mt"),
        EnsureTable("fact_tx_mt", engine="MergeTree", order_by=["pk"]), 
        Delta("fact_tx_mt", keys=["pk"]),
        Insert("fact_tx_mt"),
    ])

    with pytest.raises(Exception, match="FINAL"):
        await load.run()

def test_ensuretable_ddl_types():
    df = pl.DataFrame(schema={
        "i": pl.Int32, "u": pl.UInt8, "f": pl.Float64, "b": pl.Boolean,
        "s": pl.String, "d": pl.Date, "t": pl.Datetime,
        "lst": pl.List(pl.Int64), "dec": pl.Decimal(10, 2), "pk": pl.Int64,
    })
    ddl = EnsureTable("tbl", order_by=["pk"])._build_ddl(df.to_arrow().schema)
    for t in ("Int32", "UInt8", "Float64", "Bool", "Date", "DateTime64", "Array", "Decimal", "Nullable"):
        assert t in ddl
    assert "CREATE TABLE tbl" in ddl


class _FakeCH:
    def __init__(self, exists=1): self.exists = exists
    def command(self, sql): return self.exists

async def test_delta_empty_df():
    out = await Delta("t", keys=["pk"]).apply(pl.DataFrame(), Data(ch=_FakeCH()))
    assert out.is_empty()                                   # empty df -> early return

async def test_delta_table_absent():
    df = pl.DataFrame({"pk": [1], "row_hash": ["x"]})
    out = await Delta("t", keys=["pk"]).apply(df, Data(ch=_FakeCH(exists=0)))
    assert out.height == 1                                  # no table -> all new

async def test_save_skips_none():
    assert await Save(name="x").apply(None, Data()) is None

async def test_archive_step(tmp_path, monkeypatch):
    monkeypatch.setenv("LAKE_DATA_DIR", str(tmp_path))
    Storage.write(Storage.Layer.RAW, pl.DataFrame({"a": [1]}), "x.parquet")
    await Archive(layer=Storage.Layer.RAW, name="x").apply(pl.DataFrame({"a": [1]}), Data())
    with pytest.raises(FileNotFoundError):
        Storage.read(Storage.Layer.RAW, "x.parquet")        # archived away

async def test_ensuretable_error_if_exists():
    step = EnsureTable("t", order_by=["pk"], if_exists="error")
    with pytest.raises(ValueError):
        await step.apply(pl.DataFrame({"pk": [1]}), Data(ch=_FakeCH(exists=1)))

def test_ensuretable_replacing_version():
    df = pl.DataFrame(schema={"v": pl.Int64, "pk": pl.Int64})
    ddl = EnsureTable("t", order_by=["pk"], engine="ReplacingMergeTree(v)")._build_ddl(df.to_arrow().schema)
    assert "`v` Int64" in ddl and "Nullable(Int64)" not in ddl   # v is non_nullable

def test_ensuretable_fallback_type():
    df = pl.DataFrame(schema={"tm": pl.Time, "pk": pl.Int64})
    ddl = EnsureTable("t", order_by=["pk"])._build_ddl(df.to_arrow().schema)
    assert "`tm` String" in ddl