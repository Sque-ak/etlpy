import polars as pl
import pytest

from etl.generic import Data, Pipeline, StopPipeline
from etl.loader import Storage
from etl.extractor.steps.datalake import Read, Reads

async def test_read_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("LAKE_DATA_DIR", str(tmp_path)) # temp dir
    df = pl.DataFrame({"id": [1,2], "name":["a", "b"]})
    Storage.write(Storage.Layer.RAW, df, "bank.parquet") 

    out = await Pipeline(
        [
            Read(layer=Storage.Layer.RAW, name="bank")
        ]
    ).run()

    assert out.equals(df)

async def test_read_missing_file_stops(tmp_path, monkeypatch):
    monkeypatch.setenv("LAKE_DATA_DIR", str(tmp_path))
    out = await Pipeline([Read(layer=Storage.Layer.RAW, name="nope")]).run()
    assert out is None            

async def test_read_missing_ok_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("LAKE_DATA_DIR", str(tmp_path))
    out = await Read(layer=Storage.Layer.RAW, name="nope", missing_ok=True).apply(None, Data())
    assert out.is_empty()

async def test_reads_concats_and_filters(tmp_path, monkeypatch):
    monkeypatch.setenv("STG_DATA_DIR", str(tmp_path))
    folder = tmp_path / "static" / "manual"
    folder.mkdir(parents=True)
    pl.DataFrame({"a": [1]}).write_parquet(folder / "2026-06-01.parquet")
    pl.DataFrame({"a": [2]}).write_parquet(folder / "2026-06-15.parquet")
    out = await Reads(layer="stg", name="manual", mode=Storage.Mode.STATIC,
                      date_from="2026-06-10").apply(None, Data())
    assert out["a"].to_list() == [2]            
    
async def test_reads_missing_ok(tmp_path, monkeypatch):
    monkeypatch.setenv("STG_DATA_DIR", str(tmp_path))
    out = await Reads(layer="stg", name="nope", mode=Storage.Mode.STATIC, missing_ok=True).apply(None, Data())
    assert out.is_empty()

async def test_reads_missing_stops(tmp_path, monkeypatch):
    monkeypatch.setenv("STG_DATA_DIR", str(tmp_path))
    with pytest.raises(StopPipeline):
        await Reads(layer="stg", name="nope", mode=Storage.Mode.STATIC).apply(None, Data())

async def test_reads_date_to(tmp_path, monkeypatch):
    monkeypatch.setenv("STG_DATA_DIR", str(tmp_path))
    folder = tmp_path / "static" / "manual"
    folder.mkdir(parents=True)
    pl.DataFrame({"a": [1]}).write_parquet(folder / "2026-06-01.parquet")
    pl.DataFrame({"a": [2]}).write_parquet(folder / "2026-06-15.parquet")
    out = await Reads(layer="stg", name="manual", mode=Storage.Mode.STATIC,
                      date_to="2026-06-10").apply(None, Data())
    assert out["a"].to_list() == [1]       

def test_reads_repr():
    assert repr(Reads(layer="stg", name="manual"))    
