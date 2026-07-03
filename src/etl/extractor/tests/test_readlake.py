import polars as pl
import pytest

from etl.generic import Data, Pipeline
from etl.loader import Storage
from etl.extractor.steps.datalake import Read

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


