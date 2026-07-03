import polars as pl, pyarrow as pa, pytest
from etl.loader import Storage
from etl.generic import Data, StopPipeline
from etl.loader.steps.datalake import Archive

L = Storage.Layer
M = Storage.Mode

def _df():
    return pl.DataFrame({"id": [1, 2], "name": ["a", "b"]})


@pytest.fixture(autouse=True)
def lake(tmp_path, monkeypatch):
    monkeypatch.setenv("LAKE_DATA_DIR", str(tmp_path))
    return tmp_path

def test_layer_dir_from_base(lake):
    assert Storage.layer_dir(L.RAW) == lake / "raw"

def test_layer_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("RAW_DATA_DIR", str(tmp_path / "custom"))
    assert Storage.layer_dir(L.RAW) == tmp_path / "custom"

def test_layer_dir_unknown_raises():
    with pytest.raises(ValueError):
        Storage.layer_dir("nope")

def test_write_read_roundtrip():
    Storage.write(L.RAW, _df(), "bank.parquet")
    assert Storage.read(L.RAW, "bank.parquet").equals(_df())

def test_read_missing_raises():
    with pytest.raises(FileNotFoundError):
        Storage.read(L.RAW, "missing.parquet")

def test_read_as_arrow():
    Storage.write(L.RAW, _df(), "bank.parquet")
    assert isinstance(Storage.read(L.RAW, "bank.parquet", as_arrow=True), pa.Table)

def test_static_mode_roundtrip():
    Storage.write(L.REF, _df(), "rates.parquet", mode=M.STATIC)
    assert Storage.read(L.REF, "rates.parquet", mode=M.STATIC).equals(_df())

def test_write_archives_previous(lake):
    Storage.write(L.RAW, _df(), "bank.parquet")
    Storage.write(L.RAW, pl.DataFrame({"id": [9], "name": ["z"]}), "bank.parquet")  # overwrite=False
    assert Storage.read(L.RAW, "bank.parquet")["id"].to_list() == [9]               # new wins
    assert list((lake / "archive").rglob("*.parquet"))                              # old archived

def test_read_all():
    Storage.write(L.RAW, _df(), "a.parquet")
    Storage.write(L.RAW, pl.DataFrame({"x": [9]}), "b.parquet")
    got = Storage.read_all(L.RAW)
    assert set(got) == {"a", "b"} and got["a"].equals(_df())

def test_read_all_empty_layer():
    assert Storage.read_all(L.STG) == {}

def test_list_files():
    Storage.write(L.RAW, _df(), "a.parquet")
    Storage.write(L.RAW, _df(), "b.parquet")
    assert sorted(p.name for p in Storage.list_files(L.RAW)) == ["a.parquet", "b.parquet"]

def test_list_files_all_dates():
    Storage.write(L.RAW, _df(), "today.parquet")
    Storage.write(L.RAW, _df(), "old.parquet", date="2020-01-01")
    assert len(Storage.list_files(L.RAW, date="*")) == 2

def test_list_files_missing_folder():
    assert Storage.list_files(L.STG) == []

def test_list_dates():
    Storage.write(L.RAW, _df(), "a.parquet", date="2020-01-01")
    Storage.write(L.RAW, _df(), "b.parquet", date="2020-01-02")
    assert Storage.list_dates(L.RAW) == ["2020-01-01", "2020-01-02"]

def test_archive_file():
    Storage.write(L.RAW, _df(), "bank.parquet")
    dest = Storage.archive_file(L.RAW, "bank.parquet")
    assert dest is not None and dest.exists()
    with pytest.raises(FileNotFoundError):
        Storage.read(L.RAW, "bank.parquet")          # source moved away

def test_archive_file_missing_returns_none():
    assert Storage.archive_file(L.RAW, "nope.parquet") is None

def test_archive_layer():
    Storage.write(L.RAW, _df(), "a.parquet")
    Storage.write(L.RAW, _df(), "b.parquet")
    assert len(Storage.archive_layer(L.RAW)) == 2
    assert Storage.list_files(L.RAW) == []

def test_cleanup_dry_run_keeps(lake):
    Storage.write(L.RAW, _df(), "old.parquet", date="2000-01-01")
    assert len(Storage.cleanup(L.RAW, older_than_days=30, dry_run=True)) == 1
    assert (lake / "raw" / "2000-01-01").exists()        # dry run -> not deleted

def test_cleanup_deletes_old(lake):
    Storage.write(L.RAW, _df(), "old.parquet", date="2000-01-01")
    Storage.cleanup(L.RAW, older_than_days=30, dry_run=False)
    assert not (lake / "raw" / "2000-01-01").exists()

def test_enum_str():
    assert str(L.RAW) == "raw" and str(M.DATE) == "date"

def test_list_files_star_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("LAKE_DATA_DIR", str(tmp_path))
    assert Storage.list_files(Storage.Layer.STG, date="*") == []

def test_list_dates_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("LAKE_DATA_DIR", str(tmp_path))
    assert Storage.list_dates(Storage.Layer.STG) == []

async def test_archive_passes_df_through(tmp_path, monkeypatch):
    monkeypatch.setenv("LAKE_DATA_DIR", str(tmp_path))
    Storage.write(Storage.Layer.RAW, pl.DataFrame({"a": [1]}), "x.parquet")
    df = pl.DataFrame({"a": [1]})
    out = await Archive(layer=Storage.Layer.RAW, name="x").apply(df, Data())
    assert out.equals(df)

# archive.py:27-29: FileNotFoundError + missing_ok -> df
async def test_archive_missing_ok(monkeypatch):
    def boom(*a, **k): raise FileNotFoundError
    monkeypatch.setattr("etl.loader.steps.datalake.archive.archive_file", boom)
    df = pl.DataFrame({"a": [1]})
    out = await Archive(layer="raw", name="x", missing_ok=True).apply(df, Data())
    assert out.equals(df)

# archive.py:30: FileNotFoundError + not missing_ok -> StopPipeline
async def test_archive_missing_stops(monkeypatch):
    def boom(*a, **k): raise FileNotFoundError
    monkeypatch.setattr("etl.loader.steps.datalake.archive.archive_file", boom)
    with pytest.raises(StopPipeline):
        await Archive(layer="raw", name="x", missing_ok=False).apply(pl.DataFrame({"a": [1]}), Data())

# storage.py:181-182: rmdir catch OSError (concurrent dags)
def test_archive_file_rmdir_race(tmp_path, monkeypatch):
    from pathlib import Path
    monkeypatch.setenv("LAKE_DATA_DIR", str(tmp_path))
    Storage.write(Storage.Layer.RAW, pl.DataFrame({"a": [1]}), "x.parquet")
    monkeypatch.setattr(Path, "rmdir", lambda self: (_ for _ in ()).throw(OSError("race")))
    Storage.archive_file(Storage.Layer.RAW, "x.parquet")   
