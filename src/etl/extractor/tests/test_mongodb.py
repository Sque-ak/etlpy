import polars as pl
import pytest
from etl.generic import Data
from etl.extractor.steps.mongodb import Connect, Read
from etl.loader.steps.mongodb import Upsert

class _FakeCursor:
    def __init__(self, docs): self._docs = docs
    def sort(self, *a, **k): return self
    def limit(self, n): self._docs = self._docs[:n]; return self
    async def to_list(self): return self._docs

class _FakeCollection:
    def __init__(self, docs=None):
        self.docs, self.indexes, self.written = docs or [], [], None
    def find(self, filter=None, projection=None): return _FakeCursor(list(self.docs))
    async def create_index(self, keys, **kw): self.indexes.append((keys, kw))
    async def bulk_write(self, ops, ordered=True): self.written = ops
    def aggregate(self, pipeline, **kw):         
        self.agg_pipeline = pipeline
        return _FakeCursor(list(self.docs))

class _FakeDB:
    def __init__(self, coll=None): self.coll = coll or _FakeCollection()
    def __getitem__(self, name): return self.coll


async def test_connect_puts_db_handle(monkeypatch):
    fake_db = _FakeDB()
    class _FakeClient:
        def __init__(self, **cfg): self.cfg = cfg
        def __getitem__(self, name): return fake_db
    monkeypatch.setattr("pymongo.AsyncMongoClient", _FakeClient)
    data = Data()
    await Connect(host="h", database="analytics").apply(None, data)
    assert data["mongo"] is fake_db

async def test_read_returns_dataframe():
    data = Data(mongo=_FakeDB(_FakeCollection(docs=[{"a": 1}, {"a": 2}])))
    out = await Read("c").apply(None, data)
    assert out["a"].to_list() == [1, 2]

def test_read_default_projection_drops_id():
    assert Read("c").projection == {"_id": 0}         

async def test_read_sort_and_limit():
    data = Data(mongo=_FakeDB(_FakeCollection(docs=[{"a": 1}, {"a": 2}, {"a": 3}])))
    out = await Read("c", sort=[("a", 1)], limit=2).apply(None, data)
    assert out.height == 2                              

def test_repr_mongo_extractor():
    assert repr(Connect(host="x", database="y"))
    assert repr(Read("c"))

def aggregate(self, pipeline, **kw):
    self.agg_pipeline = pipeline
    return _FakeCursor(list(self.docs))

async def test_mongo_read_aggregate():
    coll = _FakeCollection(docs=[{"id": "abc", "amount": 10}])
    out = await Read("txn", pipeline=[{"$lookup": {}}]).apply(None, Data(mongo=_FakeDB(coll)))
    assert coll.agg_pipeline == [{"$lookup": {}}]      # aggregate вызван с пайплайном
    assert out["id"].to_list() == ["abc"]