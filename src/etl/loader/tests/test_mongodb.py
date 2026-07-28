import polars as pl
import pytest
from pymongo import ReplaceOne, UpdateOne
from etl.generic import Data
from etl.loader.steps.mongodb import EnsureIndexes, Upsert, Delta


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

class _FakeDB:
    def __init__(self, coll=None): self.coll = coll or _FakeCollection()
    def __getitem__(self, name): return self.coll


async def test_ensuretable_creates_indexes():
    coll = _FakeCollection()
    await EnsureIndexes("c", keys=["pk"], indexes=["row_hash"]).apply(None, Data(mongo=_FakeDB(coll)))
    assert ([("pk", 1)], {"unique": True}) in coll.indexes
    assert ("row_hash", {}) in coll.indexes

async def test_ensuretable_string_key():
    coll = _FakeCollection()
    await EnsureIndexes("c", keys="pk").apply(None, Data(mongo=_FakeDB(coll)))   # str -> [str]
    assert ([("pk", 1)], {"unique": True}) in coll.indexes

async def test_upsert_replace():
    coll = _FakeCollection()
    await Upsert("c", keys=["pk"]).apply(pl.DataFrame({"pk": [1], "v": [2]}), Data(mongo=_FakeDB(coll)))
    assert isinstance(coll.written[0], ReplaceOne)

async def test_upsert_update():
    coll = _FakeCollection()
    await Upsert("c", keys=["pk"], mode="update").apply(pl.DataFrame({"pk": [1], "v": [2]}), Data(mongo=_FakeDB(coll)))
    assert isinstance(coll.written[0], UpdateOne)

def test_upsert_bad_mode():
    with pytest.raises(ValueError):
        Upsert("c", keys=["pk"], mode="nope")

async def test_upsert_skips_empty():
    coll = _FakeCollection()
    await Upsert("c", keys=["pk"]).apply(pl.DataFrame(), Data(mongo=_FakeDB(coll)))
    assert coll.written is None

async def test_delta_empty_df():
    out = await Delta("c", keys=["pk"]).apply(pl.DataFrame(), Data(mongo=_FakeDB()))
    assert out.is_empty()

async def test_delta_empty_collection():
    df = pl.DataFrame({"pk": [1], "row_hash": ["a"]})
    out = await Delta("c", keys=["pk"]).apply(df, Data(mongo=_FakeDB(_FakeCollection(docs=[]))))
    assert out.equals(df)                             

async def test_delta_filters_unchanged():
    coll = _FakeCollection(docs=[{"pk": 1, "row_hash": "a"}, {"pk": 2, "row_hash": "b"}])
    df = pl.DataFrame({"pk": [1, 2, 3], "row_hash": ["a", "CHANGED", "c"]})
    out = await Delta("c", keys=["pk"]).apply(df, Data(mongo=_FakeDB(coll)))
    assert sorted(out["pk"].to_list()) == [2, 3]      

def test_repr_mongo_loader():
    assert repr(EnsureIndexes("c", keys=["pk"]))
    assert repr(Upsert("c", keys=["pk"]))
    assert repr(Delta("c", keys=["pk"]))
