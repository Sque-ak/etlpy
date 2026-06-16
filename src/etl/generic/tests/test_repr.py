import polars as pl
import pytest

from etl.transformer.steps import (
    AddColumn, CastTypes, RenameColumns, DropColumns, DropDuplicates, DropNulls,
    FillNulls, FilterRows, TrimString, ClearText, NormalizeNumeric, GenerateKey,
    RowHash, Aggregate, Join, ExtractEntities, SQL, Lambda,
)
from etl.loader.steps.clickhouse import Connect, EnsureTable, Delta, Insert
from etl.loader.steps.datalake import Save, Archive
from etl.extractor.steps.datalake import Read as ReadLake
from etl.extractor.steps.clickhouse.read import Read as ReadCH
from etl.extractor.steps.api.authenticate import Authenticate


STEPS = [
    AddColumn("c", 1),
    CastTypes({"a": pl.Int32}),
    RenameColumns({"a": "b"}),
    DropColumns(["a"]),
    DropDuplicates(),
    DropNulls(),
    FillNulls(0),
    FilterRows(pl.col("a") > 0),
    TrimString(),
    ClearText(),
    NormalizeNumeric(["a"]),
    GenerateKey(columns=["a"]),
    RowHash(),
    Aggregate(group_by=["g"], aggregations={"v": "sum"}),
    Join(other=pl.DataFrame(), on="id"),
    ExtractEntities(sources=[{"a": "b"}]),
    SQL("SELECT 1"),
    Lambda(lambda df: df),
    Connect(host="x"),
    EnsureTable("t", order_by=["pk"]),
    Delta("t", keys=["pk"]),
    Insert("t"),
    Save(name="x"),
    Archive(layer="raw", name="x"),
    ReadLake(layer="raw", name="x"),
    ReadCH("SELECT 1"),
    Authenticate(url="u", credentials={}),
]


@pytest.mark.parametrize("step", STEPS, ids=lambda s: f"{type(s).__module__.split('.')[-2]}.{type(s).__name__}")
def test_repr(step):
    assert repr(step)        # executes __repr__, must return a non-empty string