import polars as pl, pytest
from pydantic import BaseModel
from etl.generic import Data, Pipeline, StopPipeline
from etl.transformer.steps.add_column import AddColumn
from etl.transformer.steps.cast_types import CastTypes
from etl.transformer.steps.clear_text import ClearText
from etl.transformer.steps.drop_columns import DropColumns
from etl.transformer.steps.drop_duplicates import DropDuplicates
from etl.transformer.steps.drop_nulls import DropNulls
from etl.transformer.steps.fill_nulls import FillNulls
from etl.transformer.steps.filter_rows import FilterRows
from etl.transformer.steps.rename_columns import RenameColumns
from etl.transformer.steps.trim_string import TrimString
from etl.transformer.steps.generate_key import GenerateKey
from etl.transformer.steps.row_hash import RowHash
from etl.transformer.steps.slambda import Lambda
from etl.transformer.steps.sql import SQL
from etl.transformer.steps.normalize_numeric import NormalizeNumeric
from etl.transformer.steps.aggregate import Aggregate
from etl.transformer.steps.join import Join
from etl.transformer.steps.extract_entities import ExtractEntities
from etl.transformer.steps import Union
from etl.transformer.steps import ToSchema

async def _apply(step, df):
    return await step.apply(df, Data())  

async def test_add_column():
    out = await _apply(AddColumn("source", "x"), pl.DataFrame({"a": [1, 2]}))
    assert out["source"].to_list() == ["x", "x"]

async def test_cast_types():
    out = await _apply(CastTypes({"a": pl.Int32}), pl.DataFrame({"a": [1, 2]}))
    assert out.schema["a"] == pl.Int32

async def test_rename_columns():
    out = await _apply(RenameColumns({"a": "b"}), pl.DataFrame({"a": [1]}))
    assert out.columns == ["b"]

async def test_drop_columns():
    out = await _apply(DropColumns(["b"]), pl.DataFrame({"a": [1], "b": [2]}))
    assert out.columns == ["a"]

async def test_drop_duplicates():
    out = await _apply(DropDuplicates(), pl.DataFrame({"a": [1, 1, 2]}))
    assert out.height == 2

async def test_drop_nulls():
    out = await _apply(DropNulls(["a"]), pl.DataFrame({"a": [1, None, 2]}))
    assert out["a"].to_list() == [1, 2]

async def test_fill_nulls():
    out = await _apply(FillNulls(0, ["a"]), pl.DataFrame({"a": [1, None]}))
    assert out["a"].to_list() == [1, 0]

async def test_filter_rows():
    out = await _apply(FilterRows(pl.col("a") > 1), pl.DataFrame({"a": [1, 2, 3]}))
    assert out["a"].to_list() == [2, 3]

async def test_trim_string():
    out = await _apply(TrimString(["a"]), pl.DataFrame({"a": ["  x  "]}))
    assert out["a"].to_list() == ["x"]

async def test_clear_text():
    out = await _apply(ClearText(["a"]), pl.DataFrame({"a": ['"x"\n']}))
    assert out["a"].to_list() == ["x"]

async def test_generate_key():
    out = await _apply(GenerateKey(columns=["a"], key_name="pk"), pl.DataFrame({"a": [1, 2]}))
    assert out["pk"].dtype == pl.String and out["pk"].n_unique() == 2

async def test_row_hash():
    out = await _apply(RowHash(), pl.DataFrame({"a": [1, 2]}))
    assert "row_hash" in out.columns and out["row_hash"].n_unique() == 2

async def test_lambda():
    out = await _apply(Lambda(lambda df: df.with_columns((pl.col("a") + 1).alias("b"))),
                       pl.DataFrame({"a": [1]}))
    assert out["b"].to_list() == [2]

async def test_sql():
    out = await _apply(SQL("SELECT a FROM source WHERE a > 1"), pl.DataFrame({"a": [1, 2, 3]}))
    assert out["a"].to_list() == [2, 3]

async def test_normalize_numeric():
    out = await _apply(NormalizeNumeric(["a"]), pl.DataFrame({"a": [0, 5, 10]}))
    assert out["a"].to_list() == [0.0, 0.5, 1.0]

async def test_aggregate():
    df = pl.DataFrame({"g": ["x", "x", "y"], "v": [1, 2, 3]})
    out = await _apply(Aggregate(group_by=["g"], aggregations={"v": "sum"}), df)
    assert dict(zip(out["g"], out["v_sum"])) == {"x": 3, "y": 3}

async def test_join():
    out = await _apply(Join(other=pl.DataFrame({"id": [1], "name": ["a"]}), on="id", how="left"),
                       pl.DataFrame({"id": [1, 2], "x": [10, 20]}))
    assert out.sort("id")["name"].to_list() == ["a", None]

async def test_join_with_pipeline_branch():
    main = pl.DataFrame({"id": [1, 2], "x": [10, 20]})
    raw_lookup = pl.DataFrame({"id": [1, 1, 2], "name": ["a", "a", "b"]})   

    branch = Pipeline([DropDuplicates()], dataframe=raw_lookup)          

    out = await Join(other=branch, on="id", how="left").apply(main, Data())

    assert out.height == 2 
    assert out.sort("id")["name"].to_list() == ["a", "b"]

async def test_extract_entities():
    df = pl.DataFrame({"sender": ["A"], "s_amt": [10], "receiver": ["B"], "r_amt": [20]})
    out = await _apply(ExtractEntities(sources=[
        {"party": "sender", "amt": "s_amt"},
        {"party": "receiver", "amt": "r_amt"},
    ]), df)
    assert out.height == 2 and set(out["party"].to_list()) == {"A", "B"}

async def test_generate_key_hash_int():
    out = await _apply(GenerateKey(columns=["a"], mode="hash_int"), pl.DataFrame({"a": [1, 2]}))
    assert out["pk"].dtype == pl.UInt64

async def test_generate_key_sequential():
    out = await _apply(GenerateKey(key_name="id", mode="sequential", order_by="a"), pl.DataFrame({"a": [3, 1, 2]}))
    assert out["id"].to_list() == [1, 2, 3]

async def test_fill_nulls_dict():
    out = await _apply(FillNulls({"a": 0, "b": "x"}), pl.DataFrame({"a": [None], "b": [None]}))
    assert out.row(0) == (0, "x")

async def test_normalize_zscore():
    out = await _apply(NormalizeNumeric(["a"], method="zscore"), pl.DataFrame({"a": [1.0, 2.0, 3.0]}))
    assert out["a"].mean() == pytest.approx(0.0)

async def test_drop_columns_exclude():
    out = await _apply(DropColumns(["a"], exclude=True), pl.DataFrame({"a": [1], "b": [2]}))
    assert out.columns == ["a"]

async def test_join_select_prefix():
    right = pl.DataFrame({"id": [1], "name": ["a"], "extra": [9]})
    out = await _apply(Join(other=right, on="id", select=["name"], prefix="r_"), pl.DataFrame({"id": [1]}))
    assert "r_name" in out.columns and "extra" not in out.columns

async def test_extract_entities_defaults():
    out = await _apply(
        ExtractEntities(sources=[{"party": "sender", "amount": "amt"}], defaults={"party": "UNKNOWN"}),
        pl.DataFrame({"sender": ["A", None], "amt": [10, 20]}),
    )
    assert out["party"].to_list() == ["A", "UNKNOWN"]

async def test_sql_from_file(tmp_path):
    f = tmp_path / "q.sql"
    f.write_text("SELECT a FROM source WHERE a > 1")
    out = await _apply(SQL.from_file(str(f)), pl.DataFrame({"a": [1, 2, 3]}))
    assert out["a"].to_list() == [2, 3]

async def test_aggregate_unknown():
    with pytest.raises(ValueError):
        await _apply(Aggregate(group_by=["g"], aggregations={"v": "nope"}),
                     pl.DataFrame({"g": ["x"], "v": [1]}))

async def test_fill_nulls_all_columns():
    out = await _apply(FillNulls(0), pl.DataFrame({"a": [None], "b": [None]}))
    assert out.row(0) == (0, 0)

async def test_union_dataframes():            
    out = await Union(other=pl.DataFrame({"a": [2]})).apply(pl.DataFrame({"a": [1]}), Data())
    assert out["a"].to_list() == [1, 2]

async def test_union_with_pipeline():                   
    branch = Pipeline([], dataframe=pl.DataFrame({"a": [2]}))
    out = await Union(other=branch).apply(pl.DataFrame({"a": [1]}), Data())
    assert out["a"].to_list() == [1, 2]

def test_union_repr():                        
    assert repr(Union(other=pl.DataFrame())) == "Union()"

class _Txn(BaseModel):
    id: int
    name: str
    amount: float | None = None

async def test_toschema_validates_and_types():
    df = pl.DataFrame({"id": ["1"], "name": ["x"], "extra": ["drop"]})   # id str->int, extra exclude
    out = await ToSchema(_Txn).apply(df, Data())
    assert out.schema == {"id": pl.Int64, "name": pl.String, "amount": pl.Float64}
    assert out["id"].to_list() == [1]
    assert "extra" not in out.columns

async def test_toschema_mapping():
    out = await ToSchema(_Txn, mapping={"txn_id": "id"}).apply(pl.DataFrame({"txn_id": [1], "name": ["x"]}), Data())
    assert out["id"].to_list() == [1]

async def test_toschema_empty_stops():
    with pytest.raises(StopPipeline):
        await ToSchema(_Txn).apply(pl.DataFrame(), Data())

def test_toschema_repr():
    assert repr(ToSchema(_Txn))

async def test_toschema_list_field():
    from pydantic import BaseModel
    class _M(BaseModel):
        tags: list[str] | None = None
    out = await ToSchema(_M).apply(pl.DataFrame({"tags": [["a", "b"]]}), Data())
    assert out.schema["tags"] == pl.List(pl.String)
