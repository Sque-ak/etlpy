import polars as pl, pytest

from etl.generic import Pipeline, Pipestart, Data, Step, StopPipeline
from etl.transformer.steps import DropDuplicates, ExtractEntities, ClearText, RenameColumns, \
                                    Join, Aggregate, Lambda, SQL, GenerateKey, RowHash

from etl.loader.steps.datalake import Save
from etl.extractor.steps.datalake import Read
from etl.extractor.steps.clickhouse import Connect
from etl.loader.steps.clickhouse import EnsureTable, Insert

class _Stop(Step):
    def __init__(self, df=None):
        self.df = df
    async def apply(self, df, data=None):
        raise StopPipeline("halt", df=self.df)

class _MarkRan(Step):
    async def apply(self, df, data=None):
        return df.with_columns(pl.lit(True).alias("ran"))

async def test_stop_returns_current_df():
    start = pl.DataFrame({"a": [1]})
    out = await Pipeline([_Stop(), _MarkRan()], dataframe=start).run()
    assert out.equals(start)              
    assert "ran" not in out.columns      


async def test_stop_with_payload():
    payload = pl.DataFrame({"b": [9]})
    out = await Pipeline([_Stop(df=payload), _MarkRan()],
                         dataframe=pl.DataFrame({"a": [1]})).run()
    assert out.equals(payload)            


async def test_stop_verbose(capsys):
    await Pipeline([_Stop()], dataframe=pl.DataFrame({"a": [1]})).run(verbose=True)
    assert "[stop]" in capsys.readouterr().out


class _Boom(Step):
    async def apply(self, df, data=None):
        raise ValueError("boom")

async def test_step_error_keeps_type():
    with pytest.raises(ValueError) as exc:
        await Pipeline([_Boom()], dataframe=pl.DataFrame({"a": [1]})).run()
    assert any("failed at step" in note for note in exc.value.__notes__)


class _Add(Step):
    async def apply(self, df, data=None):
        return df.with_columns(pl.lit(1).alias("c"))

async def test_pipeline_add_run_verbose(capsys):
    out = await Pipeline([], dataframe=pl.DataFrame({"a": [1]})).add(_Add()).run(verbose=True)
    assert "c" in out.columns and capsys.readouterr().out      # .add() + verbose print

def test_pipeline_repr():
    assert "Pipeline" in repr(Pipeline([_Add()]))

async def test_pipestart():
    @Pipestart
    async def build():
        return Pipeline([_Add()], dataframe=pl.DataFrame({"a": [1]}))
    assert "c" in (await build()).columns                      # Pipestart wrapper

def test_data_repr():
    assert repr(Data(a=1))

def test_data_rejects_dataframe():
    with pytest.raises(TypeError):
        Data()["df"] = pl.DataFrame({"a": [1]})                # the guard

@pytest.mark.integration
async def test_heavy_pipeline_e2e(tmp_path, monkeypatch, ch_conn):
    monkeypatch.setenv("LAKE_DATA_DIR", str(tmp_path))

    raw = pl.DataFrame({
        "txn":      [1, 2, 3, 3],                          # txn=3 duplicate
        "sender":   ["Acme\n", "Beta", "Acme\n", "Acme\n"],
        "s_amount": [100, 200, 300, 300],
        "receiver": ["Beta", "Acme", "Beta", "Beta"],
        "r_amount": [100, 200, 300, 300],
    })
    lookup = pl.DataFrame({"name": ["Acme", "Beta"], "industry": ["tech", "finance"]})

    transform = Pipeline([
        DropDuplicates(),                                  # dedup: 4 to 3 rows
        ExtractEntities(sources=[                          # fact: 3 to 6 
            {"party": "sender",   "amount": "s_amount"},
            {"party": "receiver", "amount": "r_amount"},
        ]),
        ClearText(["party"]),                              # "Acme\n" to "Acme"
        RenameColumns({"party": "name"}),
        Join(other=lookup, on="name", how="left"),         # add industry
        Aggregate(group_by=["name", "industry"],           # 6 to 2
                  aggregations={"amount": "sum"}),
        Lambda(lambda df: df.with_columns((pl.col("amount_sum") > 500).alias("big"))),
        SQL("SELECT * FROM source WHERE amount_sum > 0"),
        GenerateKey(columns=["name"], key_name="pk"),
        RowHash(),
        Save(name="parties", layer="fact"),
    ], dataframe=raw)
    await transform.run()

    await Pipeline([
        Connect(**ch_conn),
        Read(layer="fact", name="parties"),
        EnsureTable("fact_parties", engine="MergeTree", order_by=["pk"]),
        Insert("fact_parties"),
    ]).run()

    import clickhouse_connect
    client = clickhouse_connect.get_client(**ch_conn)
    got = pl.from_arrow(
        client.query_arrow("SELECT name, industry, amount_sum, big FROM fact_parties ORDER BY name")
    )

    assert got.height == 2
    assert got["name"].to_list() == ["Acme", "Beta"]
    assert got["industry"].to_list() == ["tech", "finance"]
    assert got["amount_sum"].to_list() == [600, 600]
    assert "big" in got.columns

