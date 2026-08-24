import polars as pl, pytest

from etl.generic import Pipeline, Pipestart, Data, Step, StopPipeline
from etl.transformer.steps import DropDuplicates, ExtractEntities, ClearText, RenameColumns, \
                                    Join, Aggregate, Lambda, SQL, GenerateKey, RowHash

from etl.loader.steps.datalake import Save
from etl.extractor.steps.datalake import Read
from etl.extractor.steps.clickhouse import Connect
from etl.loader.steps.clickhouse import EnsureTable, Insert
from etl.generic import pipeline as pipe_mod

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


pytest.fixture(autouse=True)
def _clear_subscribers():
    pipe_mod._subscribers.clear()
    yield
    pipe_mod._subscribers.clear()


class _Add(Step):
    async def apply(self, df, data=None):
        return pl.DataFrame({"a": [1]}).with_columns(pl.lit(1).alias("c"))

class _Boom(Step):
    async def apply(self, df, data=None):
        raise ValueError("boom")


async def test_events_order():
    events = []
    pipe_mod.subscribe(lambda e, p: events.append(e))
    await Pipeline([_Add()], dataframe=pl.DataFrame({"a": [1]})).run()
    assert events == ["pipeline_start", "step_start", "step_end", "pipeline_end"]

async def test_end_status_completed():
    seen = {}
    pipe_mod.subscribe(lambda e, p: seen.setdefault(e, p))
    await Pipeline([_Add()], dataframe=pl.DataFrame({"a": [1]})).run()
    assert seen["pipeline_end"]["status"] == "completed"

async def test_stop_emits_step_stop_and_terminal():
    events = []
    pipe_mod.subscribe(lambda e, p: events.append((e, p.get("status"))))
    await Pipeline([_Stop()], dataframe=pl.DataFrame({"a": [1]})).run()
    names = [e for e, _ in events]
    assert "step_stop" in names
    assert names[-1] == "pipeline_end"
    assert dict(events)["pipeline_end"] == "stopped"

async def test_step_error_event_and_terminal():
    events = []
    pipe_mod.subscribe(lambda e, p: events.append((e, p.get("status"))))
    with pytest.raises(ValueError):
        await Pipeline([_Boom()], dataframe=pl.DataFrame({"a": [1]})).run()
    names = [e for e, _ in events]
    assert "step_error" in names
    assert names[-1] == "pipeline_end"
    assert dict(events)["pipeline_end"] == "error"

async def test_async_subscriber_awaited():
    hits = []
    async def sub(e, p):
        hits.append(e)
    pipe_mod.subscribe(sub)
    await Pipeline([_Add()], dataframe=pl.DataFrame({"a": [1]})).run()
    assert "pipeline_end" in hits

async def test_broken_subscriber_does_not_crash():
    pipe_mod.subscribe(lambda e, p: (_ for _ in ()).throw(RuntimeError("plugin bug")))
    out = await Pipeline([_Add()], dataframe=pl.DataFrame({"a": [1]})).run()
    assert "c" in out.columns

async def test_unsubscribe():
    events = []
    fn = lambda e, p: events.append(e)
    pipe_mod.subscribe(fn); pipe_mod.unsubscribe(fn)
    await Pipeline([_Add()], dataframe=pl.DataFrame({"a": [1]})).run()
    assert events == []

def test_subscribe_no_duplicate():
    fn = lambda e, p: None
    pipe_mod.subscribe(fn); pipe_mod.subscribe(fn)
    assert pipe_mod._subscribers.count(fn) == 1
