import pytest, pyarrow as pa, polars as pl
from etl.generic import Pipeline

from etl.loader.steps.clickhouse import Connect
from etl.extractor.steps.clickhouse import Read

import os
os.environ.setdefault("TESTCONTAINERS_RYUK_DISABLED", "true")


@pytest.fixture(scope="session")
def ch_conn():
    """Spin up a real clickhouse in docker once per test session"""
    from testcontainers.clickhouse import ClickHouseContainer
    with ClickHouseContainer() as ch:
        yield {
            "host": ch.get_container_host_ip(),
            "port": int(ch.get_exposed_port(8123)),   # HTTP port for clickhouse-connect
            "username": ch.username,
            "password": ch.password,
            "database": ch.dbname,
        }

@pytest.mark.integration
async def test_read_clickhouse_real(ch_conn):
    import clickhouse_connect

    # seed: Make table and test data
    seed = clickhouse_connect.get_client(**ch_conn)
    seed.command("CREATE TABLE companies (id Int32, name String) ENGINE = MergeTree ORDER BY id")
    seed.insert("companies", [[1, "a"], [2, "b"]], column_names=["id", "name"])

    pipe = Pipeline(
        [
            Connect(**ch_conn),
            Read("SELECT id, name FROM companies ORDER BY id")
        ]
    )
    out = await pipe.run()

    assert out.equals(pl.DataFrame({"id": [1, 2], "name": ["a", "b"]}))