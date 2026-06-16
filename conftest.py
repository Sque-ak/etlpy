import pytest

@pytest.fixture(scope="session")
def ch_conn():
    from testcontainers.clickhouse import ClickHouseContainer
    with ClickHouseContainer() as ch:
        yield {
            "host": ch.get_container_host_ip(),
            "port": int(ch.get_exposed_port(8123)),
            "username": ch.username,
            "password": ch.password,
            "database": ch.dbname,
        }