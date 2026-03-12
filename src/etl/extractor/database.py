from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from etl.extractor import Source

logger = logging.getLogger(__name__)


class ClickHouse(Source):
    """
    Extract data from a ClickHouse database.

    Wraps ``clickhouse-connect`` and provides convenience methods for
    querying tables, fetching row hashes for incremental comparison,
    and running arbitrary SQL.

    Example::

        from etl.extractor.database import ClickHouse

        ch = ClickHouse(
            host="clickhouse-host",
            port=8123,
            database="analytics",
        )

        # Extract a full table
        df = ch.extract()                           # requires .table to be set
        df = ch.query("SELECT * FROM fact_transactions LIMIT 100")

        # Get existing row_hash values for delta comparison
        hashes = ch.hashes("fact_transactions", biz_key=["transaction_id"])
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8123,
        database: str = "default",
        username: str = "default",
        password: str = "",
        secure: bool = False,
        table: str | None = None,
        sql: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            host: ClickHouse host.
            port: ClickHouse HTTP port (default 8123).
            database: Target database.
            username: Auth user.
            password: Auth password.
            secure: Use HTTPS.
            table: Default table name (used by :meth:`extract`).
            sql: Custom SQL query (used by :meth:`extract` if set).
            **kwargs: Extra arguments forwarded to ``clickhouse_connect.get_client``.
        """
        import clickhouse_connect

        self.database = database
        self.table = table
        self.sql = sql

        self.client = clickhouse_connect.get_client(
            host=host,
            port=port,
            database=database,
            username=username,
            password=password,
            secure=secure,
            **kwargs,
        )
        logger.info("Connected to ClickHouse at %s:%s/%s", host, port, database)

    def extract(self) -> pd.DataFrame:
        """
        Extract data using the configured ``sql`` or ``table``.

        Priority: ``self.sql`` -> ``SELECT * FROM self.table``.

        Returns:
            pandas DataFrame with the query results.

        Raises:
            ValueError: If neither ``sql`` nor ``table`` is set.
        """
        if self.sql:
            return self.query(self.sql)
        if self.table:
            return self.query(f"SELECT * FROM {self.table}")
        raise ValueError("Set 'table' or 'sql' before calling extract()")

    def query(self, sql: str) -> pd.DataFrame:
        """Execute a SELECT and return a pandas DataFrame."""
        return self.client.query_df(sql)

    def command(self, sql: str) -> Any:
        """Execute a DDL/DML statement (CREATE, DROP, INSERT, …)."""
        return self.client.command(sql)

    def tables(self) -> list[str]:
        """List all tables in the current database."""
        df = self.client.query_df(f"SHOW TABLES FROM {self.database}")
        return df.iloc[:, 0].tolist() if len(df) > 0 else []

    def columns(self, table: str | None = None) -> list[str]:
        """List columns of a table."""
        tbl = table or self.table
        if not tbl:
            raise ValueError("Specify table name")
        df = self.client.query_df(f"DESCRIBE TABLE {tbl}")
        return df.iloc[:, 0].tolist() if len(df) > 0 else []

    def count(self, table: str | None = None) -> int:
        """Return row count of a table."""
        tbl = table or self.table
        if not tbl:
            raise ValueError("Specify table name")
        return self.client.command(f"SELECT count() FROM {tbl}")

    def table_exists(self, table: str) -> bool:
        """Check if a table exists."""
        return bool(self.client.command(f"EXISTS TABLE {table}"))

    def hashes(
        self,
        table: str | None = None,
        biz_key: list[str] | None = None,
        extra_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Extract ``row_hash`` (and optionally other columns) from a ClickHouse
        table, using ``FINAL`` to get deduplicated results from
        ``ReplacingMergeTree``.

        This is the primary method for fetching existing hashes to compare
        against new data before loading.

        Args:
            table: Table name (falls back to ``self.table``).
            biz_key: Business key column(s) to select alongside ``row_hash``.
                     If *None*, only ``row_hash`` is returned.
            extra_columns: Additional columns to include in the result.

        Returns:
            DataFrame with ``row_hash`` + business key columns.

        Example::

            ch = ClickHouse(host="clickhouse", database="analytics")

            # Get all hashes with their business keys
            existing = ch.hashes(
                table="fact_transactions",
                biz_key=["transaction_id"],
            )

            # Compare with new data
            new_hashes = new_df[["transaction_id", "row_hash"]]
            delta = new_hashes.merge(
                existing, on="transaction_id", how="left", suffixes=("_new", "_old")
            )
            changed = delta[delta["row_hash_new"] != delta["row_hash_old"]]
        """
        tbl = table or self.table
        if not tbl:
            raise ValueError("Specify table name")

        if not self.table_exists(tbl):
            logger.warning("Table %s does not exist — returning empty DataFrame", tbl)
            return pd.DataFrame()

        cols = []
        if biz_key:
            cols.extend(f"`{c}`" for c in biz_key)
        cols.append("`row_hash`")
        if extra_columns:
            cols.extend(f"`{c}`" for c in extra_columns)

        select = ", ".join(cols)
        sql = f"SELECT {select} FROM {tbl} FINAL"

        try:
            df = self.client.query_df(sql)
            logger.info("Fetched %d hashes from %s", len(df), tbl)
            return df
        except Exception as e:
            logger.warning("Could not read hashes from %s: %s", tbl, e)
            return pd.DataFrame()

    def compare(
        self,
        new_df: pd.DataFrame,
        table: str | None = None,
        biz_key: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Compare new data with existing ClickHouse hashes and return only
        new or changed rows (the delta).

        Args:
            new_df: DataFrame with ``row_hash`` and ``biz_key`` columns.
            table: ClickHouse table to compare against.
            biz_key: Business key column(s) for matching rows.

        Returns:
            DataFrame containing only rows that are new or have changed.
        """
        if not biz_key or "row_hash" not in new_df.columns:
            return new_df

        existing = self.hashes(table=table, biz_key=biz_key)

        if existing.empty or "row_hash" not in existing.columns:
            return new_df

        existing = existing[biz_key + ["row_hash"]].rename(
            columns={"row_hash": "_existing_hash"}
        )

        merged = new_df.merge(existing, on=biz_key, how="left")

        mask = (
            merged["_existing_hash"].isna()
            | (merged["row_hash"] != merged["_existing_hash"])
        )
        delta = merged.loc[mask].drop(columns=["_existing_hash"])

        skipped = len(new_df) - len(delta)
        if skipped > 0:
            logger.info("Skipped %d unchanged rows for %s", skipped, table)

        return delta

    def __repr__(self) -> str:
        return f"ClickHouse(database={self.database!r}, table={self.table!r})"