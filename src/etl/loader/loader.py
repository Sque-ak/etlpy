"""
Loader module for loading data from Storage into databases.

Currently supports ClickHouse. For use, you must install the library `clickhouse-connect`.

Example::

    from etl.storage import Storage, Layer
    from etl.loader import Loader

    storage = Storage(data="/data")

    loader = Loader(
        storage=storage,
        host="clickhouse-host",
        port=8123,
        database="analytics",
    )

    # Simple full load

    # Load a single fact file -> ClickHouse table
    loader.load("transactions", layer=Layer.FACT)

    # Load ALL fact files -> one table per file, then archive
    loader.load_all(layer=Layer.FACT, archive=True)

    # Incremental load (row_hash)

    # If the parquet file contains ``row_hash`` and ``_loaded_at`` columns
    # (added by the transformer's RowHash + AddColumn steps), the loader
    # can compare hashes to only insert new or changed rows.
    #
    # Just pass ``biz_key`` - the business key column(s) used for matching:

    loader.load(
        "transactions",
        layer=Layer.FACT,
        biz_key=["transaction_id"],          # delta comparison key
        engine="ReplacingMergeTree(_loaded_at)",
        partition_by="toYYYYMM(toDate(date))",
    )

    # load_all with per-table config
    loader.load_all(
        layer=Layer.FACT,
        table_config={
            "fact_transactions": {
                "order_by": ["transaction_id"],
                "partition_by": "toYYYYMM(toDate(date))",
                "biz_key": ["transaction_id"],
            },
            "fact_accounts": {
                "order_by": ["account_id"],
                "biz_key": ["account_id"],
            },
        },
        table_map={"transactions": "fact_transactions"},
    )

    # Query ClickHouse
    df = loader.query("SELECT * FROM fact_transactions LIMIT 10")

    Notes on fact tables & BI analytics
    ------------------------------------
    Store granular data (each row = one event/transaction) with a date/datetime
    column.  Use ClickHouse's ``MergeTree`` engine ``ORDER BY (date, …)``
    and ``PARTITION BY toYYYYMM(date)`` for efficient range scans.

    In BI queries the date column gives you all time dimensions for free::

        SELECT
            toYear(date)    AS year,
            toQuarter(date) AS quarter,
            toMonth(date)   AS month,
            category,
            sum(amount)     AS total
        FROM fact_transactions
        GROUP BY CUBE(year, quarter, month, category)

    ``GROUP BY CUBE`` generates every combination of the listed dimensions —
    perfect for OLAP-style reports without pre-aggregating data.
"""

from __future__ import annotations

import logging
from typing import Literal

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from etl.storage import Storage, Layer, Mode

logger = logging.getLogger(__name__)
 
# PyArrow type -> ClickHouse type mapping
def _arrow_type_to_ch(arrow_type: pa.DataType) -> str:
    """Convert a PyArrow data type to a ClickHouse type string."""
    if pa.types.is_int8(arrow_type):
        return "Int8"
    if pa.types.is_int16(arrow_type):
        return "Int16"
    if pa.types.is_int32(arrow_type):
        return "Int32"
    if pa.types.is_int64(arrow_type):
        return "Int64"
    if pa.types.is_uint8(arrow_type):
        return "UInt8"
    if pa.types.is_uint16(arrow_type):
        return "UInt16"
    if pa.types.is_uint32(arrow_type):
        return "UInt32"
    if pa.types.is_uint64(arrow_type):
        return "UInt64"
    if pa.types.is_float16(arrow_type):
        return "Float32"
    if pa.types.is_float32(arrow_type):
        return "Float32"
    if pa.types.is_float64(arrow_type):
        return "Float64"
    if pa.types.is_boolean(arrow_type):
        return "Bool"
    if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        return "String"
    if pa.types.is_date(arrow_type):
        return "Date"
    if pa.types.is_timestamp(arrow_type):
        return "DateTime64(3)"
    if pa.types.is_decimal(arrow_type):
        return f"Decimal({arrow_type.precision}, {arrow_type.scale})"
    if pa.types.is_binary(arrow_type) or pa.types.is_large_binary(arrow_type):
        return "String"
    if pa.types.is_list(arrow_type):
        inner = _arrow_type_to_ch(arrow_type.value_type)
        return f"Array({inner})"
    return "String"  # safe fallback


class Loader:
    """
    Load data from Storage layers into ClickHouse.

    Supports two loading strategies:

    **Full load** (default)
        Reads the parquet file and inserts all rows.
        Uses ``MergeTree`` engine.

    **Incremental load** (when ``biz_key`` is provided)
        Compares ``row_hash`` from the parquet file with existing hashes in
        ClickHouse (via ``SELECT ... FINAL``).  Only new or changed rows are
        inserted.  Uses ``ReplacingMergeTree(_loaded_at)`` - ClickHouse
        automatically deduplicates rows with the same ORDER BY key, keeping
        the row with the latest ``_loaded_at``.

    After a successful load the source files are (optionally) archived
    through :class:`etl.Storage` so the Data Lake stays clean.
    """

    def __init__(
        self,
        storage: Storage,
        host: str = "localhost",
        port: int = 8123,
        database: str = "default",
        username: str = "default",
        password: str = "",
        secure: bool = False,
        **kwargs,
    ) -> None:
        import clickhouse_connect

        self.storage = storage
        self.database = database
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

    def load(
        self,
        table: str,
        layer: Layer = Layer.FACT,
        filename: str | None = None,
        date: str | None = None,
        mode: Mode = Mode.DATE,
        archive: bool = True,
        create: bool = True,
        engine: str | None = None,
        order_by: str | list[str] | None = None,
        partition_by: str | None = None,
        if_exists: Literal["append", "replace", "error"] = "append",
        biz_key: list[str] | None = None,
    ) -> int:
        """
        Load a single parquet file from Storage into a ClickHouse table.

        Args:
            table: Target ClickHouse table name.
            layer: Source layer (default: FACT).
            filename: Parquet filename.  Defaults to ``"{table}.parquet"``.
            date: Date partition.  Default: today.
            mode: Storage read mode.
            archive: If True, archive the file after successful load.
            create: Auto-create the table if it doesn't exist.
            engine: ClickHouse engine.  Auto-detected:
                    ``ReplacingMergeTree(_loaded_at)`` when ``biz_key`` is set,
                    ``MergeTree`` otherwise.
            order_by: ``ORDER BY`` column(s) for the engine.
                      Auto-detected from schema when *None*.
            partition_by: Optional ``PARTITION BY`` expression
                          (e.g. ``"toYYYYMM(toDate(date))"``).
            if_exists: ``"append"`` — insert new rows,
                       ``"replace"`` — truncate then insert,
                       ``"error"`` — raise if table exists.
            biz_key: Business key column(s) for incremental (delta) loading.
                     When provided, the loader compares ``row_hash`` in the
                     parquet with existing hashes in ClickHouse and inserts
                     only new/changed rows.  Requires ``row_hash`` column
                     (see :class:`etl.transformer.steps.RowHash`).

        Returns:
            Number of rows inserted.
        """
        fname = filename or f"{table}.parquet"

        # Auto-select engine
        if engine is None:
            engine = "ReplacingMergeTree(_loaded_at)" if biz_key else "MergeTree"

        # Read as PyArrow Table, then convert to plain pandas
        # (avoids ArrowDtype types that clickhouse-connect can't handle)
        arrow_table = self.storage.read(layer, fname, date=date, mode=mode, as_arrow=True)
        df = arrow_table.to_pandas()
        logger.info("Read %d rows from %s/%s", len(df), layer, fname)

        # DDL
        if create:
            self._ensure_table(table, df, engine, order_by, partition_by, if_exists)

        if if_exists == "replace":
            self.client.command(f"TRUNCATE TABLE {table}")

        # Delta comparison (incremental)
        if biz_key and "row_hash" in df.columns:
            df = self._compute_delta(table, df, biz_key)

        if df.empty:
            logger.info("No changes for %s — skipping INSERT", table)
        else:
            self.client.insert_df(table, df)
            logger.info("Loaded %d rows into %s", len(df), table)

        rows_loaded = len(df)

        # Archive
        if archive:
            self.storage.archive_file(layer, fname, date=date, mode=mode)
            logger.info("Archived %s/%s", layer, fname)

        return rows_loaded

    def load_all(
        self,
        layer: Layer = Layer.FACT,
        date: str | None = None,
        mode: Mode = Mode.DATE,
        archive: bool = True,
        create: bool = True,
        engine: str | None = None,
        order_by: str | list[str] | None = None,
        partition_by: str | None = None,
        if_exists: Literal["append", "replace", "error"] = "append",
        table_prefix: str = "",
        table_map: dict[str, str] | None = None,
        table_config: dict[str, dict] | None = None,
        biz_key: list[str] | None = None,
    ) -> list[str]:
        """
        Load **all** parquet files from a Storage layer into ClickHouse.

        Each file becomes a separate table (``filename stem -> table name``).

        Args:
            layer: Source layer (default: FACT).
            date: Date partition.  Default: today.
            mode: Storage read mode.
            archive: Archive files after load.
            create: Auto-create tables.
            engine: ClickHouse engine (applied to all tables unless
                    overridden by ``table_config``).
            order_by: ``ORDER BY`` column(s) (global default).
            partition_by: ``PARTITION BY`` expression (global default).
            if_exists: ``"append"`` / ``"replace"`` / ``"error"``.
            table_prefix: Auto-prefix for table names
                          (e.g. ``"fact_"`` -> ``fact_transactions``).
            table_map: Explicit mapping ``{file_stem: table_name}``.
                       Files not in the map fall back to
                       ``table_prefix + stem``.
            table_config: Per-table settings dict::

                    {
                        "fact_transactions": {
                            "order_by": ["transaction_id"],
                            "partition_by": "toYYYYMM(toDate(date))",
                            "biz_key": ["transaction_id"],
                            "engine": "ReplacingMergeTree(_loaded_at)",
                        },
                    }

                Keys: ``order_by``, ``partition_by``, ``biz_key``, ``engine``.
                Missing keys fall back to the method-level defaults.
            biz_key: Global ``biz_key`` default (used when ``table_config``
                     doesn't specify one for a given table).

        Returns:
            List of successfully loaded table names.
        """
        files = self.storage.list(layer, date=date, mode=mode)
        if not files:
            logger.warning("No files found in %s layer", layer)
            return []

        table_config = table_config or {}

        loaded: list[str] = []
        for file_path in files:
            stem = file_path.stem
            if table_map and stem in table_map:
                table_name = table_map[stem]
            else:
                table_name = f"{table_prefix}{stem}"

            # Per-table overrides
            cfg = table_config.get(table_name, {})
            t_order_by = cfg.get("order_by", order_by)
            t_partition_by = cfg.get("partition_by", partition_by)
            t_biz_key = cfg.get("biz_key", biz_key)
            t_engine = cfg.get("engine", engine)

            try:
                self.load(
                    table=table_name,
                    layer=layer,
                    filename=file_path.name,
                    date=date,
                    mode=mode,
                    archive=archive,
                    create=create,
                    engine=t_engine,
                    order_by=t_order_by,
                    partition_by=t_partition_by,
                    if_exists=if_exists,
                    biz_key=t_biz_key,
                )
                loaded.append(table_name)
            except Exception as e:
                logger.error("Failed to load %s: %s", table_name, e, exc_info=True)

        logger.info(
            "Loaded %d table(s) from %s layer: %s", len(loaded), layer, loaded
        )
        return loaded

    def query(self, sql: str):
        """Execute a ``SELECT`` and return a pandas DataFrame."""
        return self.client.query_df(sql)

    def command(self, sql: str):
        """Execute a DDL / DML statement (CREATE, DROP, ALTER, …)."""
        return self.client.command(sql)

    def tables(self) -> list[str]:
        """List all tables in the current database."""
        df = self.client.query_df(f"SHOW TABLES FROM {self.database}")
        return df.iloc[:, 0].tolist() if len(df) > 0 else []

    def table_exists(self, table: str) -> bool:
        """Check if a table exists in ClickHouse."""
        return bool(self.client.command(f"EXISTS TABLE {table}"))

    # Incremental (delta) loading internals

    def _get_existing_hashes(self, table: str, biz_key: list[str]) -> pd.DataFrame:
        """Fetch ``row_hash`` by business key from ClickHouse (FINAL for dedup)."""
        key_cols = ", ".join(f"`{c}`" for c in biz_key)
        sql = f"SELECT {key_cols}, `row_hash` FROM {table} FINAL"
        try:
            return self.client.query_df(sql)
        except Exception as e:
            logger.warning("Could not read hashes from %s: %s", table, e)
            return pd.DataFrame()

    def _compute_delta(
        self,
        table: str,
        new_df: pd.DataFrame,
        biz_key: list[str],
    ) -> pd.DataFrame:
        """
        Compare ``row_hash`` in *new_df* with existing hashes in ClickHouse.

        Returns only rows that are new or changed:
          - ``biz_key`` not found in ClickHouse -> new row
          - ``row_hash`` differs -> changed row
        """
        if not self.table_exists(table):
            return new_df

        existing_df = self._get_existing_hashes(table, biz_key)

        if existing_df.empty or "row_hash" not in existing_df.columns:
            return new_df

        existing = existing_df[biz_key + ["row_hash"]].rename(
            columns={"row_hash": "_existing_hash"}
        )

        merged = new_df.merge(existing, on=biz_key, how="left")

        # New (NaN) or changed (hash mismatch)
        mask = (
            merged["_existing_hash"].isna()
            | (merged["row_hash"] != merged["_existing_hash"])
        )
        delta = merged.loc[mask].drop(columns=["_existing_hash"])

        skipped = len(new_df) - len(delta)
        if skipped > 0:
            logger.info("Skipped %d unchanged rows for %s", skipped, table)

        return delta

    # DDL internals

    def _ensure_table(
        self,
        table: str,
        df,
        engine: str,
        order_by: str | list[str] | None,
        partition_by: str | None,
        if_exists: str,
    ) -> None:
        """Create a ClickHouse table from a DataFrame schema if needed."""
        exists = self.client.command(f"EXISTS TABLE {table}")

        if exists and if_exists == "error":
            raise ValueError(f"Table {table} already exists")

        if exists:
            return

        # Build DDL from pyarrow schema
        pa_table = pa.Table.from_pandas(df, preserve_index=False)

        columns_ddl: list[str] = []
        for field in pa_table.schema:
            ch_type = _arrow_type_to_ch(field.type)
            if field.nullable and ch_type not in ("String",):
                ch_type = f"Nullable({ch_type})"
            columns_ddl.append(f"    `{field.name}` {ch_type}")

        # ORDER BY
        if order_by is None:
            order_by = self._detect_order_by(pa_table.schema)
        if isinstance(order_by, str):
            order_by = [order_by]
        order_clause = (
            ", ".join(f"`{c}`" for c in order_by) if order_by else "tuple()"
        )

        # PARTITION BY
        partition_clause = (
            f"\nPARTITION BY {partition_by}" if partition_by else ""
        )

        ddl = (
            f"CREATE TABLE {table} (\n"
            f"{chr(44).join(columns_ddl)}\n"
            f") ENGINE = {engine}\n"
            f"ORDER BY ({order_clause})"
            f"{partition_clause}"
        )

        self.client.command(ddl)
        logger.info("Created table %s (%s)", table, engine)

    @staticmethod
    def _detect_order_by(schema: pa.Schema) -> list[str]:
        """Heuristic: pick ``pk``/``id`` or date-like columns for ORDER BY."""
        for field in schema:
            if field.name.lower() in ("id", "pk"):
                return [field.name]

        candidates: list[str] = []
        for field in schema:
            name = field.name.lower()
            if "date" in name or "time" in name:
                candidates.append(field.name)
            elif name.endswith("_id"):
                candidates.append(field.name)
        return candidates[:2] if candidates else []

    def __repr__(self) -> str:
        return f"Loader(database={self.database!r})"
