import pyarrow as pa
from etl.generic.step import Step
from polars import DataFrame


class EnsureTable(Step):
    """
    Create the PostgreSQL table from the df schema if it does not exist,
    with a UNIQUE constraint on `keys` so Insert's ON CONFLICT works.

    Reads the connection from data["pg"] (set by Connect). Passes df through.

        :param table: target table.
        :param keys: unique business key column(s) (e.g. ["pk"]).
    """

    def __init__(self, table: str, keys: list[str] | str):
        self.table = table
        self.keys = [keys] if isinstance(keys, str) else keys

    def _pg_type(self, t: pa.DataType) -> str:
        if pa.types.is_int8(t) or pa.types.is_int16(t): return "SMALLINT"
        if pa.types.is_int32(t):     return "INTEGER"
        if pa.types.is_int64(t):     return "BIGINT"
        if pa.types.is_floating(t):  return "DOUBLE PRECISION"
        if pa.types.is_boolean(t):   return "BOOLEAN"
        if pa.types.is_date(t):      return "DATE"
        if pa.types.is_timestamp(t): return "TIMESTAMP"
        if pa.types.is_decimal(t):   return f"NUMERIC({t.precision}, {t.scale})"
        return "TEXT"

    def _build_ddl(self, schema: pa.Schema) -> str:
        cols = []
        for f in schema:
            not_null = " NOT NULL" if f.name in self.keys else ""
            cols.append(f'"{f.name}" {self._pg_type(f.type)}{not_null}')
        cols.append(f'UNIQUE ({", ".join(f"{k}" for k in self.keys)})')
        body = ",\n  ".join(cols)
        return f"CREATE TABLE IF NOT EXISTS {self.table} (\n  {body}\n)"

    async def apply(self, df: DataFrame, data=None):
        await data["pg"].execute(self._build_ddl(df.head(0).to_arrow().schema))
        return df

    def __repr__(self):
        return f"EnsureTable(table={self.table!r}, keys={self.keys!r})"
