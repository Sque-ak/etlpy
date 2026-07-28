import pyarrow as pa
from etl.generic.step import Step
from polars import DataFrame


class EnsureTable(Step):
    """
    Create the SQLite table from the df schema if it does not exist, with a
    UNIQUE constraint on `keys` so Insert's ON CONFLICT works.
    Reads the connection from data["sqlite"]. Passes df through.
    """

    def __init__(self, table: str, keys: list[str] | str):
        self.table = table
        self.keys = [keys] if isinstance(keys, str) else keys

    def _sqlite_type(self, t: pa.DataType) -> str:
        if pa.types.is_integer(t) or pa.types.is_boolean(t): return "INTEGER"
        if pa.types.is_floating(t): return "REAL"
        if pa.types.is_decimal(t):  return "NUMERIC"
        return "TEXT"                       # dates/timestamps/strings -> TEXT

    def _build_ddl(self, schema: pa.Schema) -> str:
        cols = []
        for f in schema:
            not_null = " NOT NULL" if f.name in self.keys else ""
            cols.append(f'"{f.name}" {self._sqlite_type(f.type)}{not_null}')
        keys = ", ".join(f'"{k}"' for k in self.keys)
        cols.append(f"UNIQUE ({keys})")
        body = ",\n  ".join(cols)
        return f'CREATE TABLE IF NOT EXISTS "{self.table}" (\n  {body}\n)'

    async def apply(self, df: DataFrame, data=None):
        await data["sqlite"].execute(self._build_ddl(df.head(0).to_arrow().schema))
        await data["sqlite"].commit()
        return df

    def __repr__(self):
        return f"EnsureTable(table={self.table!r}, keys={self.keys!r})"
