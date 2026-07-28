import pyarrow as pa
from etl.generic.step import Step
from polars import DataFrame

class EnsureTable(Step):
    """
    Create the SQL Server table from the df schema if it does not exist,
    with a UNIQUE constraint on `keys` so Insert's MERGE matches correctly.
    Reads the connection from data["mssql"]. Passes df through.
    """

    def __init__(self, table: str, keys: list[str] | str):
        self.table = table
        self.keys = [keys] if isinstance(keys, str) else keys

    def _mssql_type(self, t: pa.DataType) -> str:
        if pa.types.is_int8(t):      return "TINYINT"
        if pa.types.is_int16(t):     return "SMALLINT"
        if pa.types.is_int32(t):     return "INT"
        if pa.types.is_int64(t):     return "BIGINT"
        if pa.types.is_float32(t) or pa.types.is_float16(t): return "REAL"
        if pa.types.is_float64(t):   return "FLOAT"
        if pa.types.is_boolean(t):   return "BIT"
        if pa.types.is_date(t):      return "DATE"
        if pa.types.is_timestamp(t): return "DATETIME2"
        if pa.types.is_decimal(t):   return f"DECIMAL({t.precision}, {t.scale})"
        return "NVARCHAR(MAX)"

    def _build_ddl(self, schema: pa.Schema) -> str:
        cols = []
        for f in schema:
            not_null = " NOT NULL" if f.name in self.keys else ""
            cols.append(f"[{f.name}] {self._mssql_type(f.type)}{not_null}")
        keys = ", ".join(f"[{k}]" for k in self.keys)
        cols.append(f"CONSTRAINT [UQ_{self.table}] UNIQUE ({keys})")
        body = ",\n  ".join(cols)
        return (f"IF OBJECT_ID(N'{self.table}', N'U') IS NULL\n"
                f"CREATE TABLE {self.table} (\n  {body}\n);")

    async def apply(self, df: DataFrame, data=None):
        async with data["mssql"].cursor() as cur:
            await cur.execute(self._build_ddl(df.head(0).to_arrow().schema))
        return df

    def __repr__(self):
        return f"EnsureTable(table={self.table!r}, keys={self.keys!r})"