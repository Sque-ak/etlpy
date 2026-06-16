import asyncio, re, pyarrow as pa
from etl.generic.step import Step
from polars import DataFrame


class EnsureTable(Step):
    """
    Create the ClickHouse table from the incoming df schema if it does not exist.

    Reads the client from data["ch"]; passes df through unchanged. ORDER BY keys,
    the ReplacingMergeTree (see https://clickhouse.com/docs/engines/table-engines/mergetree-family/replacingmergetree)
    version column and 'row_hash' are non-nullable; other non-string columns become Nullabe.

        :param table: target table name.
        :param engine: ClickHouse engine (default "MergeTree" see https://clickhouse.com/docs/engines/table-engines/mergetree-family/mergetree)
        :param partition_by: optional PARTITION BY expression.
        :param if_exists: "append"/"replace" -> create if missing; "error" -> raise if exists.
    """

    def __init__(self, table, order_by, engine="MergeTree", partition_by=None, if_exists="append"):
        self.table = table
        self.engine = engine
        self.order_by = order_by
        self.partition_by = partition_by
        self.if_exists = if_exists

    def _arrow_to_ch(self, t: pa.DataType) -> str:
        """Map a PyArrow type to a ClickHouse type string."""
        if pa.types.is_int8(t):    return "Int8"
        if pa.types.is_int16(t):   return "Int16"
        if pa.types.is_int32(t):   return "Int32"
        if pa.types.is_int64(t):   return "Int64"
        if pa.types.is_uint8(t):   return "UInt8"
        if pa.types.is_uint16(t):  return "UInt16"
        if pa.types.is_uint32(t):  return "UInt32"
        if pa.types.is_uint64(t):  return "UInt64"
        if pa.types.is_float16(t) or pa.types.is_float32(t):  return "Float32"
        if pa.types.is_float64(t):  return "Float64"
        if pa.types.is_boolean(t):  return "Bool"
        if pa.types.is_string(t) or pa.types.is_large_string(t):  return "String"
        if pa.types.is_date(t):     return "Date"
        if pa.types.is_timestamp(t):  return "DateTime64(3)"
        if pa.types.is_decimal(t):  return f"Decimal({t.precision}, {t.scale})"
        if pa.types.is_binary(t) or pa.types.is_large_binary(t):  return "String"
        if pa.types.is_list(t) or pa.types.is_large_list(t): 
            return f"Array({self._arrow_to_ch(t.value_type)})"
        return "String" 

    async def apply(self, df:DataFrame, data = None):
        ch = data["ch"]
        if await asyncio.to_thread(ch.command, f"EXISTS TABLE {self.table}"):
            if self.if_exists == "error":
                raise ValueError(f"Table {self.table} already exists")
            return df
        ddl = self._build_ddl(df.head(0).to_arrow().schema) # schema only, no data
        await asyncio.to_thread(ch.command, ddl)
        return df
    
    def _build_ddl(self, schema: pa.Schema) -> str:
        non_nullable = set(self.order_by) | {"row_hash"}
        version = re.search(r"ReplacingMergeTree\((\w+)\)", self.engine)
        if version:
            non_nullable.add(version.group(1))

        columns = []
        for f in schema:
            ch_type = self._arrow_to_ch(f.type)
            if f.nullable and ch_type != "String" and f.name not in non_nullable:
                ch_type = f"Nullable({ch_type})"
            columns.append(f"    `{f.name}` {ch_type}")

        body = ",\n".join(columns)
        order_clause = ", ".join(F"`{column}`" for column in self.order_by) if self.order_by else "tuple()"
        partition = f"\nPARTITION BY {self.partition_by}" if self.partition_by else ""
        return (
            f"CREATE TABLE {self.table} (\n{body}\n)"
            f"ENGINE = {self.engine}\nORDER BY ({order_clause}{partition})"
        )

    def __repr__(self):
        return f"EnsureTable(table={self.table!r}, engine={self.engine!r})"
