from etl.generic.step import Step
from polars import DataFrame


class Insert(Step):
    """
    Idempotently upsert the df via MERGE, matched by 'keys'.
    WITH (HOLDLOCK) makes concurrent merges safe. Needs a UNIQUE constraint on
    keys (see EnsureTable). Reads the connection from data["mssql"].
    """

    def __init__(self, table: str, keys: list[str] | str):
        self.table = table
        self.keys = [keys] if isinstance(keys, str) else keys

    def _build_merge(self, cols: list[str]) -> str:
        col_list = ", ".join(f"[{c}]" for c in cols)
        placeholders = ", ".join("?" for _ in cols)
        on = " AND ".join(f"tgt.[{k}] = src.[{k}]" for k in self.keys)
        updates = ", ".join(f"tgt.[{c}] = src.[{c}]" for c in cols if c not in self.keys)
        insert_vals = ", ".join(f"src.[{c}]" for c in cols)
        matched = f"WHEN MATCHED THEN UPDATE SET {updates} " if updates else ""
        return (
            f"MERGE INTO {self.table} WITH (HOLDLOCK) AS tgt "
            f"USING (VALUES ({placeholders})) AS src ({col_list}) "
            f"ON {on} "
            f"{matched}"
            f"WHEN NOT MATCHED THEN INSERT ({col_list}) VALUES ({insert_vals});"
        )

    async def apply(self, df: DataFrame, data=None):
        if df is None or df.is_empty():
            return df
        sql = self._build_merge(df.columns)
        async with data["mssql"].cursor() as cur:
            await cur.executemany(sql, df.rows())
        return df

    def __repr__(self):
        return f"Insert(table={self.table!r}, keys={self.keys!r})"