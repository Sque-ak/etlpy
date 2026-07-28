from etl.generic.step import Step
from polars import DataFrame


class Insert(Step):
    """
    Idempotently upsert the df via INSERT ... ON CONFLICT (keys) DO UPDATE.
    Reads the connection from data["sqlite"].
    """

    def __init__(self, table: str, keys: list[str] | str):
        self.table = table
        self.keys = [keys] if isinstance(keys, str) else keys

    async def apply(self, df: DataFrame, data=None):
        if df is None or df.is_empty():
            return df
        cols = df.columns
        col_list = ", ".join(f'"{c}"' for c in cols)
        placeholders = ", ".join("?" for _ in cols)
        conflict = ", ".join(f'"{k}"' for k in self.keys)
        updates = ", ".join(f'"{c}" = excluded."{c}"' for c in cols if c not in self.keys)
        action = f"DO UPDATE SET {updates}" if updates else "DO NOTHING"

        sql = (f'INSERT INTO "{self.table}" ({col_list}) VALUES ({placeholders}) '
               f'ON CONFLICT ({conflict}) {action}')
        await data["sqlite"].executemany(sql, df.rows())
        await data["sqlite"].commit()
        return df

    def __repr__(self):
        return f"Insert(table={self.table!r}, keys={self.keys!r})"
