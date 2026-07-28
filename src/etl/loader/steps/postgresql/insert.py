from etl.generic.step import Step
from polars import DataFrame

class Insert(Step):
    """
    Write the df via INSERT ... ON CONFLICT (keys) DO UPDATE.
    Re-running the same data changes nothing. Reads the connection from data["pg"].

        :param table: target table.
        :param keys: conflict-target columns (need a UNIQUE constraint - see EnsureTable).
    """

    def __init__(self, table: str, keys: list[str] | str):
        self.table = table
        self.keys = [keys] if isinstance(keys, str) else keys

    async def apply(self, df: DataFrame, data=None):
        if df is None or df.is_empty():
            return df

        cols = df.columns
        col_list = ", ".join(f'"{c}"' for c in cols)
        values = ", ".join(f"${i}" for i in range(1, len(cols) + 1))
        conflict = ", ".join(f'"{k}"' for k in self.keys)
        updates = ", ".join(f'"{c}" = EXCLUDED."{c}"' for c in cols if c not in self.keys)
        action = f"DO UPDATE SET {updates}" if updates else "DO NOTHING"

        sql = (f'INSERT INTO {self.table} ({col_list}) VALUES ({values}) '
               f'ON CONFLICT ({conflict}) {action}')
        await data["pg"].executemany(sql, df.rows())
        return df

    def __repr__(self):
        return f"Insert(table={self.table!r}, keys={self.keys!r})"
