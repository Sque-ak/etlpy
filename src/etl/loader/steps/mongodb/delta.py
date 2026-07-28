import polars as pl
from etl.generic.step import Step

class Delta(Step):
    """
    Keep only new or changed rows by comparing 'row_hash' against MongoDB.

    A row is kept when its key is absent (new) or its row_hash differs.
    Reads the database handle from data["mongo"]; needs a 'row_hash' column (RowHash).

        :param collection: target collection
        :param keys: key field(s) identifying a document across loads
    """

    def __init__(self, collection: str, keys: list[str] | str):
        self.collection = collection
        self.keys = [keys] if isinstance(keys, str) else keys

    async def apply(self, df: pl.DataFrame, data=None):
        if df is None or df.is_empty():
            return df

        projection = {key: 1 for key in self.keys} | {"row_hash": 1, "_id": 0}
        docs = await data["mongo"][self.collection].find({}, projection).to_list()
        existing = pl.DataFrame(docs, infer_schema_length=None)
        if existing.is_empty():
            return df                       

        existing = existing.cast({key: df.schema[key] for key in self.keys})  
        delta = (
            df.join(existing, on=self.keys, how="left", suffix="_old")
            .filter(pl.col("row_hash_old").is_null() | (pl.col("row_hash") != pl.col("row_hash_old")))
            .drop("row_hash_old")
        )

        if (skipped := df.height - delta.height) > 0:
            print(f"Skipped {skipped} unchanged rows for {self.collection}")

        return delta

    def __repr__(self) -> str:
        return f"Delta(collection={self.collection!r}, keys={self.keys!r})"
