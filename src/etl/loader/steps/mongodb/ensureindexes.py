from etl.generic.step import Step
from polars import DataFrame

class EnsureIndexes(Step):
    """
    Reads the database handle from data["mongo"] (set by Connect). Passes df through.
    create_index is idempotent - re-running is a no-op if the index already exists.

        :param collection: target collection.
        :param keys: field(s) forming the unique business key (e.g. ["pk"]).
        :param indexes: extra single-field indexes to ensure (default ["row_hash"]).
    """

    def __init__(self, collection: str, keys: list[str] | str, indexes: list[str] | None = None):
        self.collection = collection
        self.keys = [keys] if isinstance(keys, str) else keys
        self.indexes = indexes or ["row_hash"]

    async def apply(self, df: DataFrame = None, data=None):
        coll = data["mongo"][self.collection]
        await coll.create_index([(k, 1) for k in self.keys], unique=True)
        for field in self.indexes:
            await coll.create_index(field)
        return df

    def __repr__(self):
        return f"EnsureIndexes(collection={self.collection!r}, keys={self.keys!r})"
