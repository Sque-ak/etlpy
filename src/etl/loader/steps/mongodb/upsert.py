from pymongo import ReplaceOne, UpdateOne
from etl.generic.step import Step
from polars import DataFrame

_OPS = {
    "replace": lambda where, row: ReplaceOne(where, row, upsert=True),
    "update":  lambda where, row: UpdateOne(where, {"$set": row}, upsert=True),
}

class Upsert(Step):
    """
    Idempotently write the df into a MongoDB collection: replace-or-insert each
    document matched by 'keys'. Re-running the same data changes nothing.

    Reads the database handle from data["mongo"].
    The df is passed through unchanged.

        :param collection: target collection.
        :param keys: fields that identify a document (the business key, e.g. ["pk"]).
        :param mode: "replace" - the document becomes the row (ReplaceOne);
                     "update"  - merge the row into the existing document (UpdateOne + $set).
    """

    def __init__(self, collection: str, keys: list[str], mode: str = "replace"):

        if mode not in _OPS:
                    raise ValueError(f"mode must be one of {list(_OPS)}, got {mode!r}")
                    
        self.collection, self.keys, self.mode = collection, keys, mode

    async def apply(self, df: DataFrame, data=None):
        if df is None or df.is_empty():
            return df
        build = _OPS[self.mode]
        ops = [build({k: row[k] for k in self.keys}, row) for row in df.to_dicts()]
        await data["mongo"][self.collection].bulk_write(ops, ordered=False)
        return df



    def __repr__(self):
        return f"Upsert(collection={self.collection!r}, keys={self.keys}, mode={self.mode!r})"
