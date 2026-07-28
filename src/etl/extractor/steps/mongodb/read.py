import polars as pl
from etl.generic.step import Step

class Read(Step):
    """
    Query a MongoDB collection and make the result the pipeline df.

    Source step: reads the database handle from data["mongo"] (set by Connect),
    runs collection.find(filter, projection), returns a Polars DataFrame.
    The incoming df is ignored.

        :param collection: collection name to read from.
        :param filter: MongoDB query dict, e.g. {"txn_date": {"$gte": date}}. None = all.
        :param projection: fields to include/exclude, e.g. {"_id": 0}. None = all fields.
        :param sort: list of (field, direction), e.g. [("txn_date", 1)].
        :param limit: max documents (0 = no limit).

        Example:
        Read(
            collection="transactions",
            filter={"txn_date": {"$gte": "2026-06-17"}, "bank": "acme"},
            projection={"_id": 0},
        )
    """

    def __init__(self, collection: str, filter: dict | None = None,
                 projection: dict | None = None, sort: list | None = None, limit: int = 0):
        self.collection = collection
        self.filter = filter or {}
        self.projection = projection if projection else {"_id": 0}
        self.sort = sort
        self.limit = limit


    async def apply(self, df=None, data=None) -> pl.DataFrame:
        cursor = data["mongo"][self.collection].find(self.filter, self.projection)
        if self.sort:
            cursor = cursor.sort(self.sort)
        if self.limit:
            cursor = cursor.limit(self.limit)
        docs = await cursor.to_list()
        return pl.DataFrame(docs, infer_schema_length=None)
    
    def __repr__(self):
        return f"Read(collection={self.collection!r}, filter={self.filter})"