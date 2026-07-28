import polars as pl
from etl.generic.step import Step

class Read(Step):
    """
    Query a MongoDB collection and make the result the pipeline df.

    Source step: reads the database handle from data["mongo"] (set by Connect).
    Two modes: by default runs collection.find(filter, projection); if `pipeline`
    is given, runs collection.aggregate(pipeline) instead (for $lookup joins etc.).
    The incoming df is ignored.

        :param collection: collection name to read from.
        :param filter: query dict, e.g. {"txn_date": {"$gte": date}}. None = all. (find mode)
        :param projection: fields to include/exclude. None defaults to {"_id": 0}. (find mode)
        :param sort: list of (field, direction), e.g. [("txn_date", 1)]. (find mode)
        :param limit: max documents (0 = no limit). (find mode)
        :param pipeline: aggregation pipeline; if set, aggregate() is used and
                         filter/projection/sort/limit are ignored. In this mode YOU own
                         _id handling - convert/drop it, e.g. a final
                         {"$project": {"id": {"$toString": "$_id"}, "_id": 0}},
                         or the raw ObjectId column will break Polars.

        Example (find):
        Read(
            collection="transactions",
            filter={"txn_date": {"$gte": "2026-06-17"}, "bank": "acme"},
            projection={"_id": 0},
        )

        Example (aggregate with joins):
        Read(
            collection="transactions",
            pipeline=[
                {"$lookup": {...}},   # accounts
                {"$lookup": {...}},   # organizations
                {"$project": {"id": {"$toString": "$_id"}, "_id": 0}},
            ],
        )
    """

    def __init__(self, collection, filter=None, projection=None,
                sort=None, limit=0, pipeline=None):
        self.collection = collection
        self.filter = filter or {}
        self.projection = projection if projection else {"_id": 0}
        self.sort = sort
        self.limit = limit
        self.pipeline = pipeline


    async def apply(self, df=None, data=None) -> pl.DataFrame:
        coll = data["mongo"][self.collection]
        if self.pipeline is not None:
            cursor = coll.aggregate(self.pipeline)      
        else:
            cursor = coll.find(self.filter, self.projection)
            if self.sort:
                cursor = cursor.sort(self.sort)
            if self.limit:
                cursor = cursor.limit(self.limit)
        docs = await cursor.to_list()
        return pl.DataFrame(docs, infer_schema_length=None)

    
    def __repr__(self):
        return f"Read(collection={self.collection!r}, filter={self.filter})"