from etl.generic.step import Step
from polars import DataFrame

class Connect(Step):
    """
    Open a MongoDB connection and put the database handle into data["mongo"]
    for downstream steps (Read, EnsureIndexes, Upsert, Delta).

    Usually the first step of a MongoDB pipeline.

        :param host/port/database/username/password: pymongo connection params.
        :param kwargs: extra args forwarded to AsyncMongoClient (authSource, tls, replicaSet, ...).
    """

    def __init__(self, host="localhost", port=27017, database="admin",
                 username=None, password=None, **kwargs):
        self.database = database
        self.config = dict(host=host, port=port, username=username, password=password, **kwargs)

    async def apply(self, df: DataFrame, data=None):
        from pymongo import AsyncMongoClient

        client = AsyncMongoClient(**self.config)
        data["mongo"] = client[self.database]
        return df
    
    def __repr__(self):
        return f"ConnectMongo(host={self.config['host']!r}, database={self.database!r})"
