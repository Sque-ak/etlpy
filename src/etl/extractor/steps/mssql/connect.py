from etl.generic.step import Step
from polars import DataFrame

class Connect(Step):
    """
    Open a Microsoft SQL Server connection and put it into data["mssql"].
    Requires the Microsoft ODBC Driver for SQL Server installed on the host,
    and the 'aioodbc' package (pip install aioodbc).

        :param host/port/database/user/password: connection params.
        :param driver: installed ODBC driver name.
        :param kwargs: extra ODBC keywords appended to the connection string
                       (e.g. TrustServerCertificate="yes", Encrypt="no").
    """

    def __init__(self, host="localhost", port=1433, database="master",
                 user="sa", password="", driver="ODBC Driver 18 for SQL Server", **kwargs):
        self.host, self.database = host, database
        extra = "".join(f"{k}={v};" for k, v in kwargs.items())
        self.dsn = (f"Driver={{{driver}}};Server={host},{port};Database={database};"
                    f"UID={user};PWD={password};{extra}")

    async def apply(self, df: DataFrame = None, data=None):
        import aioodbc
        data["mssql"] = await aioodbc.connect(dsn=self.dsn, autocommit=True)
        return df

    def __repr__(self):
        return f"ConnectMSSQL(host={self.host!r}, database={self.database!r})"
