from etl.generic.step import Step
import polars as pl

class SQL(Step):
    """
        Run a SQL query against the incoming DataFrame (Polars SQL dialect).

        :param query: SQL text; reference the frame by 'view_name'.
        :param view_name: table name the frame is registered under (default "source").

        Note: Polars SQL dialect.

        Example:
            SQL("SELECT name, age FROM source WHERE age > 30")
            or
            SQL.from_file("queries/enrich.sql")
    """

    def __init__(self, query: str, view_name: str = "source"):
        self.query = query
        self.view_name = view_name

    @classmethod
    def from_file(cls, path:str, view_name: str = "source"):
        from pathlib import Path
        return cls(Path(path).read_text(encoding="utf-8"), view_name)

    async def apply(self, df: pl.DataFrame, data = None):
        return pl.SQLContext({self.view_name: df}).execute(self.query, eager=True)

    
    def __repr__(self) -> str:
        return f"SQLStep(query='{self.query}', view_name='{self.view_name}')"