from etl.generic.step import Step
from polars import DataFrame

class RenameColumns(Step):
    """
        Rename columns from a mapping.

        :param columns_mapping: {old_name: new_name}

        Example:
            RenameColumns(columns_mapping={'name': 'full_name', 'email': 'contact_email'})
    """

    def __init__(self, columns_mapping: dict[str, str]):
        self.columns_mapping = columns_mapping

    async def apply(self, df: DataFrame, data = None):
        return df.rename(self.columns_mapping)

        
    def __repr__(self) -> str:
        return f"RenameColumns(columns_mapping={self.columns_mapping})"