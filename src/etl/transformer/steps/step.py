from __future__ import annotations
from abc import ABC, abstractmethod
from pyspark.sql import DataFrame

class Step(ABC):
    """Abstract base class for transformation steps."""

    @abstractmethod
    def apply(self, df: DataFrame) -> DataFrame:
        """
        Logic to transform the data.

        You can return any data structure that is convenient for you,
        after you can transform it to a pandas DataFrame.
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}()"