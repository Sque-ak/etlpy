from __future__ import annotations
from abc import ABC, abstractmethod

from etl.generic.context import Data
from polars import DataFrame, LazyFrame

class StopPipeline(Exception):
    """Raised by a step to halt the pipeline gracefully (not an error)."""
    def __init__(self, message: str = "", df=None):
        super().__init__(message)
        self.df = df

class Step(ABC):
    """
    Abstract base class for extraction steps.
    """

    @abstractmethod
    async def apply(self, df: DataFrame | LazyFrame | None, data: Data|None = None) -> DataFrame | LazyFrame | None:
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"