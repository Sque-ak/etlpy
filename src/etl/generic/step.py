from __future__ import annotations
from abc import ABC, abstractmethod

from etl.generic.context import Data
from polars import DataFrame, LazyFrame


class Step(ABC):
    """
    Abstract base class for extraction steps.
    """

    @abstractmethod
    async def apply(self, df: DataFrame | LazyFrame | None, data: Data|None = None) -> DataFrame | LazyFrame | None:
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"