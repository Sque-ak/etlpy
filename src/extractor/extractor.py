from abc import ABC, abstractmethod
from typing import Callable
from pathlib import Path
import pandas, functools, os, datetime

RAW_DIR = Path(os.environ.get("RAW_DATA_DIR", "/data/raw"))

class Source(ABC):

    folder = RAW_DIR

    @abstractmethod
    def extract(self):
        """
        Logic to get data from the API service.

        You can return any data structure that is convenient for you,
        after you can transform it to a pandas DataFrame.
        """
        pass

def extractor(prefix: str = "") -> Callable:
    """
    Decorator that wraps a pipeline function returning API sources.
    Usage:
        @extractor(prefix="BankData")
        def pipeline():
            return [AnyBank(url="..."), OtherAPI(url="...")]
    """

    def decorator(fn: Callable) -> Callable[..., any]:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> any:
            sources = fn(*args, **kwargs)
            results: dict[str, pandas.DataFrame] = {}

            # Support both list and dict of sources
            if isinstance(sources, list):
                source_map = {type(s).__name__: s for s in sources}
            elif isinstance(sources, dict):
                source_map = sources
            elif isinstance(sources, Source):
                source_map = {type(sources).__name__: sources}
            else:
                raise TypeError(
                    f"Expected Source, list, or dict, got {type(sources).__name__}"
                )

            for name, source in source_map.items():
                folder = Path(f"{source.folder}/{datetime.datetime.now().strftime('%Y-%m-%d')}")
                folder.mkdir(parents=True, exist_ok=True)

                filename = f"{prefix}_{name}" if prefix else name
                dataframe = pandas.DataFrame(source.extract())
                dataframe.to_parquet(folder / f"{filename}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.parquet", index=False)

                results[name] = dataframe

            return results

        return wrapper
    return decorator