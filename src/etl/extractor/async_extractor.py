from abc import ABC, abstractmethod
from typing import Callable
import asyncio, pandas, functools, datetime, logging
from etl.storage import Storage

logger = logging.getLogger(__name__)


class AsyncSource(ABC):
    """Base class for all async data sources."""

    @abstractmethod
    async def extract(self):
        """
        Async logic to get data from the source.

        You can return any data structure that is convenient for you,
        after you can transform it to a pandas DataFrame.
        """
        pass


async def async_extract_sources(
    sources: list[AsyncSource] | dict[str, AsyncSource] | AsyncSource,
    prefix: str = "",
    storage: Storage | None = None,
    max_concurrent: int = 5,
) -> dict[str, pandas.DataFrame]:
    """
    Extract data from async sources concurrently with a concurrency limit.

    Args:
        sources: AsyncSource or list/dict of AsyncSources.
        prefix: Optional prefix for file names.
        storage: Storage instance. If None, uses default Storage().
        max_concurrent: Maximum number of simultaneous extractions.

    Example::

        results = await async_extract_sources(
            sources=[BankAPI(...), OtherAPI(...), ThirdAPI(...)],
            prefix="daily",
            max_concurrent=3,   # no more than 3 at a time
        )
    """
    storage = storage or Storage()

    if isinstance(sources, list):
        source_map = {type(s).__name__: s for s in sources}
    elif isinstance(sources, dict):
        source_map = sources
    elif isinstance(sources, AsyncSource):
        source_map = {type(sources).__name__: sources}
    else:
        raise TypeError(
            f"Expected AsyncSource, list, or dict, got {type(sources).__name__}"
        )

    semaphore = asyncio.Semaphore(max_concurrent)
    results: dict[str, pandas.DataFrame] = {}
    lock = asyncio.Lock()

    async def _run(name: str, source: AsyncSource) -> None:
        async with semaphore:
            logger.info("Starting extraction: %s", name)
            try:
                data = await source.extract()
                dataframe = pandas.DataFrame(data)

                filename = f"{prefix}_{name}" if prefix else name
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                storage.write("raw", dataframe, f"{filename}_{timestamp}.parquet")

                async with lock:
                    results[name] = dataframe

                logger.info("Finished extraction: %s (%d rows)", name, len(dataframe))
            except Exception:
                logger.exception("Failed extraction: %s", name)
                raise

    tasks = [asyncio.create_task(_run(name, source)) for name, source in source_map.items()]
    await asyncio.gather(*tasks)

    return results


def async_extractor(
    prefix: str = "",
    storage: Storage | None = None,
    max_concurrent: int = 5,
) -> Callable:
    """
    Decorator that wraps an async pipeline function returning async sources.

    Example::

        @async_extractor(prefix="BankData", max_concurrent=3)
        async def pipeline():
            return [BankAPI(url="..."), OtherAPI(url="...")]
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            return await async_extract_sources(
                await fn(*args, **kwargs),
                prefix=prefix,
                storage=storage,
                max_concurrent=max_concurrent,
            )

        return wrapper

    return decorator
