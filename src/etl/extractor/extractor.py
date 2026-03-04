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

def extract_sources(sources: list[Source] | dict[str, Source] | Source, prefix: str = "") -> dict[str, pandas.DataFrame]:
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

        dataframe = pandas.DataFrame(source.extract())
        filename = f"{prefix}_{name}" if prefix else name
        dataframe.to_parquet(folder / f"{filename}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.parquet", index=False)

        results[name] = dataframe

    return results

def extract_json(
    json_data: dict | list,
    prefix: str = "",
    flatten: dict[str, str | list[str]] | None = None,
    save: bool = True,
) -> dict[str, pandas.DataFrame]:
    """
    Extract JSON data into multiple DataFrames, flattening nested arrays.

    Args:
        json_data: The input JSON data (dict or list of dicts).
        prefix: Optional prefix for saved file names.
        flatten: Dict mapping output DataFrame names to JSON paths (dot-separated). 
                 If None, the entire JSON is flattened into a single DataFrame under "main".
        save: Whether to save the extracted DataFrames as Parquet files in the RAW_DIR.

    Returns:
        Dict of DataFrames: {"main": ..., "transactions": ..., ...}

    Example:
        JSON structure:
        {
            "name": "John",
            "amount": 1000,
            "account": "KZ123",
            "balance": [
                {
                    "currency": "KZT",
                    "total": 500000,
                    "transactions": [
                        {"id": 1, "sum": 1000, "date": "2026-01-01"},
                        {"id": 2, "sum": 2000, "date": "2026-01-02"},
                    ]
                }
            ]
        }

        result = extract_json(
            json_data=response.json(),
            prefix="bereke",
            flatten={
                "balance": "balance",
                "transactions": "balance.transactions",
            },
        )

        result["main"]          # name, amount, account (without balance)
        result["balance"]       # currency, total (without transactions)
        result["transactions"]  # id, sum, date
    """
    
    if isinstance(json_data, dict):
        json_data = [json_data]

    results: dict[str, pandas.DataFrame] = {}

    if not flatten:
        results["main"] = pandas.json_normalize(json_data)
    else:
        for output_name, path in flatten.items():
            parts = path.split(".")
            extracted_rows = []

            for record in json_data:
                _extract_nested(record, parts, 0, extracted_rows)

            if extracted_rows:
                results[output_name] = pandas.json_normalize(extracted_rows)
            else:
                results[output_name] = pandas.DataFrame()

        top_level_array_keys = set()
        for path in flatten.values():
            top_level_array_keys.add(path.split(".")[0])

        main_rows = []
        for record in json_data:
            flat = {k: v for k, v in record.items() if k not in top_level_array_keys}
            main_rows.append(flat)

        results["main"] = pandas.json_normalize(main_rows) if main_rows else pandas.DataFrame()

    if save and prefix:
        folder = Path(f"{RAW_DIR}/{datetime.datetime.now().strftime('%Y-%m-%d')}")
        folder.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        for name, df in results.items():
            if not df.empty:
                filename = f"{prefix}_{name}_{timestamp}.parquet"
                df.to_parquet(folder / filename, index=False)

    return results
   
def _extract_nested(record: dict, parts: list[str], depth: int, output: list) -> None:
    """
    Recursively extract nested arrays from a JSON record.

    Args:
        record: Current dict to look into.
        parts: Path parts, e.g. ["balance", "transactions"].
        depth: Current depth in the path.
        output: List to append extracted rows to.
    """
    if depth >= len(parts):
        # Remove nested lists/dicts to keep it flat for this level
        flat = {k: v for k, v in record.items() if not isinstance(v, (list, dict))}
        output.append(flat)
        return

    key = parts[depth]

    if key not in record:
        return

    value = record[key]

    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                _extract_nested(item, parts, depth + 1, output)
    elif isinstance(value, dict):
        _extract_nested(value, parts, depth + 1, output)

def extractor(prefix: str = "") -> Callable:
    """
    Decorator that wraps a pipeline function returning API sources.
    
    Example:
        @extractor(prefix="BankData")
        def pipeline():
            return [AnyBank(url="..."), OtherAPI(url="...")]
    """

    def decorator(fn: Callable) -> Callable[..., any]:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> any:
            return extract_sources(fn(*args, **kwargs), prefix=prefix)

        return wrapper
    return decorator