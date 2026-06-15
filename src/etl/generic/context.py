from __future__ import annotations
from typing import Any
from polars import DataFrame, LazyFrame

class Data:
    '''
    General data within the pipline; all steps communicate thought this context.

    Anti-pattern: Only simple data should be stored here; heavy dataframes must not be placed here.
        Solution: Extract dataframe to data lake and read them frome there. 
    This is required for auditing raw data.
    '''

    def __init__(self, **values: Any) -> None:
        self.store: dict[str, Any] = dict(values)

    def get(self, key: str, default: Any = None) -> Any:
        return self.store.get(key, default)
    
    def __getitem__(self, key: str) -> Any:
        return self.store[key]
    
    def __setitem__(self, key:str, value: Any) -> None:
        if isinstance(value, (DataFrame, LazyFrame)):
            raise TypeError(
                f"Data['{key}'] is Dataframe is antipattern," 
                f"should be exchanged between steps via raw exports.")

        self.store[key] = value
    
    def __contains__(self, key: str) -> bool:
        return key in self.store
    
    def __repr__(self) -> str:
        body = ", ".join(f"{key}: {type(value).__name__}" for key,value in self.store.items())
        return f"Data({body})"