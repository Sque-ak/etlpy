"""
    Transformer module for PySpark DataFrame transformations.

    Example:

    from pyspark.sql import SparkSession
    from etl.storage import Storage
    from etl.transformer import Pipeline, transformer, DropNulls, CastTypes, FilterRows

    spark = SparkSession.builder.appName("bank_etl").getOrCreate()
    storage = Storage(base_dir="/data")

    @transformer(prefix="bank_clean", storage=storage)
    def transform_bank(df):
        pipe = Pipeline([
            DropNulls(subset=["id", "amount"]),
            CastTypes({"amount": "double", "date": "timestamp"}),
            FilterRows("amount > 0"),
        ])
        return pipe, df

    if __name__ == "__main__":
        raw_df = spark.read.parquet(str(storage.path("raw", "bank_BerekeBank.parquet")))
        clean_df = transform_bank(raw_df)
        clean_df.show()
        spark.stop()
"""

from __future__ import annotations
from abc import ABC
import functools, os
from typing import Callable
from pyspark.sql import DataFrame
from pathlib import Path
from datetime import datetime

from etl.transformer.steps.step import Step
from etl.storage import Storage


class Pipeline(ABC):
    """
    Chain of transformation steps on PySpark DataFrame.

    :param steps: List of transformation steps to apply in sequence.

    Example:

        pipe = Pipeline([
            DropNulls(subset=["id"]),
            CastTypes({"amount": "double"}),
            FilterRows("amount > 0"),
            AddColumn("tax", "amount * 0.12"),
        ])
        result = pipe.run(spark_df)

    """

    def __init__(self, steps: list[Step] | None = None):
        self.steps: list[Step] = steps or []

    def add(self, step: Step) -> Pipeline:
        """Add a transformation step to the pipeline."""
        self.steps.append(step)
        return self
    
    def run(self, df: DataFrame, verbose: bool = False) -> DataFrame:
        """Run the pipeline on the input DataFrame."""
        result = df
        for i, step in enumerate(self.steps):
            rows_before = result.count() if verbose else 0
            result = step.apply(result)
            if verbose:
                rows_after = result.count()
                print(f"  [{i + 1}/{len(self.steps)}] {step!r}: {rows_before} → {rows_after} rows")
        return result
     
    def __repr__(self) -> str:
        steps_repr = "\n  ".join(repr(step) for step in self.steps)
        return f"Pipeline(steps=[\n  {steps_repr}\n])"
    

def transformer(
    prefix: str = "",     
    storage: Storage | None = None,
    layer: str = "stage",
    format: str = "parquet",
    save: bool = True
    ) -> Callable:
    """
    Decorator for Spark transformation pipelines.

    Args:
        prefix: Optional prefix for the output file name.
        storage: Storage instance. If None, uses default Storage().
        layer: Target storage layer to save results. Default: "stage".
        format: Output format. Default: "parquet".
        save: Whether to save the result. Default: True.

    Example:

        storage = Storage(base_dir="/data")

        @transformer(prefix="clean", storage=storage, layer="stage")
        def transform_bank(df):
            pipe = Pipeline([...])
            return pipe, df

        @transformer(prefix="report", storage=storage, layer="processed")
        def aggregate_bank(df):
            pipe = Pipeline([...])
            return pipe, df
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            pipe, df = fn(*args, **kwargs)
            result = pipe.run(df, verbose=True)

            if save:
                _storage = storage or Storage()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{prefix}_{timestamp}" if prefix else f"{fn.__name__}_{timestamp}"
                
                output_path = _storage.path(layer, filename)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                result.write.mode("overwrite").format(format).save(str(output_path))
                print(f"Saved to {output_path}")

            return result

        return wrapper
    return decorator