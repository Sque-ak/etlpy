"""
    Transformer module for PySpark DataFrame transformations.

    Example:

    from pyspark.sql import SparkSession
    from etl.transformer import (
        Pipeline, transformer,
        DropNulls, DropDuplicates, CastTypes,
        FilterRows, AddColumn, TrimStrings, SQLStep,
    )

    spark = SparkSession.builder.appName("bank_etl").getOrCreate()

    @transformer(prefix="bank_clean")
    def transform_bank(df):
        pipe = Pipeline([
            TrimStrings(),
            DropNulls(subset=["id", "amount"]),
            DropDuplicates(subset=["id"]),
            CastTypes({"amount": "double", "date": "timestamp"}),
            FilterRows("amount > 0"),
            AddColumn("tax", "amount * 0.12"),
            SQLStep("SELECT *, amount - tax AS net_amount FROM source"),
        ])
        return pipe, df

    if __name__ == "__main__":
        raw_df = spark.read.parquet("/data/raw/2026-03-02/bank_BerekeBank.parquet")
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

TRANSFORMED_DIR = Path(os.environ.get("TRANSFORMED_DATA_DIR", "/data/stage"))

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
    

def transformer(prefix: str = "", output_dir: str | None = None):
    """
    Decorator for Spark transformation pipelines.

    :param prefix: Optional prefix for the output file name.
    :param output_dir: Directory to save the transformed DataFrame. If None, the DataFrame is returned without saving.

    Example:

        @transformer(prefix="clean", output_dir="/data/stage")
        def transform_bank(df: DataFrame) -> tuple[Pipeline, DataFrame]:
            pipe = Pipeline([...])
            return pipe, df
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            pipe, df = fn(*args, **kwargs)
            result = pipe.run(df, verbose=True)

            if output_dir is None:
                output_dir = TRANSFORMED_DIR

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{prefix}_{timestamp}.parquet" if prefix else f"{timestamp}.parquet"
            output_path = Path(output_dir) / filename
            output_path.parent.mkdir(parents=True, exist_ok=True)

            result.write.mode("overwrite").parquet(str(output_path))
            print(f"Transformed DataFrame saved to: {output_path}")
            return result

        return wrapper
    return decorator