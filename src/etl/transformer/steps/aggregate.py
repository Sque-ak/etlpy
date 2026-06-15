from etl.transformer.steps.step import Step
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

class Aggregate(Step):
    """
    Perform group by and aggregation on the DataFrame.

    Args:
        group_by: Columns to group by.
        aggregations: {"column": "agg_func"} or {"column": ["agg1", "agg2"]}
            Supported: sum, avg, mean, count, min, max, first, last, collect_list, collect_set

    Example:
        
        Aggregate(
            group_by=["bank", "currency"],
            aggregations={
                "amount": ["sum", "avg", "count"],
                "date": "max",
            }
        )

    """

    def __init__(self, group_by: list[str], aggregations: dict[str, str | list[str]]):
        self.group_by = group_by
        self.aggregations = aggregations

    def apply(self, df: DataFrame) -> DataFrame:

        agg_funcs = {
            "sum": F.sum,
            "avg": F.avg,
            "mean": F.mean,
            "count": F.count,
            "min": F.min,
            "max": F.max,
            "first": F.first,
            "last": F.last,
            "collect_list": F.collect_list,
            "collect_set": F.collect_set,
            "stddev": F.stddev,
            "variance": F.variance,
        }

        exprs = []
        for col_name, funcs in self.aggregations.items():
            if isinstance(funcs, str):
                funcs = [funcs]
            for func_name in funcs:
                if func_name not in agg_funcs:
                    raise ValueError(f"Unknown aggregation: '{func_name}'. Supported: {list(agg_funcs.keys())}")
                alias = f"{col_name}_{func_name}"
                exprs.append(agg_funcs[func_name](col_name).alias(alias))

        return df.groupBy(*self.group_by).agg(*exprs)

    def __repr__(self) -> str:
        return f"Aggregate(group_by={self.group_by}, aggregations={self.aggregations})"
