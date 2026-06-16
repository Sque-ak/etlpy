from etl.generic.step import Step
from polars import DataFrame, col

_AGG = {
    "sum":          lambda c: col(c).sum(),
    "avg":          lambda c: col(c).mean(),
    "count":        lambda c: col(c).count(),
    "min":          lambda c: col(c).min(),
    "max":          lambda c: col(c).max(),
    "first":        lambda c: col(c).first(),
    "last":         lambda c: col(c).last(),
    "collect_list": lambda c: col(c),
    "collect_set":  lambda c: col(c).unique(),
    "stddev":       lambda c: col(c).std(),
    "variance":     lambda c: col(c).var(),
}

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

    async def apply(self, df: DataFrame, data = None):
        exprs = []
        for col_name, funcs in self.aggregations.items():
            if isinstance(funcs, str):
                funcs = [funcs]
            for func_name in funcs:
                if func_name not in _AGG:
                    raise ValueError(f"Unknown aggregation: '{func_name}'. Supported: {list(_AGG.keys())}")
                exprs.append(_AGG[func_name](col_name).alias(f"{col_name}_{func_name}"))
        return df.group_by(self.group_by).agg(exprs)

    def __repr__(self) -> str:
        return f"Aggregate(group_by={self.group_by}, aggregations={self.aggregations})"
