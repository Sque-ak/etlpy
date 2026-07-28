from etl.generic import Step, StopPipeline
import polars as pl

from datetime import datetime, date
from typing import get_args, get_origin, Union
import types


_PL_TYPES = {
    str: pl.String,
    int: pl.Int64,
    float: pl.Float64,
    bool: pl.Boolean,
    datetime: pl.Datetime("us"),
    date: pl.Date,
}

class ToSchema(Step):
    """
    Validate and coerce the df against a Pydantic model, row by row.

    Slow by design (per-row validation) but it is the guarantee that downstream
    gets exactly the expected schema and types. Speed is recovered in ClickHouse.

    NOTE: this step does not import pydantic - it only uses the model you pass,
    so pydantic stays an optional, user-side dependency.

        :param model: a Pydantic v2 (or Patito) model class defining the canon.
        :param mapping: optional {raw_field: canon_field} rename map.
    """

    def __init__(self, model, mapping: dict[str, str] | None = None):
        self.model = model           
        self.mapping = mapping or {}

    async def apply(self, df, data = None):
        if df is None or df.is_empty():
            raise StopPipeline(message="df is empty")
        
        if self.mapping:
            df = df.rename({s: d for s, d in self.mapping.items() if s in df.columns})

        canon = set(self.model.model_fields)
        df = df.select([c for c in df.columns if c in canon])

        rows = [self.model(**row).model_dump() for row in df.iter_rows(named=True)]
        return pl.DataFrame(rows, schema=_model_schema(self.model))

    def __repr__(self) -> str:
        return f"ToSchema(model='{self.model!r}', mapping='{self.mapping!r}')"
    
def _pl_type(t):
    """Type pydantic-field -> type polars. list[str] -> List(String)."""
    if get_origin(t) is list:
        inner = get_args(t)[0] if get_args(t) else str
        return pl.List(_pl_type(inner))
    return _PL_TYPES.get(t, pl.String)

def _model_schema(model) -> dict:
    schema = {}
    for name, field in model.model_fields.items():
        ann = field.annotation
        if get_origin(ann) in (Union, types.UnionType):
            variants = [a for a in get_args(ann) if a is not type(None)]
            ann = variants[0] if variants else ann
        schema[name] = _pl_type(ann)
    return schema
