from __future__ import annotations
import functools
from typing import Callable

from polars import DataFrame, LazyFrame
from etl.generic.context import Data
from etl.generic.step import Step, StopPipeline


class Pipeline:
    """
    Chain of async steps.

    Example: 
        pipe = Pipeline([Authenticate(api), GetAccounts(), GetTransactions()])
        data = await pipe.run()
    """

    def __init__(
            self,
            steps: list[Step] | None = None,  
            dataframe: DataFrame | LazyFrame | None = None, 
            data: Data | None = None
            ) -> None:
        self.steps: list[Step] = steps or []
        self.data = data or Data()
        self.dataframe = dataframe

    def add(self, step: Step) -> "Pipeline":
        self.steps.append(step)
        return self
    
    async def run(self, verbose: bool = False) -> DataFrame | LazyFrame | None:
        for i, step in enumerate(self.steps):

            try:
                self.dataframe = await step.apply(self.dataframe, self.data)
            except StopPipeline as stop:
                if verbose:
                    print(f" [stop] {step!r}: {stop}")
                return stop.df if stop.df is not None else self.dataframe
            except Exception as error:
                error.add_note(f"Pipeline failed at step [{i + 1}/{len(self.steps)}] - {step!r}")
                raise                       

            if verbose:
                print(f" [{i + 1}/{len(self.steps)}] {step!r}")
        
        return self.dataframe
    
    def __repr__(self) -> str:
        body = "\n ".join(repr(s) for s in self.steps)
        return f"Pipeline(steps=[\n {body} \n])"
    
def Pipestart(fn: Callable) -> Callable:
    @functools.wraps(fn)
    async def wrapper(*args, **kwargs) -> DataFrame:
        pipe = await fn(*args, **kwargs)
        return await pipe.run(verbose=True)
    return wrapper
    