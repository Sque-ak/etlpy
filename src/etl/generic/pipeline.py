from __future__ import annotations
import functools, inspect, logging
from typing import Any, Awaitable, Callable

from polars import DataFrame, LazyFrame
from etl.generic.context import Data
from etl.generic.step import Step, StopPipeline

_log = logging.getLogger(__name__)

Subscriber = Callable[[str, dict[str, Any]], None | Awaitable[None]]
_subscribers: list[Subscriber] = []

def subscribe(fn: Subscriber) -> Subscriber:
    if fn not in _subscribers:
        _subscribers.append(fn)
    return fn

def unsubscribe(fn: Subscriber) -> None:
    if fn in _subscribers:
        _subscribers.remove(fn)

async def _emit(event: str, payload: dict[str, Any]) -> None:
    """
    Notify subscribers. Sync and async callables are supported
    a broken subscriber is logged at debug level and never breaks the pipelines
    """
    for fn in _subscribers:
        try:
            result = fn(event, payload)
            if inspect.isawaitable(result):
                await result
        except Exception as error:
            _log.debug("event subscriber %r failed on %r", fn, event, exc_info=True) 

class Pipeline:
    """
    Chain of async steps.

    Example: 
        pipe = Pipeline([OAuthenticate(api), GetAccounts(), GetTransactions()])
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
        status = "completed"
        await _emit("pipeline_start", {"pipeline": self})
        try:
            for i, step in enumerate(self.steps):
                if verbose:
                    print(f" [{i + 1}/{len(self.steps)}] {step!r}")
                await _emit("step_start", {"pipeline": self, "index": i, "step": step})
                try:
                    self.dataframe = await  step.apply(self.dataframe, self.data)
                except StopPipeline as stop:
                    if verbose:
                        print(f" [stop] {step!r}: {stop}")
                    await _emit("step_stop", {"pipeline": self, "index":i, "step": step, "reason": str(stop)})
                    status = "stopped"
                    return stop.df if stop.df is not None else self.dataframe
                except Exception as error:
                    error.add_note(f"Pipeline failed at step [{i+1}/{len(self.steps)}] - {step!r}")
                    await _emit("step_error", {"pipeline": self, "index": i, "step": step, "error": error})
                    status = "error"
                    raise
                await _emit("step_end", {"pipeline": self, "index": i, "step": step, "dataframe": self.dataframe})
            return self.dataframe
        finally:
            await _emit("pipeline_end", {"pipeline": self, "status": status})
    
    
    def __repr__(self) -> str:
        body = "\n ".join(repr(s) for s in self.steps)
        return f"Pipeline(steps=[\n {body} \n])"
    
def Pipestart(fn: Callable) -> Callable:
    @functools.wraps(fn)
    async def wrapper(*args, **kwargs) -> DataFrame:
        pipe = await fn(*args, **kwargs)
        return await pipe.run(verbose=True)
    return wrapper
    