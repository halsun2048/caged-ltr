"""Background task abstraction: synchronous fallback plus optional queue hook."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor


class TaskRunner:
    def __init__(self, workers: int = 2) -> None:
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="caged-task")

    def submit(self, function: Callable, *args, **kwargs) -> str:
        future = self.executor.submit(function, *args, **kwargs)
        return f"local-{id(future)}"

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
