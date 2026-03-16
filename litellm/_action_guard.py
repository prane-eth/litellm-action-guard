import asyncio
import inspect
import threading
from concurrent.futures import Future
from typing import Any, Awaitable, Callable, Optional, TypeVar


T = TypeVar("T")


async def _await_result(awaitable: Awaitable[T]) -> T:
    return await awaitable


class _AsyncToSyncRunner:
    """
    Run coroutines from synchronous code even when a loop
    is already running in the current thread.
    """

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        self._ready.set()
        loop.run_forever()

    def _ensure_started(self) -> asyncio.AbstractEventLoop:
        if self._loop is not None and self._thread is not None:
            return self._loop
        t = threading.Thread(target=self._thread_main, daemon=True)
        self._thread = t
        t.start()
        self._ready.wait()
        assert self._loop is not None
        return self._loop

    def run(self, coro: Awaitable[T], timeout: Optional[float] = None) -> T:
        loop = self._ensure_started()
        fut: Future[T] = asyncio.run_coroutine_threadsafe(_await_result(coro), loop)
        return fut.result(timeout=timeout)


_RUNNER = _AsyncToSyncRunner()


async def call_action_guard_async(
    action_guard: Callable[[Any], Any],
    guard_input: Any,
) -> Any:
    """
    Invoke an action_guard from async code.

    Supports both sync and async guards. If the guard returns an awaitable,
    it will be awaited.
    """

    result = action_guard(guard_input)
    if inspect.isawaitable(result):
        return await result
    return result


def call_action_guard_sync(
    action_guard: Callable[[Any], Any],
    guard_input: Any,
) -> Any:
    """
    Invoke an action_guard from sync code.

    Supports both sync and async guards. If the guard returns an awaitable:
    - when no event loop is running in the current thread, it uses asyncio.run()
    - when an event loop is already running, it runs the awaitable on a dedicated
      background event loop thread.
    """

    result = action_guard(guard_input)
    if not inspect.isawaitable(result):
        return result

    # If it's a Future bound to the *current* running loop, we can't safely block
    # here without deadlocking. Ask the caller to use an async flow.
    if asyncio.isfuture(result):
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is not None:
            fut_loop = result.get_loop()  # type: ignore[no-untyped-call]
            if fut_loop is running_loop:
                raise RuntimeError(
                    "Async action_guard returned a Future bound to the running loop. "
                    "Use litellm.acompletion()/async flow or return a coroutine instead."
                )

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop in this thread, safe to use asyncio.run()
        return asyncio.run(_await_result(result))  # type: ignore[arg-type]

    # Running loop present: execute in background event loop thread.
    return _RUNNER.run(result)  # type: ignore[arg-type]

