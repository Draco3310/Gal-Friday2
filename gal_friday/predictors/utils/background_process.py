"""Module for managing background processes using ProcessPoolExecutor."""
from collections.abc import Callable
from enum import Enum
import logging
from types import TracebackType
from typing import Any, Generic, Optional, TypeVar, cast

import asyncio
from concurrent.futures import (
    Future,
    ProcessPoolExecutor,
    TimeoutError as FutureTimeoutError,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Exception messages
MSG_PROCESS_CANCELLED = "Process %s was cancelled."
MSG_PROCESS_FAILED_NO_SPECIFIC_EXCEPTION = "Process %s failed without specific future exception."
MSG_TASK_SUBMISSION_FAILED = "Task[Any] '%s' submission failed."
MSG_TIMEOUT_WAITING = "Timeout waiting for result from %s"
MSG_FUTURE_TYPE_ERROR = "future must be a Future[Any] instance"


class ProcessStatus(Enum):
    """Represents the status of a background process."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BackgroundProcessError(Exception):
    """Base exception for background process errors."""


class BackgroundProcessNotStartedError(BackgroundProcessError):
    """Raised when an operation is attempted on a process that hasn't started."""


class BackgroundProcessAlreadyStartedError(BackgroundProcessError):
    """Raised when attempting to start an already started process."""


class BackgroundProcess(Generic[T]):
    """Represents a task submitted to a ProcessPoolExecutor.

    This class wraps a concurrent.futures.Future[Any] object and provides
    status tracking and result retrieval.
    """

    def __init__(self, future: Future[Any], target_name: str = "Unnamed Process") -> None:
        """Initialize a BackgroundProcess instance.

        Args:
            future: A concurrent.futures.Future[Any] object representing the background task.
            target_name: A human-readable name for the background process. Defaults to
                "Unnamed Process".

        Raises:
            TypeError: If the provided future is not a Future[Any] instance.
        """
        if not isinstance(future, Future):
            raise TypeError(MSG_FUTURE_TYPE_ERROR)
        self._future: Future[Any] = future
        self._target_name: str = target_name
        self._status: ProcessStatus = ProcessStatus.PENDING
        self._update_initial_status()

    def _update_initial_status(self) -> None:
        if self._future.running():
            self._status = ProcessStatus.RUNNING
        elif self._future.done():
            if self._future.cancelled():
                self._status = ProcessStatus.CANCELLED
            elif self._future.exception() is not None:
                self._status = ProcessStatus.FAILED
            else:
                self._status = ProcessStatus.COMPLETED

    @property
    def status(self) -> ProcessStatus:
        """Get the current status of the background process."""
        if self._status not in {
            ProcessStatus.COMPLETED,
            ProcessStatus.FAILED,
            ProcessStatus.CANCELLED,
        }:
            if self._future.running():
                self._status = ProcessStatus.RUNNING
            elif self._future.done():
                if self._future.cancelled():
                    self._status = ProcessStatus.CANCELLED
                elif self._future.exception() is not None:
                    self._status = ProcessStatus.FAILED
                else:
                    self._status = ProcessStatus.COMPLETED
        return self._status

    def get_result(self, timeout: float | None = None) -> T:
        """Get the result of the background process.

        Blocks until the process is completed or timeout occurs.

        Args:
            timeout: Maximum time in seconds to wait for the result.

        Returns:
        -------
            The result of the target function.

        Raises:
        ------
            TimeoutError: If the timeout is reached while waiting for the result.
            Exception: If the target function raised an exception.
            BackgroundProcessError: If the process was cancelled.
        """
        if self.status == ProcessStatus.CANCELLED:
            raise BackgroundProcessError(MSG_PROCESS_CANCELLED % self._target_name)
        try:
            result = self._future.result(timeout=timeout)
        except FutureTimeoutError as e:
            logger.warning(MSG_TIMEOUT_WAITING, self._target_name)
            raise TimeoutError(MSG_TIMEOUT_WAITING % self._target_name) from e
        except Exception:
            logger.exception("Process %s raised an exception", self._target_name)
            self._status = ProcessStatus.FAILED
            raise
        else:
            self._status = ProcessStatus.COMPLETED
            return cast("T", result)

    async def aget_result(self, poll_interval: float = 0.1) -> T:
        """Asynchronously get the result of the background process.

        Polls the future until it's done.

        Args:
            poll_interval: How often to check the future's status.

        Returns:
        -------
            The result of the target function.

        Raises:
        ------
            Exception: If the target function raised an exception.
            BackgroundProcessError: If the process was cancelled or failed
                without specific future exception.
        """
        while not self._future.done():
            await asyncio.sleep(poll_interval)

        _ = self.status  # Update status explicitly after loop

        if self._status == ProcessStatus.CANCELLED:
            raise BackgroundProcessError(MSG_PROCESS_CANCELLED % self._target_name)
        if self._status == ProcessStatus.FAILED:
            exc = self._future.exception()
            if exc:
                logger.exception(
                    "Process %s raised an exception (async)",
                    self._target_name)
                raise exc
            # This part is reached if status is FAILED but future had no exception
            # (e.g. pre-run fail)
            raise BackgroundProcessError(
                MSG_PROCESS_FAILED_NO_SPECIFIC_EXCEPTION % self._target_name)

        return cast("T", self._future.result())

    def cancel(self) -> bool:
        """Attempt to cancel the background process.

        Note: Cancellation of tasks in ProcessPoolExecutor is best-effort
        and may not actually interrupt a running task.

        Returns:
        -------
            True if the cancellation attempt was successful (i.e., future was cancelled),
            False otherwise (e.g., task already running or completed).
        """
        was_cancelled = self._future.cancel()
        if was_cancelled:
            self._status = ProcessStatus.CANCELLED
            logger.info("Cancellation requested and successful for process %s.", self._target_name)
        else:
            logger.warning(
                "Failed to cancel process %s (may be running/completed/cancelled). Status: %s",
                self._target_name,
                self.status.value)
        return was_cancelled

    def __repr__(self) -> str:
        """Return a string representation of the BackgroundProcess instance."""
        return f"<BackgroundProcess target='{self._target_name}' status='{self.status.value}'>"


class BackgroundProcessManager:
    """Manages a ProcessPoolExecutor for running background tasks.

    Provides a clean interface to submit tasks and get BackgroundProcess handles.
    This is designed as a singleton for the executor to ensure one pool is shared.
    """

    _instance: Optional["BackgroundProcessManager"] = None
    _executor: ProcessPoolExecutor | None = None
    _max_workers_at_init: int | None = None  # Class attribute to store initial max_workers

    def __new__(cls) -> "BackgroundProcessManager":
        """Ensure a single instance of BackgroundProcessManager."""
        if not cls._instance:
            cls._instance = super().__new__(cls)  # Use super() directly
        return cls._instance

    def __init__(self, max_workers: int | None = None) -> None:
        """Initialize the BackgroundProcessManager and its executor if not already done."""
        if not hasattr(self, "_initialized"):  # Ensure __init__ logic runs once per instance
            if BackgroundProcessManager._executor is None:
                # Store max_workers from the first instantiation call at class level for reuse
                if (
                    BackgroundProcessManager._max_workers_at_init is None
                    and max_workers is not None
                ):
                    BackgroundProcessManager._max_workers_at_init = max_workers

                resolved_max_workers = (
                    max_workers
                    if max_workers is not None
                    else BackgroundProcessManager._max_workers_at_init
                )

                BackgroundProcessManager._executor = ProcessPoolExecutor(
                    max_workers=resolved_max_workers)
                logger.info(
                    "ProcessPoolExecutor initialized with max_workers=%s",
                    resolved_max_workers or "default")
            else:
                logger.info("ProcessPoolExecutor already initialized.")
            self._initialized = True

    @classmethod
    def get_instance(cls, max_workers: int | None = None) -> "BackgroundProcessManager":
        """Get the singleton instance of the BackgroundProcessManager, initializing if needed."""
        if cls._instance is None:
            cls._instance = cls(max_workers=max_workers)
        elif cls._executor is None:
            logger.warning(
                "BackgroundProcessManager instance exists but executor is None. "
                "Reinitializing executor.")
            # Use max_workers if provided, else the class-stored initial value
            resolved_max_workers = (
                max_workers if max_workers is not None else cls._max_workers_at_init
            )
            BackgroundProcessManager._executor = ProcessPoolExecutor(
                max_workers=resolved_max_workers)
            if cls._max_workers_at_init is None and resolved_max_workers is not None:
                cls._max_workers_at_init = resolved_max_workers
        elif max_workers is not None and cls._max_workers_at_init is None:
            # Instance and executor exist, but this call might be the first to specify max_workers
            cls._max_workers_at_init = max_workers
            # Note: This doesn't re-init the pool if it's already running with default workers.
            # For simplicity, we assume pool is configured at first get_instance or __init__.

        return cls._instance

    def submit(
        self,
        func: Callable[..., Any],
        *args: object,
        **kwargs: object) -> BackgroundProcess[T]:
        """Submit a function to be executed in the background process pool.

        Args:
            func: The function to execute.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
        -------
            A BackgroundProcess instance representing the submitted task.

        Raises:
        ------
            RuntimeError: If the executor is not initialized or has been shut down.
            BackgroundProcessError: If submission fails (e.g. PicklingError).
        """
        if self._executor is None:
            msg = "ProcessPoolExecutor not initialized or shut down."
            logger.error("%s Call get_instance.", msg)
            raise RuntimeError(msg)

        target_name = getattr(func, "__name__", "unknown_function")
        try:
            future = self._executor.submit(func, *args, **kwargs)
            logger.info("Submitted task '%s' to process pool.", target_name)
            return BackgroundProcess(future, target_name=target_name)
        except Exception as e:
            logger.exception("Failed to submit task '%s' to process pool.", target_name)
            raise BackgroundProcessError(
                MSG_TASK_SUBMISSION_FAILED % target_name) from e

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        """Shutdown the ProcessPoolExecutor.

        Args:
            wait: Whether to wait for all pending tasks to complete execution.
            cancel_futures: If True, attempt to cancel all pending futures before shutting down.
                            This is only effective if 'wait' is also True. (Python 3.9+)
        """
        if self._executor is not None:
            logger.info(
                "Shutting down ProcessPoolExecutor (wait=%s, cancel_futures=%s)...",
                wait,
                cancel_futures)
            if hasattr(self._executor, "shutdown"):  # Python 3.9+ for cancel_futures
                self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)
            else:  # Older Python versions
                self._executor.shutdown(wait=wait)
            BackgroundProcessManager._executor = None
            logger.info("ProcessPoolExecutor shut down.")
        else:
            logger.info("ProcessPoolExecutor was not running or already shut down.")

    def __enter__(self) -> "BackgroundProcessManager":
        """Enter the runtime context related to this object, ensuring pool is initialized."""
        if self._executor is None:
            logger.info("Initializing executor via BackgroundProcessManager context entry.")
            # Use class-stored _max_workers_at_init or None if not set
            resolved_max_workers = BackgroundProcessManager._max_workers_at_init
            BackgroundProcessManager._executor = ProcessPoolExecutor(
                max_workers=resolved_max_workers)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None) -> None:
        """Exit the runtime context, shutting down the pool."""
        self.shutdown(wait=True)
