"""
LLM API Retry Logic with Circuit Breaker Pattern

Provides robust retry logic, exponential backoff, and circuit breaker
for handling LLM API calls in production environments.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, TypeVar, Dict
from functools import wraps
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Type variables for generic functions
T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Circuit is tripped, blocking calls
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class RetryConfig:
    """Configuration for retry logic."""
    max_attempts: int = 3
    initial_backoff: float = 1.0  # seconds
    max_backoff: float = 60.0  # seconds
    exponential_base: float = 2.0
    jitter: bool = True  # Add randomness to prevent thundering herd
    retry_on: tuple = (Exception,)  # Exception types to retry on
    timeout: Optional[float] = None  # Request timeout in seconds


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5  # Failures before tripping
    success_threshold: int = 2  # Successes to close circuit
    timeout: float = 60.0  # Seconds before trying half-open
    window_size: int = 100  # Rolling window size for tracking


@dataclass
class CallResult:
    """Result of an API call attempt."""
    success: bool
    duration: float
    error: Optional[Exception] = None
    timestamp: datetime = field(default_factory=datetime.now)


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open and blocks a call."""
    pass


class CircuitBreaker:
    """
    Circuit breaker for failing fast when a service is down.

    Prevents cascading failures by stopping calls to a service
    that has been failing repeatedly.
    """

    def __init__(self, config: CircuitBreakerConfig, service_name: str):
        """
        Initialize circuit breaker.

        Args:
            config: Circuit breaker configuration
            service_name: Name of the service being monitored
        """
        self.config = config
        self.service_name = service_name
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.call_history: list[CallResult] = []  # Rolling window

    def record_success(self):
        """Record a successful call."""
        self.success_count += 1
        self.failure_count = 0  # Reset on success
        self.call_history.append(CallResult(success=True, duration=0.0))

        # Trim rolling window
        if len(self.call_history) > self.config.window_size:
            self.call_history = self.call_history[-self.config.window_size :]

        # If in half-open and enough successes, close circuit
        if (self.state == CircuitState.HALF_OPEN and
            self.success_count >= self.config.success_threshold):
            self.state = CircuitState.CLOSED
            self.success_count = 0
            logger.info(f"Circuit breaker CLOSED for {self.service_name} - service recovered")

    def record_failure(self, error: Exception):
        """Record a failed call."""
        self.failure_count += 1
        self.success_count = 0
        self.last_failure_time = datetime.now()
        self.call_history.append(CallResult(success=False, duration=0.0, error=error))

        # Trim rolling window
        if len(self.call_history) > self.config.window_size:
            self.call_history = self.call_history[-self.config.window_size :]

        # Trip circuit if too many failures
        if (self.state == CircuitState.CLOSED and
            self.failure_count >= self.config.failure_threshold):
            self.state = CircuitState.OPEN
            logger.warning(
                f"Circuit breaker OPENED for {self.service_name} - "
                f"{self.failure_count} failures in {self.config.window_size} calls"
            )

    def can_attempt(self) -> bool:
        """
        Check if a call is allowed.

        Returns:
            True if call can proceed, False if circuit is open
        """
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # Check if timeout has passed
            if (self.last_failure_time and
                datetime.now() - self.last_failure_time > timedelta(seconds=self.config.timeout)):
                self.state = CircuitState.HALF_OPEN
                logger.info(f"Circuit breaker HALF_OPEN for {self.service_name} - attempting recovery")
                return True
            return False

        if self.state == CircuitState.HALF_OPEN:
            return True

        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        recent_failures = sum(1 for r in self.call_history if not r.success)
        recent_successes = sum(1 for r in self.call_history if r.success)
        total_calls = len(self.call_history)

        return {
            "service": self.service_name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "recent_failures": recent_failures,
            "recent_successes": recent_successes,
            "total_calls": total_calls,
            "failure_rate": recent_failures / total_calls if total_calls > 0 else 0.0,
        }


class RetryManager:
    """
    Manages retry logic with exponential backoff and jitter.

    Provides decorators and context managers for retrying operations
    with intelligent backoff strategies.
    """

    def __init__(self, config: RetryConfig):
        """
        Initialize retry manager.

        Args:
            config: Retry configuration
        """
        self.config = config

    def get_backoff_time(self, attempt: int) -> float:
        """
        Calculate backoff time with exponential backoff and jitter.

        Args:
            attempt: Attempt number (1-based)

        Returns:
            Backoff time in seconds
        """
        # Calculate exponential backoff
        backoff = min(
            self.config.initial_backoff * (self.config.exponential_base ** (attempt - 1)),
            self.config.max_backoff
        )

        # Add jitter to prevent thundering herd
        if self.config.jitter:
            import random
            backoff = backoff * (0.5 + random.random() * 0.5)

        return backoff

    def should_retry(self, error: Exception, attempt: int) -> bool:
        """
        Determine if operation should be retried.

        Args:
            error: Exception that occurred
            attempt: Current attempt number

        Returns:
            True if should retry, False otherwise
        """
        # Check if we've exceeded max attempts
        if attempt >= self.config.max_attempts:
            return False

        # Check if error is retryable
        for retryable_type in self.config.retry_on:
            if isinstance(error, retryable_type):
                return True

        return False

    def retry(self, func: Callable[..., T]) -> Callable[..., T]:
        """
        Decorator to add retry logic to a function.

        Args:
            func: Function to decorate

        Returns:
            Decorated function with retry logic
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 1
            last_error = None

            while attempt <= self.config.max_attempts:
                try:
                    if self.config.timeout:
                        # Run with timeout
                        import signal

                        def timeout_handler(signum, frame):
                            raise TimeoutError(f"Function {func.__name__} timed out after {self.config.timeout}s")

                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(int(self.config.timeout))

                        try:
                            result = func(*args, **kwargs)
                            signal.alarm(0)  # Cancel alarm
                            return result
                        except Exception as e:
                            signal.alarm(0)  # Cancel alarm
                            raise
                    else:
                        return func(*args, **kwargs)

                except Exception as e:
                    last_error = e

                    if not self.should_retry(e, attempt):
                        logger.error(
                            f"Function {func.__name__} failed on attempt {attempt}: {e}"
                        )
                        raise

                    # Log retry
                    backoff = self.get_backoff_time(attempt)
                    logger.warning(
                        f"Function {func.__name__} failed on attempt {attempt}/{self.config.max_attempts}: "
                        f"{e}. Retrying in {backoff:.1f}s..."
                    )

                    time.sleep(backoff)
                    attempt += 1

            # All retries exhausted
            logger.error(
                f"Function {func.__name__} failed after {self.config.max_attempts} attempts"
            )
            raise last_error

        return wrapper


class LLMClientWithRetry:
    """
    Wrapper for LLM clients with retry logic and circuit breaker.

    Combines retry logic with circuit breaker for robust API calls.
    """

    def __init__(
        self,
        client: Any,
        service_name: str,
        retry_config: Optional[RetryConfig] = None,
        circuit_config: Optional[CircuitBreakerConfig] = None,
    ):
        """
        Initialize LLM client wrapper.

        Args:
            client: Underlying LLM client (e.g., OpenAI, Anthropic)
            service_name: Name of the service (for logging)
            retry_config: Retry configuration (uses defaults if None)
            circuit_config: Circuit breaker configuration (uses defaults if None)
        """
        self.client = client
        self.service_name = service_name
        self.retry_config = retry_config or RetryConfig()
        self.circuit_config = circuit_config or CircuitBreakerConfig()
        self.circuit_breaker = CircuitBreaker(self.circuit_config, service_name)
        self.retry_manager = RetryManager(self.retry_config)

    def call_with_retry(self, method_name: str, *args, **kwargs) -> Any:
        """
        Call LLM client method with retry logic and circuit breaker.

        Args:
            method_name: Name of the method to call
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method

        Returns:
            Method return value

        Raises:
            CircuitBreakerError: If circuit is open
            Exception: If all retries are exhausted
        """
        # Check circuit breaker
        if not self.circuit_breaker.can_attempt():
            stats = self.circuit_breaker.get_stats()
            raise CircuitBreakerError(
                f"Circuit breaker is OPEN for {self.service_name}. "
                f"Recent failure rate: {stats['failure_rate']:.1%}"
            )

        # Make the call with retry logic
        start_time = time.time()
        attempt = 1
        last_error = None

        while attempt <= self.retry_config.max_attempts:
            try:
                # Call the actual method
                method = getattr(self.client, method_name)
                result = method(*args, **kwargs)

                # Record success
                duration = time.time() - start_time
                self.circuit_breaker.record_success()
                logger.debug(
                    f"{self.service_name}.{method_name} succeeded in {duration:.2f}s "
                    f"(attempt {attempt})"
                )

                return result

            except Exception as e:
                last_error = e

                if not self.retry_manager.should_retry(e, attempt):
                    # Don't retry this error
                    self.circuit_breaker.record_failure(e)
                    logger.error(
                        f"{self.service_name}.{method_name} failed with non-retryable error: {e}"
                    )
                    raise

                # Check if we should retry
                if attempt < self.retry_config.max_attempts:
                    backoff = self.retry_manager.get_backoff_time(attempt)
                    logger.warning(
                        f"{self.service_name}.{method_name} failed on attempt {attempt}: {e}. "
                        f"Retrying in {backoff:.1f}s..."
                    )
                    time.sleep(backoff)
                    attempt += 1

                # All retries exhausted
                else:
                    # Record failure and raise
                    self.circuit_breaker.record_failure(e)
                    logger.error(
                        f"{self.service_name}.{method_name} failed after {attempt} attempts: {e}"
                    )
                    raise

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the service."""
        return {
            "service": self.service_name,
            "circuit_breaker": self.circuit_breaker.get_stats(),
            "retry_config": {
                "max_attempts": self.retry_config.max_attempts,
                "initial_backoff": self.retry_config.initial_backoff,
                "max_backoff": self.retry_config.max_backoff,
            }
        }


# ============================================================================
# Convenience Decorators
# ============================================================================

def retry_on_failure(
    max_attempts: int = 3,
    initial_backoff: float = 1.0,
    max_backoff: float = 60.0,
    retry_on: tuple = (Exception,),
):
    """
    Decorator to add retry logic to a function.

    Args:
        max_attempts: Maximum number of attempts
        initial_backoff: Initial backoff time in seconds
        max_backoff: Maximum backoff time in seconds
        retry_on: Tuple of exception types to retry on

    Returns:
        Decorated function with retry logic

    Example:
        @retry_on_failure(max_attempts=5, retry_on=(TimeoutError, ConnectionError))
        def risky_operation():
            # Might fail temporarily
            return api_call()
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        initial_backoff=initial_backoff,
        max_backoff=max_backoff,
        retry_on=retry_on,
    )
    manager = RetryManager(config)

    def decorator(func):
        return manager.retry(func)

    return decorator


def with_circuit_breaker(
    service_name: str,
    failure_threshold: int = 5,
    timeout: float = 60.0,
):
    """
    Decorator to add circuit breaker to a function.

    Args:
        service_name: Name of the service being called
        failure_threshold: Failures before tripping circuit
        timeout: Seconds before trying half-open

    Returns:
        Decorated function with circuit breaker

    Example:
        @with_circuit_breaker(service_name="openai", failure_threshold=10)
        def call_openai_api():
            # May fail repeatedly
            return openai.ChatCompletion.create(...)
    """
    config = CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        timeout=timeout,
    )
    circuit_breaker = CircuitBreaker(config, service_name)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check circuit
            if not circuit_breaker.can_attempt():
                raise CircuitBreakerError(f"Circuit breaker is OPEN for {service_name}")

            try:
                result = func(*args, **kwargs)
                circuit_breaker.record_success()
                return result
            except Exception as e:
                circuit_breaker.record_failure(e)
                raise

        return wrapper

    return decorator


# ============================================================================
# Common Exception Types for Retry
# ============================================================================

class TransientError(Exception):
    """Base class for transient errors that should be retried."""
    pass


class RateLimitError(TransientError):
    """Rate limit exceeded - retry with backoff."""
    pass


class ServiceUnavailableError(TransientError):
    """Service temporarily unavailable - retry with backoff."""
    pass


class TimeoutError(TransientError):
    """Request timed out - retry with backoff."""
    pass


# Common retry configuration for LLM APIs
LLM_RETRY_CONFIG = RetryConfig(
    max_attempts=5,
    initial_backoff=2.0,
    max_backoff=60.0,
    exponential_base=2.0,
    jitter=True,
    retry_on=(
        TimeoutError,
        ConnectionError,
        RateLimitError,
        ServiceUnavailableError,
    ),
    timeout=30.0,  # 30 second timeout per request
)

LLM_CIRCUIT_CONFIG = CircuitBreakerConfig(
    failure_threshold=10,  # Trip after 10 failures
    success_threshold=3,   # Close after 3 successes
    timeout=60.0,          # Try again after 1 minute
    window_size=100,       # Track last 100 calls
)
