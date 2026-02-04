"""Unit tests for LLM retry logic and circuit breaker."""

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.utils.llm_retry import (
    RetryConfig,
    CircuitBreakerConfig,
    CircuitBreaker,
    RetryManager,
    CircuitBreakerError,
    LLMClientWithRetry,
    retry_on_failure,
    with_circuit_breaker,
    TransientError,
    RateLimitError,
    ServiceUnavailableError,
)


class TestRetryConfig:
    """Test retry configuration."""

    def test_default_config(self):
        """Test default retry configuration."""
        config = RetryConfig()
        assert config.max_attempts == 3
        assert config.initial_backoff == 1.0
        assert config.max_backoff == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True
        assert config.retry_on == (Exception,)

    def test_custom_config(self):
        """Test custom retry configuration."""
        config = RetryConfig(
            max_attempts=5,
            initial_backoff=2.0,
            retry_on=(ValueError, TypeError),
        )
        assert config.max_attempts == 5
        assert config.initial_backoff == 2.0
        assert config.retry_on == (ValueError, TypeError)


class TestCircuitBreakerConfig:
    """Test circuit breaker configuration."""

    def test_default_config(self):
        """Test default circuit breaker configuration."""
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 5
        assert config.success_threshold == 2
        assert config.timeout == 60.0
        assert config.window_size == 100


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = CircuitBreakerConfig(failure_threshold=3, success_threshold=2)
        self.cb = CircuitBreaker(self.config, "test_service")

    def test_initial_state_closed(self):
        """Test circuit breaker starts in CLOSED state."""
        assert self.cb.state.value == "closed"
        assert self.cb.can_attempt() is True

    def test_record_success(self):
        """Test recording successful calls."""
        self.cb.record_success()
        assert self.cb.success_count == 1
        assert self.cb.failure_count == 0
        assert self.cb.state.value == "closed"

    def test_record_failure(self):
        """Test recording failed calls."""
        error = Exception("Test error")
        self.cb.record_failure(error)
        assert self.cb.failure_count == 1
        assert self.cb.success_count == 0
        assert self.cb.state.value == "closed"  # Still below threshold

    def test_trip_circuit(self):
        """Test circuit breaker trips after threshold."""
        # Record failures up to threshold
        for i in range(3):
            self.cb.record_failure(Exception(f"Error {i}"))

        # Should now be open
        assert self.cb.state.value == "open"
        assert self.cb.can_attempt() is False

    def test_reset_on_success(self):
        """Test failure count resets on success."""
        # Add some failures
        self.cb.record_failure(Exception("Error 1"))
        self.cb.record_failure(Exception("Error 2"))
        assert self.cb.failure_count == 2

        # Success resets failure count
        self.cb.record_success()
        assert self.cb.failure_count == 0

    def test_half_open_recovery(self):
        """Test circuit breaker recovers after timeout."""
        # Trip the circuit
        for i in range(3):
            self.cb.record_failure(Exception(f"Error {i}"))

        assert self.cb.state.value == "open"
        assert self.cb.can_attempt() is False

        # Simulate timeout passing
        self.cb.last_failure_time = datetime.now() - timedelta(seconds=61)

        # Check circuit state transitions to half-open
        assert self.cb.can_attempt() is True
        assert self.cb.state.value == "half_open"

    def test_close_after_successes(self):
        """Test circuit closes after enough successes."""
        # Trip the circuit
        for i in range(3):
            self.cb.record_failure(Exception(f"Error {i}"))

        # Move to half-open
        self.cb.last_failure_time = datetime.now() - timedelta(seconds=61)
        self.cb.can_attempt()

        # Add successes to close circuit
        for i in range(2):
            self.cb.record_success()

        assert self.cb.state.value == "closed"

    def test_get_stats(self):
        """Test getting circuit breaker statistics."""
        self.cb.record_success()
        self.cb.record_success()
        self.cb.record_failure(Exception("Error"))

        stats = self.cb.get_stats()
        assert stats["service"] == "test_service"
        assert stats["recent_successes"] == 2
        assert stats["recent_failures"] == 1
        assert stats["total_calls"] == 3
        assert 0.0 < stats["failure_rate"] < 1.0


class TestRetryManager:
    """Test retry manager functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = RetryConfig(
            max_attempts=3,
            initial_backoff=0.1,  # Short for tests
            exponential_base=2.0,
            jitter=False,  # Disable for predictable tests
        )
        self.manager = RetryManager(self.config)

    def test_backoff_calculation(self):
        """Test exponential backoff calculation."""
        # First attempt: 0.1s
        assert self.manager.get_backoff_time(1) == 0.1
        # Second attempt: 0.2s
        assert self.manager.get_backoff_time(2) == 0.2
        # Third attempt: 0.4s
        assert self.manager.get_backoff_time(3) == 0.4

    def test_max_backoff(self):
        """Test backoff caps at max value."""
        backoff = self.manager.get_backoff_time(10)
        assert backoff <= self.config.max_backoff

    def test_should_retry_on_retryable_error(self):
        """Test should_retry returns True for retryable errors."""
        error = ValueError("Test error")
        assert self.manager.should_retry(error, 1) is True
        assert self.manager.should_retry(error, 2) is True
        assert self.manager.should_retry(error, 3) is False  # Max attempts

    def test_should_not_retry_on_non_retryable(self):
        """Test should_retry returns False for non-retryable errors."""
        config = RetryConfig(retry_on=(TypeError,))
        manager = RetryManager(config)

        error = ValueError("Not retryable")
        assert manager.should_retry(error, 1) is False


class TestRetryDecorator:
    """Test retry decorator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.attempt_count = 0

    def test_retry_on_failure(self):
        """Test function retries on failure."""
        @retry_on_failure(max_attempts=3, initial_backoff=0.01)
        def failing_function():
            self.attempt_count += 1
            if self.attempt_count < 3:
                raise ValueError("Not yet")
            return "success"

        result = failing_function()
        assert result == "success"
        assert self.attempt_count == 3

    def test_no_retry_on_success(self):
        """Test function doesn't retry on success."""
        @retry_on_failure(max_attempts=3)
        def successful_function():
            return "success"

        result = successful_function()
        assert result == "success"

    def test_raises_after_max_attempts(self):
        """Test exception raised after max attempts."""
        @retry_on_failure(max_attempts=2, initial_backoff=0.01)
        def always_failing_function():
            raise ValueError("Always fails")

        with pytest.raises(ValueError, match="Always fails"):
            always_failing_function()

    def test_no_retry_for_unlisted_exceptions(self):
        """Test exceptions not in retry_on list are not retried."""
        @retry_on_failure(max_attempts=3, retry_on=(ValueError,))
        def wrong_error():
            raise TypeError("Wrong type")

        with pytest.raises(TypeError, match="Wrong type"):
            wrong_error()


class TestCircuitBreakerDecorator:
    """Test circuit breaker decorator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.call_count = 0

    def test_allows_calls_initially(self):
        """Test decorator allows calls when circuit is closed."""
        @with_circuit_breaker(service_name="test", failure_threshold=3)
        def service_function():
            return "success"

        result = service_function()
        assert result == "success"

    def test_blocks_calls_when_open(self):
        """Test decorator blocks calls when circuit is open."""
        config = CircuitBreakerConfig(failure_threshold=2, timeout=0.1)
        cb = CircuitBreaker(config, "test")

        @with_circuit_breaker(service_name="test", failure_threshold=2, timeout=0.1)
        def failing_function():
            cb.record_failure(Exception("Fail"))
            raise Exception("Failed")

        # Trip the circuit
        for i in range(2):
            try:
                failing_function()
            except:
                pass

        # Circuit should be open now
        with pytest.raises(CircuitBreakerError, match="Circuit breaker is OPEN"):
            failing_function()


class TestLLMClientWithRetry:
    """Test LLM client wrapper with retry and circuit breaker."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_client = Mock()
        self.wrapper = LLMClientWithRetry(
            self.mock_client,
            "test_service",
            retry_config=RetryConfig(max_attempts=2, initial_backoff=0.01),
            circuit_config=CircuitBreakerConfig(failure_threshold=2, timeout=0.1),
        )

    def test_successful_call(self):
        """Test successful call passes through."""
        # Create a simple mock client with a method that returns a value
        class MockClient:
            def create_method(self):
                return {"result": "success"}

        self.wrapper.client = MockClient()

        result = self.wrapper.call_with_retry("create_method")

        assert result == {"result": "success"}
        assert self.wrapper.circuit_breaker.state.value == "closed"

    def test_retry_on_failure(self):
        """Test call retries on transient failure."""
        self.mock_client.some_method.side_effect = [
            TransientError("First fail"),
            {"result": "success"},
        ]

        result = self.wrapper.call_with_retry("some_method")

        assert result == {"result": "success"}
        assert self.mock_client.some_method.call_count == 2

    def test_circuit_breaker_trips(self):
        """Test circuit breaker trips after repeated failures."""
        self.mock_client.some_method.side_effect = Exception("Always fails")

        # Try multiple times
        for i in range(5):
            try:
                self.wrapper.call_with_retry("some_method")
            except:
                pass

        # Circuit should be open
        assert self.wrapper.circuit_breaker.state.value == "open"

        # Next call should be blocked
        with pytest.raises(CircuitBreakerError):
            self.wrapper.call_with_retry("some_method")

    def test_get_stats(self):
        """Test getting client statistics."""
        stats = self.wrapper.get_stats()
        assert stats["service"] == "test_service"
        assert "circuit_breaker" in stats
        assert "retry_config" in stats


class TestIntegration:
    """Integration tests for retry and circuit breaker."""

    def test_flaky_service_succeeds(self):
        """Test flaky service eventually succeeds."""
        call_count = [0]

        @retry_on_failure(max_attempts=5, initial_backoff=0.01, retry_on=(TransientError,))
        def flaky_service():
            call_count[0] += 1
            if call_count[0] < 3:
                raise TransientError("Temporary failure")
            return "success"

        result = flaky_service()
        assert result == "success"
        assert call_count[0] == 3

    def test_permanent_failure_fails_fast(self):
        """Test permanent failure fails fast."""
        @with_circuit_breaker(service_name="bad_service", failure_threshold=2)
        def bad_service():
            raise Exception("Permanent failure")

        # Trip circuit breaker
        for i in range(5):
            try:
                bad_service()
            except CircuitBreakerError:
                break  # Circuit opened
            except:
                pass

        # Should be blocked now
        with pytest.raises(CircuitBreakerError):
            bad_service()
