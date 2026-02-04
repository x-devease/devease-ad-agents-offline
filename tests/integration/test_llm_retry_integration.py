"""
Integration tests for LLM Retry Logic and Circuit Breaker.

Tests end-to-end functionality of retry logic, circuit breaker,
and their integration with the agent framework.
"""

import pytest
import time
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from src.utils.llm_retry import (
    RetryConfig,
    CircuitBreakerConfig,
    CircuitBreaker,
    RetryManager,
    LLMClientWithRetry,
    CircuitBreakerError,
    TransientError,
    RateLimitError,
)


class TestRetryLogicIntegration:
    """Integration tests for retry logic with realistic scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.retry_config = RetryConfig(
            max_attempts=3,
            initial_backoff=0.1,  # Fast for tests
            exponential_base=2.0,
            jitter=False,  # Predictable for tests
        )
        self.manager = RetryManager(self.retry_config)

    def test_retry_with_transient_failure(self):
        """Test retry succeeds after transient failure."""
        attempt_count = [0]

        def flaky_operation():
            attempt_count[0] += 1
            if attempt_count[0] < 2:
                raise TransientError("Temporary failure")
            return "success"

        result = self.manager.retry(flaky_operation)()

        assert result == "success"
        assert attempt_count[0] == 2

    def test_retry_exhaustion(self):
        """Test retry gives up after max attempts."""
        attempt_count = [0]

        def always_failing_operation():
            attempt_count[0] += 1
            raise TransientError("Always fails")

        with pytest.raises(TransientError):
            self.manager.retry(always_failing_operation)()

        assert attempt_count[0] == 3  # max_attempts

    def test_retry_with_different_errors(self):
        """Test retry with different error types."""
        config = RetryConfig(
            max_attempts=3,
            initial_backoff=0.1,
            retry_on=(TransientError, RateLimitError),
        )
        manager = RetryManager(config)

        # Should retry
        transient_count = [0]

        def transient_operation():
            transient_count[0] += 1
            if transient_count[0] < 2:
                raise TransientError("Transient")
            return "success"

        result = manager.retry(transient_operation)()
        assert result == "success"

        # Should NOT retry (different error type)
        def value_error_operation():
            raise ValueError("Not retryable")

        with pytest.raises(ValueError):
            manager.retry(value_error_operation)()


class TestCircuitBreakerIntegration:
    """Integration tests for circuit breaker with realistic scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            timeout=1.0,  # 1 second for tests
            window_size=10,
        )
        self.circuit_breaker = CircuitBreaker(self.config, "test_service")

    def test_circuit_opens_on_failures(self):
        """Test circuit opens after threshold failures."""
        # Record failures
        for i in range(3):
            self.circuit_breaker.record_failure(Exception(f"Error {i}"))

        # Circuit should be open
        assert self.circuit_breaker.state.value == "open"
        assert self.circuit_breaker.can_attempt() is False

    def test_circuit_recovers_after_timeout(self):
        """Test circuit transitions to half-open after timeout."""
        # Trip the circuit
        for i in range(3):
            self.circuit_breaker.record_failure(Exception(f"Error {i}"))

        assert self.circuit_breaker.state.value == "open"

        # Simulate timeout passing
        self.circuit_breaker.last_failure_time = datetime.now() - timedelta(seconds=2)

        # Check circuit state
        assert self.circuit_breaker.can_attempt() is True
        assert self.circuit_breaker.state.value == "half_open"

    def test_circuit_closes_on_successes(self):
        """Test circuit closes after successes in half-open state."""
        # Trip the circuit
        for i in range(3):
            self.circuit_breaker.record_failure(Exception(f"Error {i}"))

        # Move to half-open
        self.circuit_breaker.last_failure_time = datetime.now() - timedelta(seconds=2)
        self.circuit_breaker.can_attempt()

        # Record successes
        for i in range(2):
            self.circuit_breaker.record_success()

        # Circuit should be closed
        assert self.circuit_breaker.state.value == "closed"

    def test_circuit_stats_tracking(self):
        """Test circuit breaker tracks statistics correctly."""
        # Record some calls
        self.circuit_breaker.record_success()
        self.circuit_breaker.record_success()
        self.circuit_breaker.record_failure(Exception("Error"))

        stats = self.circuit_breaker.get_stats()

        assert stats["recent_successes"] == 2
        assert stats["recent_failures"] == 1
        assert stats["total_calls"] == 3
        assert 0.0 < stats["failure_rate"] < 1.0


class TestLLMClientWithRetryIntegration:
    """Integration tests for LLM client wrapper."""

    def setup_method(self):
        """Set up test fixtures."""
        self.retry_config = RetryConfig(
            max_attempts=3,
            initial_backoff=0.1,
        )
        self.circuit_config = CircuitBreakerConfig(
            failure_threshold=3,
            timeout=1.0,
        )

    def test_successful_call_with_retry(self):
        """Test successful call passes through wrapper."""
        mock_client = Mock()
        mock_client.generate.return_value = {"result": "success"}

        wrapper = LLMClientWithRetry(
            mock_client,
            "test_service",
            retry_config=self.retry_config,
            circuit_config=self.circuit_config,
        )

        result = wrapper.call_with_retry("generate")

        assert result == {"result": "success"}
        assert mock_client.generate.call_count == 1

    def test_retry_on_transient_failure(self):
        """Test wrapper retries on transient failure."""
        mock_client = Mock()
        mock_client.generate.side_effect = [
            TransientError("First fail"),
            {"result": "success"},
        ]

        wrapper = LLMClientWithRetry(
            mock_client,
            "test_service",
            retry_config=self.retry_config,
            circuit_config=self.circuit_config,
        )

        result = wrapper.call_with_retry("generate")

        assert result == {"result": "success"}
        assert mock_client.generate.call_count == 2

    def test_circuit_opens_on_repeated_failures(self):
        """Test circuit opens after repeated failures."""
        mock_client = Mock()
        mock_client.generate.side_effect = Exception("Always fails")

        wrapper = LLMClientWithRetry(
            mock_client,
            "test_service",
            retry_config=RetryConfig(max_attempts=2, initial_backoff=0.1),
            circuit_config=CircuitBreakerConfig(failure_threshold=2, timeout=1.0),
        )

        # Try multiple times
        for i in range(5):
            try:
                wrapper.call_with_retry("generate")
            except:
                pass

        # Circuit should be open
        assert wrapper.circuit_breaker.state.value == "open"

        # Next call should be blocked
        with pytest.raises(CircuitBreakerError):
            wrapper.call_with_retry("generate")

    def test_get_stats(self):
        """Test getting statistics from wrapper."""
        mock_client = Mock()
        mock_client.generate.return_value = {"result": "success"}

        wrapper = LLMClientWithRetry(
            mock_client,
            "test_service",
            retry_config=self.retry_config,
            circuit_config=self.circuit_config,
        )

        # Make a successful call
        wrapper.call_with_retry("generate")

        # Get stats
        stats = wrapper.get_stats()

        assert stats["service"] == "test_service"
        assert "circuit_breaker" in stats
        assert "retry_config" in stats


class TestCircuitBreakerRecoveryScenarios:
    """Test circuit breaker recovery in various scenarios."""

    def test_fast_recovery(self):
        """Test circuit recovers quickly after service returns."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            timeout=0.5,  # Short timeout
        )
        cb = CircuitBreaker(config, "fast_service")

        # Trip the circuit
        for i in range(3):
            cb.record_failure(Exception(f"Error {i}"))

        assert cb.state.value == "open"

        # Wait for timeout
        time.sleep(0.6)

        # Service is back
        assert cb.can_attempt() is True
        assert cb.state.value == "half_open"

        # Record successes
        cb.record_success()
        cb.record_success()

        # Circuit should close
        assert cb.state.value == "closed"

    def test_flaky_service(self):
        """Test circuit with flaky service (intermittent failures)."""
        config = CircuitBreakerConfig(
            failure_threshold=5,  # Higher threshold
            success_threshold=2,
            timeout=1.0,
            window_size=20,  # Larger window
        )
        cb = CircuitBreaker(config, "flaky_service")

        # Simulate flaky behavior
        results = [True, True, False, True, False, True, True, False, True, True]

        for success in results:
            if success:
                cb.record_success()
            else:
                cb.record_failure(Exception("Flaky failure"))

        # Circuit should remain closed despite some failures
        # (not enough consecutive failures to trip)
        assert cb.state.value == "closed"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
