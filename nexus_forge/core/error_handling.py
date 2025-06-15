"""
Error handling framework with retry logic and circuit breaker pattern.

This module provides robust error handling utilities including:
- Retry decorators with exponential backoff and jitter
- Circuit breaker pattern for failing services
- Error categorization and handling strategies
"""

import asyncio
import functools
import logging
import random
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, Optional, Type, TypeVar, Union

from nexus_forge.core.exceptions import (
    NexusForgeError,
    ServiceUnavailableError,
    RateLimitError,
    AuthenticationError
)

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Service is failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service has recovered


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for fault tolerance.
    
    Prevents cascading failures by temporarily blocking calls to failing services.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: Type[Exception] = Exception
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before testing recovery
            expected_exception: Exception type to count as failure
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self._failure_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._state = CircuitState.CLOSED
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        if self._state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._state = CircuitState.HALF_OPEN
        return self._state
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        return (
            self._last_failure_time is not None and
            datetime.now() - self._last_failure_time > timedelta(seconds=self.recovery_timeout)
        )
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            ServiceUnavailableError: If circuit is open
        """
        if self.state == CircuitState.OPEN:
            raise ServiceUnavailableError(
                f"Circuit breaker is open. Service unavailable. "
                f"Will retry after {self.recovery_timeout} seconds."
            )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    async def async_call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute async function with circuit breaker protection.
        
        Args:
            func: Async function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            ServiceUnavailableError: If circuit is open
        """
        if self.state == CircuitState.OPEN:
            raise ServiceUnavailableError(
                f"Circuit breaker is open. Service unavailable. "
                f"Will retry after {self.recovery_timeout} seconds."
            )
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        """Handle successful call."""
        self._failure_count = 0
        self._state = CircuitState.CLOSED
    
    def _on_failure(self):
        """Handle failed call."""
        self._failure_count += 1
        self._last_failure_time = datetime.now()
        
        if self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            logger.warning(
                f"Circuit breaker opened after {self._failure_count} failures. "
                f"Will retry after {self.recovery_timeout} seconds."
            )


def exponential_backoff_with_jitter(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter_range: float = 0.1
) -> float:
    """
    Calculate exponential backoff delay with jitter.
    
    Args:
        attempt: Current attempt number (0-based)
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        jitter_range: Jitter range as fraction of delay (0.1 = ±10%)
        
    Returns:
        Delay in seconds
    """
    # Calculate exponential delay
    delay = min(base_delay * (2 ** attempt), max_delay)
    
    # Add jitter to prevent thundering herd
    jitter = delay * jitter_range * (2 * random.random() - 1)
    
    return max(0, delay + jitter)


def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential: bool = True,
    exceptions: tuple = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None
):
    """
    Retry decorator with configurable backoff strategy.
    
    Args:
        max_attempts: Maximum number of attempts
        base_delay: Base delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exponential: Use exponential backoff if True, linear if False
        exceptions: Tuple of exceptions to retry on
        on_retry: Optional callback on retry (exception, attempt)
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    # Check if this is a permanent error
                    if isinstance(e, (AuthenticationError,)):
                        logger.error(f"Permanent error in {func.__name__}: {e}")
                        raise
                    
                    if attempt < max_attempts - 1:
                        if exponential:
                            delay = exponential_backoff_with_jitter(
                                attempt, base_delay, max_delay
                            )
                        else:
                            delay = base_delay
                        
                        logger.warning(
                            f"Retry {attempt + 1}/{max_attempts} for {func.__name__} "
                            f"after {delay:.2f}s delay. Error: {e}"
                        )
                        
                        if on_retry:
                            on_retry(e, attempt + 1)
                        
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"Max retries ({max_attempts}) exceeded for {func.__name__}. "
                            f"Last error: {e}"
                        )
            
            raise last_exception
        
        return wrapper
    return decorator


def async_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential: bool = True,
    exceptions: tuple = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None
):
    """
    Async retry decorator with configurable backoff strategy.
    
    Args:
        max_attempts: Maximum number of attempts
        base_delay: Base delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exponential: Use exponential backoff if True, linear if False
        exceptions: Tuple of exceptions to retry on
        on_retry: Optional callback on retry (exception, attempt)
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    # Check if this is a permanent error
                    if isinstance(e, (AuthenticationError,)):
                        logger.error(f"Permanent error in {func.__name__}: {e}")
                        raise
                    
                    if attempt < max_attempts - 1:
                        if exponential:
                            delay = exponential_backoff_with_jitter(
                                attempt, base_delay, max_delay
                            )
                        else:
                            delay = base_delay
                        
                        logger.warning(
                            f"Retry {attempt + 1}/{max_attempts} for {func.__name__} "
                            f"after {delay:.2f}s delay. Error: {e}"
                        )
                        
                        if on_retry:
                            on_retry(e, attempt + 1)
                        
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"Max retries ({max_attempts}) exceeded for {func.__name__}. "
                            f"Last error: {e}"
                        )
            
            raise last_exception
        
        return wrapper
    return decorator


class ErrorHandler:
    """
    Centralized error handler with categorization and recovery strategies.
    """
    
    def __init__(self):
        """Initialize error handler."""
        self._error_counts: Dict[str, int] = {}
        self._recovery_strategies: Dict[Type[Exception], Callable] = {}
    
    def register_recovery_strategy(
        self,
        exception_type: Type[Exception],
        strategy: Callable[[Exception], Any]
    ):
        """
        Register a recovery strategy for an exception type.
        
        Args:
            exception_type: Exception type to handle
            strategy: Recovery strategy function
        """
        self._recovery_strategies[exception_type] = strategy
    
    def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """
        Handle an error with appropriate recovery strategy.
        
        Args:
            error: Exception to handle
            context: Optional context information
            
        Returns:
            Recovery result if applicable
        """
        error_type = type(error).__name__
        self._error_counts[error_type] = self._error_counts.get(error_type, 0) + 1
        
        # Log error with context
        logger.error(
            f"Error occurred: {error_type} - {str(error)}",
            extra={"context": context, "error_count": self._error_counts[error_type]}
        )
        
        # Apply recovery strategy if available
        for exc_type, strategy in self._recovery_strategies.items():
            if isinstance(error, exc_type):
                try:
                    return strategy(error)
                except Exception as recovery_error:
                    logger.error(
                        f"Recovery strategy failed for {error_type}: {recovery_error}"
                    )
        
        return None
    
    def get_error_stats(self) -> Dict[str, int]:
        """Get error statistics."""
        return self._error_counts.copy()


# Global error handler instance
error_handler = ErrorHandler()


def handle_ai_service_error(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to handle AI service errors with appropriate strategies.
    
    Provides unified error handling for AI service calls with:
    - Automatic retries for transient errors
    - Circuit breaking for persistent failures
    - Proper error categorization and logging
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    # Create circuit breaker for this function
    circuit_breaker = CircuitBreaker(
        failure_threshold=5,
        recovery_timeout=60,
        expected_exception=ServiceUnavailableError
    )
    
    @functools.wraps(func)
    @retry(
        max_attempts=3,
        exponential=True,
        exceptions=(
            ServiceUnavailableError,
            RateLimitError,
            ConnectionError,
            TimeoutError
        )
    )
    def wrapper(*args, **kwargs) -> T:
        try:
            # Execute with circuit breaker protection
            return circuit_breaker.call(func, *args, **kwargs)
        except NexusForgeError:
            # Re-raise framework errors
            raise
        except Exception as e:
            # Handle unexpected errors
            error_handler.handle_error(
                e,
                context={
                    "function": func.__name__,
                    "args": str(args)[:100],  # Truncate for logging
                    "kwargs": str(kwargs)[:100]
                }
            )
            raise ServiceUnavailableError(
                f"Unexpected error in {func.__name__}: {str(e)}"
            ) from e
    
    return wrapper


def handle_ai_service_error_async(func: Callable[..., T]) -> Callable[..., T]:
    """
    Async decorator to handle AI service errors with appropriate strategies.
    
    Args:
        func: Async function to decorate
        
    Returns:
        Decorated function
    """
    # Create circuit breaker for this function
    circuit_breaker = CircuitBreaker(
        failure_threshold=5,
        recovery_timeout=60,
        expected_exception=ServiceUnavailableError
    )
    
    @functools.wraps(func)
    @async_retry(
        max_attempts=3,
        exponential=True,
        exceptions=(
            ServiceUnavailableError,
            RateLimitError,
            ConnectionError,
            TimeoutError
        )
    )
    async def wrapper(*args, **kwargs) -> T:
        try:
            # Execute with circuit breaker protection
            return await circuit_breaker.async_call(func, *args, **kwargs)
        except NexusForgeError:
            # Re-raise framework errors
            raise
        except Exception as e:
            # Handle unexpected errors
            error_handler.handle_error(
                e,
                context={
                    "function": func.__name__,
                    "args": str(args)[:100],  # Truncate for logging
                    "kwargs": str(kwargs)[:100]
                }
            )
            raise ServiceUnavailableError(
                f"Unexpected error in {func.__name__}: {str(e)}"
            ) from e
    
    return wrapper