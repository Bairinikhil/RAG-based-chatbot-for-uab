"""
Rate Limiter for API Requests (especially Gemini API)
Implements token bucket algorithm with exponential backoff
"""

import time
import logging
from typing import Optional, Callable, Any
from threading import Lock
from functools import wraps
import random

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Token bucket rate limiter with exponential backoff
    Thread-safe implementation
    """

    def __init__(
        self,
        requests_per_minute: int = 2,
        burst_size: Optional[int] = None,
        max_retries: int = 3
    ):
        """
        Initialize rate limiter

        Args:
            requests_per_minute: Maximum requests per minute
            burst_size: Maximum burst size (defaults to requests_per_minute)
            max_retries: Maximum retry attempts with backoff
        """
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size or requests_per_minute
        self.max_retries = max_retries

        # Token bucket
        self.tokens = float(self.burst_size)
        self.last_update = time.time()
        self.tokens_per_second = requests_per_minute / 60.0

        # Thread safety
        self._lock = Lock()

        # Statistics
        self._total_requests = 0
        self._throttled_requests = 0
        self._failed_requests = 0

        logger.info(
            f"RateLimiter initialized: {requests_per_minute} req/min, "
            f"burst={self.burst_size}, max_retries={max_retries}"
        )

    def _refill_tokens(self):
        """Refill tokens based on time elapsed"""
        now = time.time()
        elapsed = now - self.last_update
        self.tokens = min(
            self.burst_size,
            self.tokens + elapsed * self.tokens_per_second
        )
        self.last_update = now

    def acquire(self, tokens: int = 1, block: bool = True) -> bool:
        """
        Acquire tokens from the bucket

        Args:
            tokens: Number of tokens to acquire
            block: Whether to block until tokens available

        Returns:
            True if tokens acquired, False otherwise
        """
        with self._lock:
            self._refill_tokens()

            if self.tokens >= tokens:
                self.tokens -= tokens
                self._total_requests += 1
                return True

            if not block:
                self._throttled_requests += 1
                return False

            # Calculate wait time
            tokens_needed = tokens - self.tokens
            wait_time = tokens_needed / self.tokens_per_second

            logger.debug(f"Rate limit reached, waiting {wait_time:.2f}s")
            self._throttled_requests += 1

        # Wait outside the lock
        time.sleep(wait_time)

        # Try again
        with self._lock:
            self._refill_tokens()
            if self.tokens >= tokens:
                self.tokens -= tokens
                self._total_requests += 1
                return True
            return False

    def execute_with_backoff(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> tuple[bool, Any, Optional[str]]:
        """
        Execute function with rate limiting and exponential backoff

        Args:
            func: Function to execute
            *args, **kwargs: Arguments to pass to function

        Returns:
            Tuple of (success, result, error_message)
        """
        last_error = None
        base_delay = 1.0  # Start with 1 second

        for attempt in range(self.max_retries + 1):
            try:
                # Acquire token (blocking)
                self.acquire(tokens=1, block=True)

                # Execute function
                result = func(*args, **kwargs)
                return True, result, None

            except Exception as e:
                last_error = str(e)
                error_lower = last_error.lower()

                # Check if it's a rate limit error
                is_rate_limit = any(
                    phrase in error_lower
                    for phrase in ["rate limit", "quota", "too many requests", "429"]
                )

                if not is_rate_limit or attempt >= self.max_retries:
                    logger.error(f"Request failed (attempt {attempt + 1}): {e}")
                    self._failed_requests += 1
                    return False, None, last_error

                # Calculate backoff with jitter
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(
                    f"Rate limit hit (attempt {attempt + 1}/{self.max_retries + 1}), "
                    f"retrying in {delay:.2f}s"
                )
                time.sleep(delay)

        self._failed_requests += 1
        return False, None, f"Max retries exceeded: {last_error}"

    def get_stats(self) -> dict[str, Any]:
        """Get rate limiter statistics"""
        with self._lock:
            self._refill_tokens()
            return {
                "requests_per_minute": self.requests_per_minute,
                "burst_size": self.burst_size,
                "current_tokens": round(self.tokens, 2),
                "total_requests": self._total_requests,
                "throttled_requests": self._throttled_requests,
                "failed_requests": self._failed_requests,
                "throttle_rate_percent": (
                    round(self._throttled_requests / self._total_requests * 100, 2)
                    if self._total_requests > 0 else 0
                )
            }

    def reset_stats(self):
        """Reset statistics"""
        with self._lock:
            self._total_requests = 0
            self._throttled_requests = 0
            self._failed_requests = 0
            logger.info("Rate limiter statistics reset")


def rate_limited(limiter: RateLimiter):
    """
    Decorator to apply rate limiting to a function

    Usage:
        @rate_limited(my_rate_limiter)
        def my_api_call():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            success, result, error = limiter.execute_with_backoff(func, *args, **kwargs)
            if not success:
                raise Exception(f"Rate limited execution failed: {error}")
            return result
        return wrapper
    return decorator


# Global rate limiter instance for Gemini API
_gemini_rate_limiter: Optional[RateLimiter] = None


def get_gemini_rate_limiter() -> Optional[RateLimiter]:
    """Get global Gemini API rate limiter"""
    return _gemini_rate_limiter


def set_gemini_rate_limiter(limiter: RateLimiter):
    """Set global Gemini API rate limiter"""
    global _gemini_rate_limiter
    _gemini_rate_limiter = limiter
    logger.info("Global Gemini rate limiter set")


def init_gemini_rate_limiter(
    requests_per_minute: int = 2,
    burst_size: Optional[int] = None,
    max_retries: int = 3
) -> RateLimiter:
    """
    Initialize and set global Gemini rate limiter

    Args:
        requests_per_minute: Maximum requests per minute (default: 2 for free tier)
        burst_size: Maximum burst size
        max_retries: Maximum retry attempts

    Returns:
        RateLimiter instance
    """
    limiter = RateLimiter(
        requests_per_minute=requests_per_minute,
        burst_size=burst_size,
        max_retries=max_retries
    )
    set_gemini_rate_limiter(limiter)
    return limiter
