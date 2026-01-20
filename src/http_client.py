"""
Shared HTTP client with connection pooling.

Provides a singleton aiohttp ClientSession for efficient connection reuse
across the application.
"""

import aiohttp
import logging
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

# Singleton session instance
_session: aiohttp.ClientSession | None = None

# Default timeout configuration
DEFAULT_TIMEOUT = aiohttp.ClientTimeout(total=60, connect=10)

# Default headers
DEFAULT_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate',
}


async def get_session() -> aiohttp.ClientSession:
    """
    Get or create a shared aiohttp session for connection pooling.

    The session is created lazily on first call and reused for all subsequent requests.
    This provides significant performance improvements by reusing TCP connections.

    Returns:
        A configured aiohttp.ClientSession instance
    """
    global _session
    if _session is None or _session.closed:
        connector = aiohttp.TCPConnector(
            limit=100,  # Max concurrent connections
            limit_per_host=10,  # Max connections per host
            enable_cleanup_closed=True,
        )
        _session = aiohttp.ClientSession(
            connector=connector,
            timeout=DEFAULT_TIMEOUT,
            headers=DEFAULT_HEADERS,
        )
        logger.debug("Created new HTTP session with connection pooling")
    return _session


async def close_session() -> None:
    """
    Close the shared session.

    Should be called during application shutdown to cleanly close all connections.
    """
    global _session
    if _session and not _session.closed:
        await _session.close()
        logger.debug("Closed HTTP session")
        _session = None


@asynccontextmanager
async def http_session():
    """
    Async context manager for the HTTP session.

    Usage:
        async with http_session() as session:
            async with session.get(url) as response:
                data = await response.text()
    """
    session = await get_session()
    try:
        yield session
    finally:
        # Don't close here - session is reused
        pass


async def fetch_url(url: str, **kwargs) -> str:
    """
    Convenience function to fetch content from a URL.

    Args:
        url: The URL to fetch
        **kwargs: Additional arguments to pass to session.get()

    Returns:
        The response text content

    Raises:
        aiohttp.ClientError: If the request fails
    """
    session = await get_session()
    async with session.get(url, **kwargs) as response:
        response.raise_for_status()
        return await response.text()


async def fetch_json(url: str, **kwargs) -> dict:
    """
    Convenience function to fetch JSON from a URL.

    Args:
        url: The URL to fetch
        **kwargs: Additional arguments to pass to session.get()

    Returns:
        The parsed JSON response

    Raises:
        aiohttp.ClientError: If the request fails
    """
    session = await get_session()
    async with session.get(url, **kwargs) as response:
        response.raise_for_status()
        return await response.json()
