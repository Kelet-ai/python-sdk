"""Internal configuration state for Kelet SDK."""

import os
import threading
from typing import Optional

import httpx
from pydantic import BaseModel, PrivateAttr


class KeletConfig(BaseModel):
    """Internal configuration for Kelet SDK."""

    api_key: str
    base_url: str
    project: str

    _http_client: Optional[httpx.AsyncClient] = PrivateAttr(default=None)

    async def get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client for API requests."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                headers={"Authorization": self.api_key},
                timeout=30.0,
            )
        return self._http_client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None


# Module-level config (set by configure() or auto-created from env)
_config: Optional[KeletConfig] = None
_config_lock = threading.Lock()


def get_config() -> KeletConfig:
    """Get the current configuration.

    If configure() has not been called, attempts to create config from
    environment variables (KELET_API_KEY, KELET_PROJECT, KELET_API_URL).

    Thread-safe: uses lock to prevent race conditions.

    Raises:
        ValueError: If KELET_API_KEY environment variable is not set.
    """
    global _config

    if _config is not None:
        return _config

    with _config_lock:
        # Double-check after acquiring lock
        if _config is not None:
            return _config

        # Auto-create from environment variables
        api_key = os.environ.get("KELET_API_KEY")
        if not api_key:
            raise ValueError(
                "KELET_API_KEY required. Set KELET_API_KEY env var or call configure()."
            )

        base_url = os.environ.get("KELET_API_URL", "https://api.kelet.ai")
        base_url = base_url.removesuffix("/api").rstrip("/")
        _config = KeletConfig(
            api_key=api_key,
            base_url=base_url,
            project=os.environ.get("KELET_PROJECT", "default"),
        )
        return _config


def set_config(config: KeletConfig) -> None:
    """Set the module-level configuration."""
    global _config
    with _config_lock:
        _config = config


def is_configured() -> bool:
    """Check if Kelet has been configured."""
    return _config is not None
