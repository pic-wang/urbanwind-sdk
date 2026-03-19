"""Exception hierarchy for the Urban Wind Solver Python SDK."""

from __future__ import annotations

from typing import Any, Dict, Optional


class UrbanWindError(Exception):
    """Base exception for all SDK errors."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        code: Optional[str] = None,
        request_id: Optional[str] = None,
        body: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.request_id = request_id
        self.body = body


class AuthenticationError(UrbanWindError):
    """Raised when the API key is missing or invalid (HTTP 401)."""
    pass


class NotFoundError(UrbanWindError):
    """Raised when a resource is not found (HTTP 404)."""
    pass


class TimeoutError(UrbanWindError):
    """Raised when a synchronous request or polling exceeds the timeout."""
    pass


class JobFailedError(UrbanWindError):
    """Raised when a prediction job ends with ``failed`` status."""
    pass


class ValidationError(UrbanWindError):
    """Raised for bad input or request validation errors (HTTP 4xx)."""
    pass
