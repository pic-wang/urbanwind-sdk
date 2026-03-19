"""Low-level HTTP helpers for the Urban Wind Solver Python SDK.

Wraps ``requests.Session`` with:
* Bearer token injection
* Configurable timeout
* Automatic retry on 429 / 5xx with exponential backoff
* Consistent error raising via SDK exceptions
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import requests

from .exceptions import (
    AuthenticationError,
    NotFoundError,
    UrbanWindError,
    ValidationError,
)


class HttpClient:
    """Thin wrapper around :class:`requests.Session`."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float = 300.0,
        max_retries: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "urbanwind-python-sdk/0.1.0",
        })

    # ---- public verbs ----

    def get(self, path: str, **kwargs: Any) -> requests.Response:
        return self._request("GET", path, **kwargs)

    def post(self, path: str, **kwargs: Any) -> requests.Response:
        return self._request("POST", path, **kwargs)

    def delete(self, path: str, **kwargs: Any) -> requests.Response:
        return self._request("DELETE", path, **kwargs)

    def download(self, path: str) -> bytes:
        """GET *path* and return the raw response bytes."""
        resp = self.get(path, stream=True)
        return resp.content

    # ---- internals ----

    def _request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        url = f"{self.base_url}{path}"
        kwargs.setdefault("timeout", self.timeout)

        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._session.request(method, url, **kwargs)
            except requests.ConnectionError as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 10))
                    continue
                raise UrbanWindError(f"Connection error: {exc}") from exc

            if resp.status_code == 429 or resp.status_code >= 500:
                last_exc = UrbanWindError(
                    f"HTTP {resp.status_code}: {resp.text[:200]}",
                    status_code=resp.status_code,
                )
                if attempt < self.max_retries:
                    retry_after = float(resp.headers.get("Retry-After", 2 ** attempt))
                    time.sleep(min(retry_after, 30))
                    continue

            self._raise_for_status(resp)
            return resp

        # Exhausted retries
        if last_exc is not None:
            raise last_exc
        raise UrbanWindError("Request failed after retries")  # pragma: no cover

    @staticmethod
    def _raise_for_status(resp: requests.Response) -> None:
        if resp.status_code < 400:
            return

        # Try to extract structured error
        body: Dict[str, Any] = {}
        try:
            body = resp.json()
        except Exception:
            pass

        error_obj = body.get("error", {})
        if isinstance(error_obj, str):
            error_obj = {"message": error_obj}
        code = error_obj.get("code", "")
        message = error_obj.get("message", resp.text[:300])
        request_id = body.get("request_id")

        kwargs = dict(
            status_code=resp.status_code,
            code=code,
            request_id=request_id,
            body=body,
        )

        if resp.status_code == 401:
            raise AuthenticationError(message, **kwargs)
        if resp.status_code == 404:
            raise NotFoundError(message, **kwargs)
        if 400 <= resp.status_code < 500:
            raise ValidationError(message, **kwargs)
        raise UrbanWindError(message, **kwargs)

    def close(self) -> None:
        self._session.close()
