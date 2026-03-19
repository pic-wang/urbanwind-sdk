"""Response models for the Urban Wind Solver Python SDK.

All models are plain data classes — no network calls happen at import
time and there are no hard dependencies beyond ``requests`` (and
``numpy`` for downloads).
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    import numpy as np


# ------------------------------------------------------------------ #
#  Velocity point                                                     #
# ------------------------------------------------------------------ #

@dataclass
class VelocityPoint:
    """A single velocity measurement at a 3-D location."""
    x: float
    y: float
    z: float
    ux: float
    uy: float
    uz: float
    speed: float

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VelocityPoint":
        return cls(
            x=float(d["x"]), y=float(d["y"]), z=float(d["z"]),
            ux=float(d["ux"]), uy=float(d["uy"]), uz=float(d["uz"]),
            speed=float(d["speed"]),
        )


# ------------------------------------------------------------------ #
#  Predict result                                                     #
# ------------------------------------------------------------------ #

@dataclass
class PredictResult:
    """Result of a completed prediction job."""
    job_id: str
    n_points: Optional[int] = None
    elapsed_ms: Optional[int] = None
    artifacts: Dict[str, Optional[str]] = field(default_factory=dict)

    # Injected by the Client so download helpers can call back
    _download_fn: Optional[Callable[..., bytes]] = field(default=None, repr=False)

    def download_zip(self, path: str) -> Path:
        """Download the full result ZIP to *path*."""
        url = self.artifacts.get("result_zip") or self.artifacts.get("zip_url")
        if not url:
            raise ValueError("No ZIP artifact URL available")
        data = self._download_fn(url)  # type: ignore[misc]
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        return out

    def download_pred_npy(self) -> "np.ndarray":
        """Download ``pred.npy`` and return it as a numpy array."""
        import numpy as _np

        url = self.artifacts.get("pred_npy") or self.artifacts.get("pred_npy_url")
        if not url:
            raise ValueError("No pred.npy artifact URL available")
        data = self._download_fn(url)  # type: ignore[misc]
        return _np.load(io.BytesIO(data))

    def download_pred_mesh_vtu(self, path: str) -> Path:
        """Download ``pred_mesh.vtu`` to *path*."""
        url = self.artifacts.get("pred_mesh_vtu") or self.artifacts.get("pred_mesh_vtu_url")
        if not url:
            raise ValueError("No pred_mesh.vtu artifact URL available")
        data = self._download_fn(url)  # type: ignore[misc]
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        return out


# ------------------------------------------------------------------ #
#  Predict job (returned before completion)                           #
# ------------------------------------------------------------------ #

@dataclass
class PredictJob:
    """Handle to an async prediction job.

    Call :meth:`wait` to block until inference completes.
    """
    job_id: str
    status: str
    created_at: Optional[str] = None

    # Injected by Client
    _poll_fn: Optional[Callable[..., Dict[str, Any]]] = field(default=None, repr=False)
    _result_fn: Optional[Callable[..., PredictResult]] = field(default=None, repr=False)

    def refresh(self) -> None:
        """Poll the server for the latest status."""
        data = self._poll_fn(self.job_id)  # type: ignore[misc]
        self.status = data.get("status", self.status)

    def wait(
        self,
        poll_interval: float = 2.0,
        timeout: float = 900.0,
    ) -> PredictResult:
        """Block until the job finishes, then return a :class:`PredictResult`.

        Raises :class:`~urbanwind.exceptions.TimeoutError` if the job
        does not complete within *timeout* seconds, or
        :class:`~urbanwind.exceptions.JobFailedError` if it fails.
        """
        import time

        from .exceptions import JobFailedError
        from .exceptions import TimeoutError as UWTimeout

        start = time.time()
        while True:
            self.refresh()
            if self.status == "succeeded":
                return self._result_fn(self.job_id)  # type: ignore[misc]
            if self.status in ("failed", "expired"):
                raise JobFailedError(
                    f"Job {self.job_id} ended with status={self.status}",
                    code=self.status,
                )
            if time.time() - start > timeout:
                raise UWTimeout(
                    f"Job {self.job_id} did not complete within {timeout}s",
                )
            time.sleep(poll_interval)


# ------------------------------------------------------------------ #
#  Contour result                                                     #
# ------------------------------------------------------------------ #

@dataclass
class ContourResult:
    """Result of a contour generation request."""
    job_id: str
    z: float
    vmin: float
    vmax: float
    elapsed_ms: int
    png_url: Optional[str] = None
    overlay_png_url: Optional[str] = None

    _download_fn: Optional[Callable[..., bytes]] = field(default=None, repr=False)

    def save_png(self, path: str) -> Path:
        """Download the contour PNG to *path*."""
        if not self.png_url:
            raise ValueError("No contour PNG URL available")
        data = self._download_fn(self.png_url)  # type: ignore[misc]
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        return out


# ------------------------------------------------------------------ #
#  One-shot predict result                                            #
# ------------------------------------------------------------------ #

@dataclass
class PredictAtResult:
    """Result of a one-shot predict-at-locations request."""
    job_id: str
    n_points_field: Optional[int] = None
    elapsed_ms: Optional[int] = None
    velocities: List[VelocityPoint] = field(default_factory=list)
    download: Dict[str, Optional[str]] = field(default_factory=dict)

    _download_fn: Optional[Callable[..., bytes]] = field(default=None, repr=False)

    def download_zip(self, path: str) -> Path:
        url = self.download.get("result_zip")
        if not url:
            raise ValueError("No ZIP URL available")
        data = self._download_fn(url)  # type: ignore[misc]
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        return out

    def download_pred_npy(self) -> "np.ndarray":
        import numpy as _np

        url = self.download.get("pred_npy")
        if not url:
            raise ValueError("No pred.npy URL available")
        data = self._download_fn(url)  # type: ignore[misc]
        return _np.load(io.BytesIO(data))
