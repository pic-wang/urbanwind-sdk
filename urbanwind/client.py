"""Main client class for the Urban Wind Solver Python SDK.

Usage::

    from urbanwind import Client

    client = Client(api_key="sk-xxx", base_url="https://urbanwind.xyz")
    job = client.predict("building.stl")
    result = job.wait()
    pred = result.download_pred_npy()
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ._http import HttpClient
from .exceptions import JobFailedError, UrbanWindError
from .exceptions import TimeoutError as UWTimeout
from .models import (
    ContourResult,
    PredictAtResult,
    PredictJob,
    PredictResult,
    VelocityPoint,
)


class Client:
    """Urban Wind Solver API client.

    Parameters
    ----------
    api_key : str, optional
        Bearer token. Falls back to ``URBANWIND_API_KEY`` env var.
    base_url : str, optional
        Server base URL. Falls back to ``URBANWIND_BASE_URL`` env var.
    timeout : float
        Default request timeout in seconds.
    max_retries : int
        Retry count for transient failures.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 300.0,
        max_retries: int = 3,
    ):
        self._api_key = api_key or os.getenv("URBANWIND_API_KEY", "")
        self._base_url = (
            base_url
            or os.getenv("URBANWIND_BASE_URL", "http://127.0.0.1:8000")
        ).rstrip("/")

        if not self._api_key:
            raise ValueError(
                "api_key is required. Pass it directly or set URBANWIND_API_KEY."
            )

        self._http = HttpClient(
            base_url=self._base_url,
            api_key=self._api_key,
            timeout=timeout,
            max_retries=max_retries,
        )

    # ------------------------------------------------------------------ #
    #  Core prediction                                                    #
    # ------------------------------------------------------------------ #

    def predict(self, stl: Union[str, Path]) -> PredictJob:
        """Submit an STL file for async prediction.

        Returns a :class:`PredictJob`.  Call ``job.wait()`` to block
        until inference completes.
        """
        stl = Path(stl)
        with open(stl, "rb") as fp:
            resp = self._http.post(
                "/api/v1/jobs/predict",
                files={"file": (stl.name, fp, "application/sla")},
            )
        body = self._http.json(resp)
        return PredictJob(
            job_id=body["job_id"],
            status=body["status"],
            created_at=body.get("created_at"),
            _poll_fn=self._poll_job,
            _result_fn=self._get_result,
        )

    def predict_sync(
        self,
        stl: Union[str, Path],
        timeout: float = 300.0,
    ) -> PredictResult:
        """
        Synchronous full-field prediction (blocks until done).

        Calls ``POST /api/v1/predict/field`` which blocks server-side.
        """
        stl = Path(stl)
        with open(stl, "rb") as fp:
            resp = self._http.post(
                "/api/v1/predict/field",
                files={"file": (stl.name, fp, "application/sla")},
                data={"timeout": str(timeout)},
                timeout=timeout + 30,  # HTTP timeout slightly longer
            )
        body = self._http.json(resp)
        return PredictResult(
            job_id=body["job_id"],
            n_points=body.get("n_points"),
            elapsed_ms=body.get("elapsed_ms"),
            artifacts=body.get("artifacts", {}),
            _download_fn=self._http.download,
        )

    # ------------------------------------------------------------------ #
    #  Query velocity                                                     #
    # ------------------------------------------------------------------ #

    def query(
        self,
        job_id: str,
        *,
        points: Optional[List[List[float]]] = None,
        points_file: Optional[Union[str, Path]] = None,
        output_path: Optional[Union[str, Path]] = None,
    ) -> List[VelocityPoint]:
        """
        Query wind velocity at specific coordinates.

        Provide *either* ``points`` (list of ``[x, y, z]``) or
        ``points_file`` (path to a coordinates text file).

        Parameters
        ----------
        output_path : str or Path, optional
            If provided, save the query results to this local file as
            plain text.  When ``points_file`` is used the server response
            (which preserves comments / blank lines from the input) is
            written directly; when ``points`` is used a text file is
            generated from the returned velocity data.
        """
        kwargs: Dict[str, Any] = {}

        if points_file is not None:
            points_file = Path(points_file)
            with open(points_file, "rb") as fp:
                kwargs["files"] = {"file": (points_file.name, fp, "text/plain")}
                resp = self._http.post(
                    f"/api/v1/jobs/{job_id}/query", **kwargs,
                )
            # Server returns plain text when input is a file
            if output_path is not None:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(resp.text, encoding="utf-8")
            return self._parse_velocity_text(resp.text)
        elif points is not None:
            resp = self._http.post(
                f"/api/v1/jobs/{job_id}/query",
                data={"points_json": json.dumps(points)},
            )
        else:
            raise ValueError("Provide either points or points_file")

        body = self._http.json(resp)
        results = [VelocityPoint.from_dict(r) for r in body.get("results", [])]

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("# x y z  ux uy uz  speed\n")
                for v in results:
                    f.write(
                        f"{v.x:.6f} {v.y:.6f} {v.z:.6f}  "
                        f"{v.ux:.6f} {v.uy:.6f} {v.uz:.6f}  "
                        f"{v.speed:.6f}\n"
                    )

        return results

    @staticmethod
    def _parse_velocity_text(text: str) -> List[VelocityPoint]:
        """Parse plain-text velocity output (one ``x y z ux uy uz speed`` per line)."""
        results: List[VelocityPoint] = []
        for line in text.splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 7:
                continue
            results.append(VelocityPoint(
                x=float(parts[0]), y=float(parts[1]), z=float(parts[2]),
                ux=float(parts[3]), uy=float(parts[4]), uz=float(parts[5]),
                speed=float(parts[6]),
            ))
        return results

    # ------------------------------------------------------------------ #
    #  Contour                                                            #
    # ------------------------------------------------------------------ #

    def contour(
        self,
        job_id: str,
        z: float = 2.0,
        component: int = -1,
        n_levels: int = 8,
    ) -> ContourResult:
        """Generate a contour-slice PNG for a completed job."""
        resp = self._http.post(
            f"/api/v1/jobs/{job_id}/contour",
            data={
                "z": str(z),
                "component": str(component),
                "n_levels": str(n_levels),
            },
        )
        body = self._http.json(resp)
        return ContourResult(
            job_id=body["job_id"],
            z=body["z"],
            vmin=body["vmin"],
            vmax=body["vmax"],
            elapsed_ms=body["elapsed_ms"],
            png_url=body.get("png_url"),
            overlay_png_url=body.get("overlay_png_url"),
            _download_fn=self._http.download,
        )

    # ------------------------------------------------------------------ #
    #  One-shot predict + query                                           #
    # ------------------------------------------------------------------ #

    def predict_at(
        self,
        stl: Union[str, Path],
        *,
        points: Optional[List[List[float]]] = None,
        points_file: Optional[Union[str, Path]] = None,
        timeout: float = 300.0,
    ) -> PredictAtResult:
        """One-shot: upload STL + query coordinates → get velocities back.

        Calls ``POST /api/v1/predict`` which blocks until inference + query
        are complete.
        """
        stl = Path(stl)
        files: Dict[str, Any] = {}
        data: Dict[str, str] = {"timeout": str(timeout)}

        with open(stl, "rb") as fp:
            files["file"] = (stl.name, fp, "application/sla")

            if points_file is not None:
                points_file = Path(points_file)
                with open(points_file, "rb") as qfp:
                    files["query_file"] = (points_file.name, qfp, "text/plain")
                    resp = self._http.post(
                        "/api/v1/predict",
                        files=files, data=data,
                        timeout=timeout + 30,
                    )
            elif points is not None:
                data["points_json"] = json.dumps(points)
                resp = self._http.post(
                    "/api/v1/predict",
                    files=files, data=data,
                    timeout=timeout + 30,
                )
            else:
                # No query — just sync predict
                resp = self._http.post(
                    "/api/v1/predict",
                    files=files, data=data,
                    timeout=timeout + 30,
                )

        body = self._http.json(resp)
        velocities = [
            VelocityPoint.from_dict(r)
            for r in body.get("query_results", [])
        ]
        return PredictAtResult(
            job_id=body["job_id"],
            n_points_field=body.get("n_points_field"),
            elapsed_ms=body.get("elapsed_ms"),
            velocities=velocities,
            download=body.get("download", {}),
            _download_fn=self._http.download,
        )

    # ------------------------------------------------------------------ #
    #  CFD case pack                                                      #
    # ------------------------------------------------------------------ #

    def cfd_case_pack(
        self,
        geometry: Union[str, Path],
        inflow_csv: Union[str, Path],
        mesh_resolution: str = "std",
        output_zip: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Generate an OpenFOAM case ZIP and download it.

        Returns the path to the saved ZIP file.
        """
        geometry = Path(geometry)
        inflow_csv = Path(inflow_csv)
        out = Path(output_zip or f"{geometry.stem}_case.zip")

        with open(geometry, "rb") as gfp, open(inflow_csv, "rb") as ifp:
            resp = self._http.post(
                "/api/v1/cfd/case-pack",
                files={
                    "geometry": (geometry.name, gfp, "application/sla"),
                    "inflow_csv": (inflow_csv.name, ifp, "text/csv"),
                },
                data={"mesh_resolution": mesh_resolution},
            )

        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(resp.content)
        return out

    # ------------------------------------------------------------------ #
    #  Job management                                                     #
    # ------------------------------------------------------------------ #

    def get_job(self, job_id: str) -> Dict[str, Any]:
        """Get the current status of a job."""
        return self._poll_job(job_id)

    def list_jobs(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List all jobs with pagination."""
        resp = self._http.get(
            "/api/v1/jobs",
            params={"limit": limit, "offset": offset},
        )
        return self._http.json(resp)

    def delete_job(self, job_id: str) -> bool:
        """Delete a job and its artifacts."""
        resp = self._http.delete(f"/api/v1/jobs/{job_id}")
        return self._http.json(resp).get("deleted", False)

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _poll_job(self, job_id: str) -> Dict[str, Any]:
        resp = self._http.get(f"/api/v1/jobs/{job_id}")
        return self._http.json(resp)

    def _get_result(self, job_id: str) -> PredictResult:
        resp = self._http.get(f"/api/v1/jobs/{job_id}/result")
        body = self._http.json(resp)
        return PredictResult(
            job_id=body["job_id"],
            n_points=body.get("summary", {}).get("n_points"),
            elapsed_ms=body.get("summary", {}).get("elapsed_ms"),
            artifacts=body.get("artifacts", {}),
            _download_fn=self._http.download,
        )

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self._http.close()

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
