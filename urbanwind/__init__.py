"""
Urban Wind Solver Python SDK
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A Python client for the Urban Wind Solver API.

Usage::

    from urbanwind import Client

    client = Client(api_key="sk-xxx", base_url="https://urbanwind.xyz")
    job = client.predict("building.stl")
    result = job.wait()
    pred = result.download_pred_npy()          # numpy [N, 3]
    velocities = client.query(job.job_id, points=[[10, 0, 2]])
"""

__version__ = "0.1.0"

from .client import Client
from .exceptions import (
    AuthenticationError,
    JobFailedError,
    NotFoundError,
    TimeoutError,
    UrbanWindError,
    ValidationError,
)
from .models import (
    ContourResult,
    PredictAtResult,
    PredictJob,
    PredictResult,
    VelocityPoint,
)

__all__ = [
    "Client",
    # Models
    "PredictJob",
    "PredictResult",
    "PredictAtResult",
    "VelocityPoint",
    "ContourResult",
    # Exceptions
    "UrbanWindError",
    "AuthenticationError",
    "NotFoundError",
    "TimeoutError",
    "JobFailedError",
    "ValidationError",
]
