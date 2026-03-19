# Urban Wind Solver — Python SDK

A Python client for the Urban Wind Solver API, providing programmatic access to AI-based urban wind field prediction.

## Installation

**From GitHub (recommended):**

```bash
pip install git+https://github.com/pic-wang/urbanwind-sdk.git
```

**From local source:**

```bash
git clone https://github.com/pic-wang/urbanwind-sdk.git
cd urbanwind-sdk
pip install .
```

## Quick Start

```python
from urbanwind import Client

client = Client(
    api_key="your-api-key",
    base_url="https://urbanwind.xyz",   # your server URL
)

# Predict wind field from an STL building geometry
job = client.predict("building.stl")
result = job.wait()                      # blocks until done
pred = result.download_pred_npy()        # numpy array [N, 3]
print(f"Predicted {pred.shape[0]} points")

# Query velocity at specific locations
velocities = client.query(
    job.job_id,
    points=[[10, 0, 2], [20, 5, 2], [50, 0, 5]],
)
for v in velocities:
    print(f"  ({v.x}, {v.y}, {v.z}) -> {v.speed:.2f} m/s")

# One-shot: upload STL + query in a single call
result = client.predict_at(
    stl="building.stl",
    points=[[10, 0, 2], [20, 5, 2]],
)
for v in result.velocities:
    print(f"  speed = {v.speed:.2f} m/s")

# Visualize:
from IPython.display import display, Image

# currently support xy-plane visualization
contour = client.coutour(
    jobs.job_id,
    z=2.0 # define the evaluation height, in unit (m)
)
contour_path = contour.save_png("contour.png")
display(Image(filename="contour.png"))
```

## All Methods

| Method | Description |
|--------|-------------|
| `client.predict(stl)` | Async predict → `PredictJob` (call `.wait()` for result) |
| `client.predict_sync(stl)` | Synchronous predict → `PredictResult` directly |
| `client.predict_at(stl, points=...)` | One-shot: predict + query in one call |
| `client.query(job_id, points=...)` | Query velocity at specific coordinates |
| `client.query(job_id, points_file=...)` | Query velocity from a coordinates .txt file |
| `client.contour(job_id, z=2.0)` | Generate contour slice PNG |
| `client.cfd_case_pack(geometry, inflow_csv)` | Generate OpenFOAM case ZIP |
| `client.list_jobs()` | List all jobs |
| `client.get_job(job_id)` | Get job status |
| `client.delete_job(job_id)` | Delete a job |

## Configuration

| Parameter | Env Variable | Default |
|-----------|-------------|---------|
| `api_key` | `URBANWIND_API_KEY` | (required) |
| `base_url` | `URBANWIND_BASE_URL` | `http://127.0.0.1:8000` |
| `timeout` | — | `300` seconds |

```python
# Option 1: pass directly
client = Client(api_key="sk-xxx", base_url="https://urbanwind.xyz")

# Option 2: use environment variables
#   export URBANWIND_API_KEY=sk-xxx
#   export URBANWIND_BASE_URL=https://urbanwind.xyz
client = Client()
```

## Requirements

- Python >= 3.8
- `requests`
- `numpy`
