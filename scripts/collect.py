"""Pull every logged (profiled) inference job into results/profile.csv.

    python -m scripts.collect
"""

from collections import Counter

import numpy as np
import pandas as pd
import qai_hub as hub

from egbench.config import RESULTS
from egbench.hub import JOB_LOG

rows = []
for job_row in pd.read_csv(JOB_LOG).query("kind == 'inference'").itertuples():
    job = hub.get_job(job_row.job_id)
    status = job.get_status()
    row = {"label": job_row.label, "device": job_row.device, "job_id": job_row.job_id, "status": status.code}
    if status.success:
        prof = job.download_profile()
        s = prof["execution_summary"]
        times_ms = np.asarray(s["all_inference_times"]) / 1e3
        units = Counter(layer["compute_unit"] for layer in prof["execution_detail"])
        row |= {
            "median_ms": np.median(times_ms),
            "p90_ms": np.percentile(times_ms, 90),
            "first_load_ms": s["first_load_time"] / 1e3,
            "warm_load_ms": s["warm_load_time"] / 1e3,
            "peak_mem_mb": s["estimated_inference_peak_memory"] / 2**20,
            "model_mb": job.model.get_size() / 2**20 if hasattr(job.model, "get_size") else None,
            "ops_npu": units.get("NPU", 0),
            "ops_gpu": units.get("GPU", 0),
            "ops_cpu": units.get("CPU", 0),
        }
    else:
        row["error"] = status.message
    rows.append(row)

out = RESULTS / "profile.csv"
pd.DataFrame(rows).round(2).to_csv(out, index=False)
print(pd.read_csv(out).to_string(index=False))
