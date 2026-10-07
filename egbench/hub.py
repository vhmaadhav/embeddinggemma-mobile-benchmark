"""Thin wrappers over qai_hub that log every submitted job to results/jobs.csv."""

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import qai_hub as hub

from .config import ARTIFACTS, RESULTS

# qai_hub prints emoji progress; the default Windows console codec chokes on it.
sys.stdout.reconfigure(encoding="utf-8")

JOB_LOG = RESULTS / "jobs.csv"
FIELDS = ("submitted", "label", "kind", "job_id", "device", "options", "url")


def log(job, label: str, device: str = "", options: str = "") -> None:
    new = not JOB_LOG.exists()
    JOB_LOG.parent.mkdir(exist_ok=True)
    with open(JOB_LOG, "a", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, FIELDS)
        if new:
            writer.writeheader()
        writer.writerow({
            "submitted": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "label": label,
            "kind": type(job).__name__.removesuffix("Job").lower(),
            "job_id": job.job_id,
            "device": device,
            "options": options,
            "url": job.url,
        })


def upload(path: Path, label: str):
    """Upload once per file version; later calls reuse the cached model id."""
    cache = ARTIFACTS / "uploads.json"
    ids = json.loads(cache.read_text()) if cache.exists() else {}
    key = f"{path.resolve()}@{path.stat().st_mtime_ns}"
    if key in ids:
        return hub.get_model(ids[key])
    model = hub.upload_model(str(path))
    print(f"uploaded {label}: {model.model_id}")
    ids[key] = model.model_id
    cache.write_text(json.dumps(ids, indent=1))
    return model


def compile(model, device: str, label: str, options: str):
    job = hub.submit_compile_job(model, hub.Device(device), name=label, options=options)
    log(job, label, device, options)
    return job


def quantize(model, calibration: dict, label: str, weights: str, activations: str, options: str = ""):
    job = hub.submit_quantize_job(
        model,
        calibration,
        weights_dtype=hub.QuantizeDtype[weights.upper()],
        activations_dtype=hub.QuantizeDtype[activations.upper()],
        name=label,
        options=options,
    )
    log(job, label, options=f"w={weights} a={activations} {options}".strip())
    return job


def infer(target_model, device: str, label: str, inputs: dict, options: str = ""):
    """Inference jobs also profile (latency, memory, per-op compute unit)."""
    job = hub.submit_inference_job(target_model, hub.Device(device), inputs, name=label, options=options, profile=True)
    log(job, label, device, options)
    return job
