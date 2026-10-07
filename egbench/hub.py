"""Thin wrappers over qai_hub that log every submitted job to results/jobs.csv."""

import csv
from datetime import datetime, timezone
from pathlib import Path

import qai_hub as hub

from .config import RESULTS

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
    """Upload once and reuse the model id across compile jobs."""
    model = hub.upload_model(str(path))
    print(f"uploaded {label}: {model.model_id}")
    return model


def compile(model, device: str, label: str, options: str):
    job = hub.submit_compile_job(model, hub.Device(device), name=label, options=options)
    log(job, label, device, options)
    return job


def profile(target_model, device: str, label: str, options: str = ""):
    job = hub.submit_profile_job(target_model, hub.Device(device), name=label, options=options)
    log(job, label, device, options)
    return job


def infer(target_model, device: str, label: str, inputs: dict, options: str = ""):
    job = hub.submit_inference_job(target_model, hub.Device(device), inputs, name=label, options=options)
    log(job, label, device, options)
    return job
