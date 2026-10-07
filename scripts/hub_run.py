"""Compile a model on AI Hub, then run a profiled inference job on device.

    python -m scripts.hub_run --model artifacts/fp32-s128.onnx --label fp32-s128 \
        --device "Samsung Galaxy S24 (Family)" --options "--target_runtime qnn_dlc"

Every job is logged to results/jobs.csv. The inference job runs a few real
sentences and reports cosine vs. ONNX Runtime on CPU, which catches NaN and
silent fp16 overflow.
"""

import argparse
from pathlib import Path

import numpy as np

from egbench import hub
from egbench.config import STS_PREFIX
from egbench.encoder import Encoder
from egbench.evals import load_stsb

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, type=Path)
parser.add_argument("--label", required=True)
parser.add_argument("--device", default="Samsung Galaxy S24 (Family)")
parser.add_argument("--options", default="--target_runtime qnn_dlc")
parser.add_argument("--run-options", default="", help="e.g. --compute_unit cpu")
args = parser.parse_args()

local = Encoder(args.model)
texts = [STS_PREFIX + t for t in load_stsb().sentence1[:4]]
inputs = hub.dataset(local.feed(texts))

compiled = hub.compile(hub.upload(args.model, args.label), args.device, args.label, args.options)
target = compiled.get_target_model()
if target is None:
    raise SystemExit(f"compile failed: {compiled.get_status().message}\n{compiled.url}")

run = hub.infer(target, args.device, args.label, inputs, args.run_options)

device_emb = np.concatenate(run.download_output_data()["sentence_embedding"])
ref = local(texts)
print("finite on device:", np.isfinite(device_emb).all())
norm = device_emb / np.linalg.norm(device_emb, axis=1, keepdims=True)
print("cosine vs CPU:", (norm * ref).sum(1).round(5))
print("profile:", run.url)
