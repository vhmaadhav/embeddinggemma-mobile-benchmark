"""Step 4: quantize on AI Hub, download the fake-quant ONNX, score it on CPU.

    python -m scripts.quantize --weights int8 --activations int16
"""

import argparse
import subprocess
import sys

from egbench import hub
from egbench.config import ARTIFACTS
from egbench.encoder import Encoder
from egbench.evals import calibration_texts

parser = argparse.ArgumentParser()
parser.add_argument("--weights", default="int8", choices=["int4", "int8", "int16"])
parser.add_argument("--activations", default="int8", choices=["int8", "int16"])
parser.add_argument("--seq-len", type=int, default=128)
parser.add_argument("--samples", type=int, default=600)
parser.add_argument("--options", default="", help="extra quantize options, e.g. LiteMP")
parser.add_argument("--source", default="fp32-s128-split", help="artifact to quantize")
parser.add_argument("--label", help="defaults to w8a16-s128-split style")
args = parser.parse_args()

bits = lambda d: d.removeprefix("int")
label = args.label or f"w{bits(args.weights)}a{bits(args.activations)}" + args.source.removeprefix("fp32")
source = ARTIFACTS / f"{args.source}.onnx"

calibration = hub.dataset(Encoder(source, args.seq_len).feed(calibration_texts(args.samples)))

job = hub.quantize(hub.upload(source, "fp32"), calibration, label, args.weights, args.activations, args.options)
model = job.get_target_model()
if model is None:
    raise SystemExit(f"quantize failed: {job.get_status().message}\n{job.url}")
out = model.download(str(ARTIFACTS / f"{label}.onnx"))
print("downloaded", out)

subprocess.run(
    [sys.executable, "-m", "scripts.evaluate", "--name", label, "--model", str(out), "--seq-len", str(args.seq_len)],
    check=True,
)
