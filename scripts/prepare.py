"""Step 2: build the static AI Hub-ready graph and check it against the source.

    python -m scripts.prepare --seq-len 128 [--embed-on-host]
"""

import argparse

import numpy as np

from egbench.config import ARTIFACTS, QUERY_PREFIX, SOURCE_ONNX, STS_PREFIX
from egbench.encoder import Encoder
from egbench.evals import load_stsb
from egbench.surgery import build, op_histogram

parser = argparse.ArgumentParser()
parser.add_argument("--seq-len", type=int, default=128)
parser.add_argument("--embed-on-host", action="store_true", help="split the token lookup off the graph")
args = parser.parse_args()

name = f"fp32-s{args.seq_len}" + ("-split" if args.embed_on_host else "")
model = build(SOURCE_ONNX, ARTIFACTS / f"{name}.onnx", args.seq_len, args.embed_on_host)
print(model)
print(dict(op_histogram(model).most_common()))

texts = list(load_stsb().sentence1[:24])
ref, new = Encoder(SOURCE_ONNX, args.seq_len), Encoder(model, args.seq_len)
for prefix in (STS_PREFIX, QUERY_PREFIX):
    cos = (ref(texts, prefix) * new(texts, prefix)).sum(1)
    print(f"cosine vs source: min {cos.min():.6f}  mean {cos.mean():.6f}")
    assert cos.min() > 0.999, "static graph diverges from the source"
assert np.isfinite(cos).all()
