"""Score an ONNX model on CPU and upsert the row into results/accuracy.csv.

    python -m scripts.evaluate --name fp32-reference --model models/embeddinggemma-2/onnx/model.onnx
"""

import argparse
import time
from pathlib import Path

import pandas as pd

from egbench.config import RESULTS, SEQ_LEN
from egbench.encoder import Encoder
from egbench.evals import evaluate

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True, help="row label, e.g. w8a16")
parser.add_argument("--model", required=True, type=Path)
parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
args = parser.parse_args()

start = time.perf_counter()
row = {"name": args.name, "seq_len": args.seq_len, **evaluate(Encoder(args.model, args.seq_len))}
print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})
print(f"{time.perf_counter() - start:.0f}s")

csv = RESULTS / "accuracy.csv"
table = pd.read_csv(csv) if csv.exists() else pd.DataFrame()
if not table.empty:
    table = table[table.name != args.name]
csv.parent.mkdir(exist_ok=True)
pd.concat([table, pd.DataFrame([row])]).round(4).to_csv(csv, index=False)
