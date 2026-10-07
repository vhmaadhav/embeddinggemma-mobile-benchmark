"""Paths and constants shared by every script."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "embeddinggemma-2"
SOURCE_ONNX = MODEL_DIR / "onnx" / "model.onnx"
TOKENIZER = MODEL_DIR / "tokenizer.json"
ARTIFACTS = ROOT / "artifacts"  # generated models (git-ignored)
DATA = ROOT / "data"  # cached eval sets (git-ignored)
RESULTS = ROOT / "results"  # committed CSVs and the AI Hub job log

HF_MODEL_REPO = "onnx-community/embeddinggemma-2-ONNX"
HF_MODEL_FILES = (
    "onnx/model.onnx",
    "onnx/model.onnx_data",
    "tokenizer.json",
    "tokenizer_config.json",
    "config.json",
)

SEQ_LEN = 128
HIDDEN = 768
MRL_DIMS = (768, 256)
PAD_ID = 0

# Task prefixes from the model card; quality drops sharply without them.
QUERY_PREFIX = "task: search result | query: "
STS_PREFIX = "task: sentence similarity | query: "
DOC_TEMPLATE = "title: {title} | text: {text}"  # title "none" when missing

RETRIEVAL_CORPUS_SIZE = 2000  # SciFact subset: all relevant docs + distractors
SEED = 0
