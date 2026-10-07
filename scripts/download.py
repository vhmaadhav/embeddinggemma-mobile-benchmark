"""Fetch the fp32 text-only ONNX export and tokenizer (~1.1 GB)."""

from huggingface_hub import hf_hub_download

from egbench.config import HF_MODEL_FILES, HF_MODEL_REPO, MODEL_DIR

for path in HF_MODEL_FILES:
    print(hf_hub_download(HF_MODEL_REPO, path, local_dir=MODEL_DIR))
