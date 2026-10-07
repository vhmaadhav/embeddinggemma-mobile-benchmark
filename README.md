# EmbeddingGemma 2 on mobile

How fast, how small and how accurate can Google's [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) text encoder be on a phone?

This repo compiles the text-only model (270M params) for Snapdragon NPUs with [Qualcomm AI Hub](https://workbench.aihub.qualcomm.com), profiles it on hosted devices, and walks down a quantization ladder (fp32 → W8A16 → W8A8 → mixed → W4), checking accuracy at every rung.

## Status

| Step | | |
|---|---|---|
| 1 | CPU accuracy reference | done |
| 2 | Static, NPU-friendly ONNX | done |
| 3 | First AI Hub compile + profile | blocked, see below |
| 4 | Quantization ladder | — |
| 5 | Device sweep | — |
| 6 | Report | — |

## Findings so far

**Accuracy reference** (fp32, ONNX Runtime CPU, seq 128), from [`results/accuracy.csv`](results/accuracy.csv):

| | 768-d | 256-d |
|---|---|---|
| STS-B Spearman | 0.875 | 0.874 |
| SciFact-subset NDCG@10 | 0.868 | 0.851 |

**Making the graph NPU-ready.** The upstream export targets ONNX Runtime: it uses ORT-only fused ops (RoPE, RMSNorm), a multimodal merge, dynamic shapes and a -3.4e38 attention mask. [`egbench/surgery.py`](egbench/surgery.py) rewrites it into static, standard ONNX. The rewritten graph matches the source at cosine 0.999999.

**First AI Hub runs** (Galaxy S24, Snapdragon 8 Gen 3, QNN DLC). All jobs are listed in [`results/jobs.csv`](results/jobs.csv).

| Graph | Compile | On device |
|---|---|---|
| Full fp32 (271M params, 1.08 GB) | ok | `QNN_COMMON_ERROR_MEM_ALLOC` |
| Transformer only, embedding lookup on host (137M, 552 MB) | ok | `QNN_COMMON_ERROR_MEM_ALLOC` |

The device log shows the HTP running float graphs in fp16 and rejecting rank-5 elementwise ops (`fp16 elementwise applied to tensor with rank != 4`). The attention block reshapes Q/K/V and the mask to 5-D.

**Next:** rewrite attention to 4-D tensors, measure the fp32 baseline on the CPU (fp16 on the NPU is invalid for this model), then quantize for the NPU (Step 4).

## Quick start

```bash
pip install -r requirements.txt
python -m scripts.download
python -m scripts.evaluate --name fp32-reference --model models/embeddinggemma-2/onnx/model.onnx
python -m scripts.prepare --seq-len 128 --embed-on-host
python -m scripts.hub_run --model artifacts/fp32-s128-split.onnx --label fp32-s128-split-npu
```

AI Hub steps need `qai-hub configure --api_token ...` once.

## Layout

```
egbench/   library: config, encoder, evals, graph surgery, AI Hub wrappers
scripts/   one entry point per step
results/   committed CSVs and the AI Hub job log
```

## Method notes

- **Accuracy**: STS-B test (Spearman) and a fixed 2,000-doc SciFact subset (NDCG@10), at 768-d and MRL-truncated 256-d, with the model's task prefixes. Inputs are truncated to 128 tokens. Absolute scores are not MTEB scores; the deltas between rungs are what matter.
- **No fp16.** The model's activations overflow in fp16. Every float run here is fp32.
- **Hosted devices** report latency and memory only, no energy or thermals.

## Licence

Code: MIT. Model: Apache-2.0 (Google).
