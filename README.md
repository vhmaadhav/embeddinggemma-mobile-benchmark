# EmbeddingGemma 2 on mobile

How fast, how small and how accurate can Google's [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) text encoder be on a phone?

This repo compiles the text-only model (270M params) for Snapdragon NPUs with [Qualcomm AI Hub](https://workbench.aihub.qualcomm.com), profiles it on hosted devices, and walks down a quantization ladder (fp32 → W8A16 → W8A8 → mixed → W4), checking accuracy at every rung.

## Status

| Step | | |
|---|---|---|
| 1 | CPU accuracy reference | in progress |
| 2 | Static, NPU-friendly ONNX | — |
| 3 | First AI Hub compile + profile | — |
| 4 | Quantization ladder | — |
| 5 | Device sweep | — |
| 6 | Report | — |

## Quick start

```bash
pip install -r requirements.txt
python -m scripts.download
python -m scripts.evaluate --name fp32-reference --model models/embeddinggemma-2/onnx/model.onnx
```

AI Hub steps need `qai-hub configure --api_token ...` once.

## Layout

```
egbench/   library: config, ONNX Runtime encoder, eval suite
scripts/   one entry point per step
results/   committed CSVs and the AI Hub job log
```

## Method notes

- **Accuracy**: STS-B test (Spearman) and a fixed 2,000-doc SciFact subset (NDCG@10), at 768-d and MRL-truncated 256-d, with the model's task prefixes. Inputs are truncated to 128 tokens. Absolute scores are not MTEB scores; the deltas between rungs are what matter.
- **No fp16.** The model's activations overflow in fp16. Every float run here is fp32.
- **Hosted devices** report latency and memory only, no energy or thermals.

## Licence

Code: MIT. Model: Apache-2.0 (Google).
