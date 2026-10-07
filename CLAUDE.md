# Agent context

Working notes for coding agents. Keep short; update the status line when a step lands.

## Model facts
- Source: `onnx-community/embeddinggemma-2-ONNX`, `onnx/model.onnx` (+1.08 GB external data), fp32, text path.
- Inputs: `input_ids`, `attention_mask` (int64, dynamic) plus `image/video/audio_features` (unused for text; fed as empty `(0, 512)`).
- Output `sentence_embedding` is already pooled, projected to 768 and L2-normalised.
- Tokenizer adds `<bos>` … `<eos>`; pad id 0.
- 24 layers, hidden 512; full attention in layers 5/11/17/23, sliding elsewhere.

## Graph blockers for AI Hub (Step 2)
- ORT contrib ops: `com.microsoft.RotaryEmbedding` ×48, `SkipSimplifiedLayerNormalization` ×48; also `SimplifiedLayerNormalization` ×98 (ORT-only). Decompose to standard ONNX.
- Attention bias uses -3.4e38 (float min): saturates under quantization; replace with a moderate negative.
- Multimodal merge (`/model/multimodal_merge/*`): bypass for text-only.
- Dynamic shapes: fix to batch 1, seq 128 (later 512), then constant-fold.

## Rules
- Never fp16 (NaN). Check outputs for NaN.
- Re-normalise after MRL truncation. Always use task prefixes.
- Ask the user before large downloads or batches of AI Hub jobs.
- Log every AI Hub job to `results/jobs.csv`.
- Don't `pip install` into the Anaconda base env (a failed install broke `huggingface_hub` once).

## Status
Step 1 (CPU reference) in progress.
