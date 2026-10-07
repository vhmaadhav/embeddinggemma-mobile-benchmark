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

## AI Hub (qai-hub 0.56)
- Target runtimes: `qnn_dlc`, `precompiled_qnn_onnx`, `onnx`, `tflite` (`qnn_context_binary` is gone).
- `submit_profile_job` is deprecated; `submit_inference_job(..., profile=True)` gives outputs and profile in one job.
- Upload ONNX with external data as a directory `name.onnx/{model.onnx, model.data}`. Upload ids are cached in `artifacts/uploads.json`.
- Set UTF-8 stdout on Windows; qai_hub's progress emoji crash cp1252.

## Rules
- Never fp16 (NaN). Check outputs for NaN.
- Re-normalise after MRL truncation. Always use task prefixes.
- Ask the user before large downloads or batches of AI Hub jobs.
- Log every AI Hub job to `results/jobs.csv`.
- Don't `pip install` into the Anaconda base env (a failed install broke `huggingface_hub` once).

## Status
- Steps 1-2 done. Reference in `results/accuracy.csv`. `artifacts/fp32-s128[-split].onnx` matches the source at cos 0.999999.
- Step 3 blocked. On S24 with qnn_dlc, both the full and the split (`--embed-on-host`) fp32 graphs compile but fail on device with `QNN_COMMON_ERROR_MEM_ALLOC`. The runtime log (`job.download_job_logs`) shows HTP fp16 kernels rejecting rank-5 elementwise ops (Q reshape `[0,0,2,2,256]`, `Unsqueeze_5d` mask).
- Next: 1) rewrite attention to rank <= 4 in `surgery.py`; 2) fp32 CPU baseline (`--compute_unit cpu`, tflite or onnx); 3) quantize ladder on the split graph via `scripts/quantize.py` (untested end to end); 4) `scripts/collect.py` is untested (profile keys assumed).
- Reuse finished compile jobs with `hub_run --compile-job <id>`.
