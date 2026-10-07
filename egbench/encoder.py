"""Run an EmbeddingGemma ONNX graph with ONNX Runtime on CPU."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from .config import HIDDEN, PAD_ID, SEQ_LEN, TOKENIZER

# The upstream export merges multimodal features into the token stream.
# Text-only runs feed them as empty tensors.
_MULTIMODAL_INPUTS = ("image_features", "video_features", "audio_features")


def truncate(emb: np.ndarray, dim: int) -> np.ndarray:
    """MRL truncation; re-normalising is mandatory after slicing."""
    out = emb[:, :dim]
    return out / np.linalg.norm(out, axis=1, keepdims=True)


class Encoder:
    """Tokenise, pad to `seq_len` and return L2-normalised 768-d embeddings.

    Static-shape graphs take exactly one row of `seq_len` tokens, so they
    are run one text at a time; dynamic graphs are batched.
    """

    def __init__(self, model: Path, seq_len: int = SEQ_LEN, batch_size: int = 16):
        self.seq_len = seq_len
        self.tok = Tokenizer.from_file(str(TOKENIZER))
        self.tok.enable_truncation(seq_len)
        self.sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.sess.get_inputs()}
        self.static = all(isinstance(d, int) for d in self.sess.get_inputs()[0].shape)
        self.batch_size = 1 if self.static else batch_size

    def _feed(self, ids: np.ndarray, mask: np.ndarray) -> dict:
        feed = {"input_ids": ids, "attention_mask": mask}
        for name in _MULTIMODAL_INPUTS:
            if name in self.input_names:
                feed[name] = np.zeros((0, 512), np.float32)
        return feed

    def _batch(self, texts: list[str]) -> np.ndarray:
        enc = self.tok.encode_batch(texts)
        width = self.seq_len if self.static else max(len(e.ids) for e in enc)
        ids = np.full((len(enc), width), PAD_ID, np.int64)
        mask = np.zeros((len(enc), width), np.int64)
        for row, e in enumerate(enc):
            ids[row, : len(e.ids)] = e.ids
            mask[row, : len(e.ids)] = 1
        (emb,) = self.sess.run(["sentence_embedding"], self._feed(ids, mask))
        return emb

    def __call__(self, texts: list[str], prefix: str = "") -> np.ndarray:
        texts = [prefix + t for t in texts]
        # Sort by length so dynamic batches carry little padding.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = np.empty((len(texts), HIDDEN), np.float32)
        for start in range(0, len(texts), self.batch_size):
            idx = order[start : start + self.batch_size]
            out[idx] = self._batch([texts[i] for i in idx])
        if not np.isfinite(out).all():
            raise FloatingPointError("NaN/Inf in embeddings (fp16 overflow?)")
        return out
