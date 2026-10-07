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
    are run one text at a time; dynamic graphs are batched. Graphs built
    with the embedding split take `inputs_embeds`, looked up here from
    `embed_tokens.npy` beside the model directory.
    """

    def __init__(self, model: Path, seq_len: int = SEQ_LEN, batch_size: int = 16):
        model_dir = model if model.is_dir() else model.parent
        if model.is_dir():  # AI Hub layout: name.onnx/{model.onnx, model.data}
            model = model / "model.onnx"
        self.seq_len = seq_len
        self.tok = Tokenizer.from_file(str(TOKENIZER))
        self.tok.enable_truncation(seq_len)
        self.sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
        inputs = {i.name: i for i in self.sess.get_inputs()}
        self.int_type = np.int32 if inputs["attention_mask"].type == "tensor(int32)" else np.int64
        self.static = all(isinstance(d, int) for d in inputs["attention_mask"].shape)
        self.batch_size = 1 if self.static else batch_size
        self.media = [name for name in _MULTIMODAL_INPUTS if name in inputs]
        self.table = None
        if "inputs_embeds" in inputs:
            self.table = np.load(model_dir.parent / "embed_tokens.npy", mmap_mode="r")

    def tokenize(self, texts: list[str], width: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Right-padded ids and mask; width defaults to the longest text."""
        enc = self.tok.encode_batch(texts)
        width = width or max(len(e.ids) for e in enc)
        ids = np.full((len(enc), width), PAD_ID, self.int_type)
        mask = np.zeros((len(enc), width), self.int_type)
        for row, e in enumerate(enc):
            ids[row, : len(e.ids)] = e.ids
            mask[row, : len(e.ids)] = 1
        return ids, mask

    def feed(self, texts: list[str]) -> dict[str, np.ndarray]:
        """Graph inputs for a batch, padded to `seq_len` for static graphs."""
        ids, mask = self.tokenize(texts, self.seq_len if self.static else None)
        feed = {"attention_mask": mask}
        if self.table is None:
            feed["input_ids"] = ids
        else:
            feed["inputs_embeds"] = np.asarray(self.table[ids], np.float32)
        for name in self.media:
            feed[name] = np.zeros((0, 512), np.float32)
        return feed

    def __call__(self, texts: list[str], prefix: str = "") -> np.ndarray:
        texts = [prefix + t for t in texts]
        # Sort by length so dynamic batches carry little padding.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = np.empty((len(texts), HIDDEN), np.float32)
        for start in range(0, len(texts), self.batch_size):
            idx = order[start : start + self.batch_size]
            (out[idx],) = self.sess.run(["sentence_embedding"], self.feed([texts[i] for i in idx]))
        if not np.isfinite(out).all():
            raise FloatingPointError("NaN/Inf in embeddings (fp16 overflow?)")
        return out
