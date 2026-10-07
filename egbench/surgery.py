"""Turn the upstream ONNX export into a static, standard-ONNX text encoder.

The upstream graph targets ONNX Runtime on CPU. AI Hub's QNN compiler needs
static shapes and standard ops, so this module:

1. bypasses the multimodal merge (text-only),
2. decomposes ORT-only ops (RotaryEmbedding, [Skip]SimplifiedLayerNormalization),
3. softens the -3.4e38 attention-mask constant, which wrecks quantization ranges,
4. fixes inputs to (1, seq_len) int32 and constant-folds what that unlocks.
"""

from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

MERGE_OUT = "/model/multimodal_merge/Gather/output_0"
TEXT_EMBEDS = "/model/multimodal_merge/Reshape_text/output_0"
MASK_CONST = "/model/constants/FLOAT/-3.4028234663852886e+38"
MASK_VALUE = -100.0  # exp(-100) is 0 in fp32, yet keeps int ranges sane


class _Builder:
    """Collects replacement nodes and constants with unique names."""

    def __init__(self, graph: onnx.GraphProto):
        self.graph = graph
        self.nodes: list[onnx.NodeProto] = []
        self.consts: dict[tuple, str] = {}

    def const(self, value, dtype=np.int64) -> str:
        arr = np.asarray(value, dtype)
        key = (arr.dtype.str, arr.shape, arr.tobytes())
        if key not in self.consts:
            name = f"/surgery/const_{len(self.consts)}"
            self.graph.initializer.append(numpy_helper.from_array(arr, name))
            self.consts[key] = name
        return self.consts[key]

    def op(self, op_type: str, inputs: list[str], prefix: str, out: str | None = None, n_out: int = 1, **attrs):
        """Append a node; returns its output name (or names when n_out > 1)."""
        name = out or f"{prefix}/{op_type}_{len(self.nodes)}"
        outs = [name] if n_out == 1 else [f"{name}:{i}" for i in range(n_out)]
        self.nodes.append(helper.make_node(op_type, inputs, outs, name=name, **attrs))
        return outs[0] if n_out == 1 else outs


def _rms_norm(b: _Builder, x: str, weight: str, eps: float, prefix: str, out: str) -> None:
    sq = b.op("Mul", [x, x], prefix)
    mean = b.op("ReduceMean", [sq, b.const([-1])], prefix, keepdims=1)
    std = b.op("Sqrt", [b.op("Add", [mean, b.const(eps, np.float32)], prefix)], prefix)
    b.op("Mul", [b.op("Div", [x, std], prefix), weight], prefix, out=out)


def _rotary(b: _Builder, node: onnx.NodeProto, half: int) -> None:
    """Non-interleaved RoPE on (B, S, N*H) with H = 2*half, as ORT computes it."""
    x, pos, cos, sin = node.input
    prefix, out = node.name, node.output[0]
    assert pos == "/model/constants/INT64/[0]", "only zero position offset is handled"
    x4 = b.op("Reshape", [x, b.const([0, 0, -1, 2 * half])], prefix)
    x1, x2 = b.op("Split", [x4, b.const([half, half])], prefix, n_out=2, axis=-1)
    axes = b.const([1])
    c = b.op("Unsqueeze", [cos, axes], prefix)  # (S, 1, half)
    s = b.op("Unsqueeze", [sin, axes], prefix)
    lo = b.op("Sub", [b.op("Mul", [x1, c], prefix), b.op("Mul", [x2, s], prefix)], prefix)
    hi = b.op("Add", [b.op("Mul", [x2, c], prefix), b.op("Mul", [x1, s], prefix)], prefix)
    rot = b.op("Concat", [lo, hi], prefix, axis=-1)
    b.op("Reshape", [rot, b.const([0, 0, -1])], prefix, out=out)


def _rotary_half_dims(graph: onnx.GraphProto) -> dict[str, int]:
    """Map each cos/sin tensor to its frequency count (half the head size)."""
    sizes = {t.name: t.dims[-1] for t in graph.initializer if "inv_freq" in t.name}
    halves = {}
    for n in graph.node:
        if n.op_type == "Mul" and n.input[1] in sizes:
            halves[n.output[0]] = sizes[n.input[1]]
    for n in graph.node:
        if n.op_type in ("Cos", "Sin"):
            halves[n.output[0]] = halves[n.input[0]]
    return halves


def _attr(node: onnx.NodeProto, name: str, default=None):
    for a in node.attribute:
        if a.name == name:
            return helper.get_attribute_value(a)
    return default


def decompose(model: onnx.ModelProto) -> None:
    """Replace ORT contrib ops with standard ONNX, in place."""
    g = model.graph
    halves = _rotary_half_dims(g)
    b = _Builder(g)
    kept = []
    for n in g.node:
        if n.op_type == "RotaryEmbedding":
            assert not _attr(n, "interleaved"), "interleaved RoPE not handled"
            _rotary(b, n, halves[n.input[2]])
        elif n.op_type == "SimplifiedLayerNormalization":
            _rms_norm(b, n.input[0], n.input[1], _attr(n, "epsilon"), n.name, n.output[0])
        elif n.op_type == "SkipSimplifiedLayerNormalization":
            total = n.output[3] if len(n.output) > 3 and n.output[3] else n.name + "/sum"
            b.op("Add", list(n.input[:2]), n.name, out=total)
            if len(n.input) > 3 and n.input[3]:
                total = b.op("Add", [total, n.input[3]], n.name)
            _rms_norm(b, total, n.input[2], _attr(n, "epsilon"), n.name, n.output[0])
        else:
            kept.append(n)
            continue
        kept.extend(b.nodes)
        b.nodes.clear()
    del g.node[:]
    g.node.extend(kept)
    opsets = [o for o in model.opset_import if o.domain != "com.microsoft"]
    del model.opset_import[:]
    model.opset_import.extend(opsets)


def text_only(model: onnx.ModelProto) -> None:
    """Feed token embeddings straight into the decoder stack; drop media inputs."""
    g = model.graph
    for n in g.node:
        for i, name in enumerate(n.input):
            if name == MERGE_OUT:
                n.input[i] = TEXT_EMBEDS
    _prune(g)
    used = {i for n in g.node for i in n.input}
    keep = [i for i in g.input if i.name in used]
    del g.input[:]
    g.input.extend(keep)


def soften_mask(model: onnx.ModelProto) -> None:
    for t in model.graph.initializer:
        if t.name == MASK_CONST:
            t.CopyFrom(numpy_helper.from_array(np.array(MASK_VALUE, np.float32), t.name))


def fix_shapes(model: onnx.ModelProto, seq_len: int) -> None:
    """Static (1, seq_len) int32 inputs; keep only the pooled embedding output."""
    g = model.graph
    for inp in g.input:
        inp.type.tensor_type.elem_type = TensorProto.INT32
        dims = inp.type.tensor_type.shape.dim
        dims[0].dim_value, dims[1].dim_value = 1, seq_len
    keep = [o for o in g.output if o.name == "sentence_embedding"]
    del g.output[:]
    g.output.extend(keep)
    g.output[0].type.tensor_type.shape.dim[0].dim_value = 1
    _prune(g)


def _prune(graph: onnx.GraphProto) -> None:
    """Drop nodes and initializers that no graph output depends on."""
    producer = {o: n for n in graph.node for o in n.output}
    live, stack = set(), [o.name for o in graph.output]
    while stack:
        n = producer.get(stack.pop())
        if n is not None and id(n) not in live:
            live.add(id(n))
            stack.extend(n.input)
    nodes = [n for n in graph.node if id(n) in live]
    del graph.node[:]
    graph.node.extend(nodes)
    used = {i for n in nodes for i in n.input}
    inits = [t for t in graph.initializer if t.name in used]
    del graph.initializer[:]
    graph.initializer.extend(inits)


def fold(src: Path, dst: Path) -> None:
    """Constant-fold with ORT's basic level (no contrib fusions)."""
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    opts.optimized_model_filepath = str(dst)
    opts.add_session_config_entry("session.optimized_model_external_initializers_file_name", "model.data")
    opts.add_session_config_entry("session.optimized_model_external_initializers_min_size_in_bytes", "1024")
    ort.InferenceSession(str(src), opts, providers=["CPUExecutionProvider"])


def op_histogram(path: Path) -> Counter:
    m = onnx.load(str(path), load_external_data=False)
    return Counter(f"{n.domain + '.' if n.domain else ''}{n.op_type}" for n in m.graph.node)


def build(source: Path, out_dir: Path, seq_len: int) -> Path:
    """Write `out_dir/model.onnx` (+ `model.data`); returns the model path."""
    model = onnx.load(str(source))
    text_only(model)
    decompose(model)
    soften_mask(model)
    fix_shapes(model, seq_len)
    onnx.checker.check_model(model, full_check=False)

    out_dir.mkdir(parents=True, exist_ok=True)
    staged = out_dir / "staged.onnx"
    onnx.save(model, str(staged), save_as_external_data=True, location="staged.data")
    del model
    final = out_dir / "model.onnx"
    fold(staged, final)
    staged.unlink()
    (out_dir / "staged.data").unlink()
    return final
