#!/usr/bin/env python3
"""Generate ONNX models for the ISOLDE transformer building blocks.

The graphs use the vocabulary of "Attention Is All You Need" (Vaswani et al.
2017, arXiv:1706.03762) in the custom domain ``com.isolde``:

    EncoderLayer            = MultiHeadAttention -> Add -> PositionwiseFeedForward -> Add
    MultiHeadAttention      (sec. 3.2.2)
    ScaledDotProductAttention (sec. 3.2.1, eq. 1)
    PositionwiseFeedForward (sec. 3.3, eq. 2)

Each com.isolde operator is shipped as a model-local FunctionProto whose body
uses only standard ai.onnx ops, so the model runs as-is in onnxruntime.
onnx-mlir imports MultiHeadAttention and PositionwiseFeedForward as
onnx.MultiHeadAttention / onnx.PositionwiseFeedForward (the bodies are ignored
unless --functions-to-decompose names them) and inlines EncoderLayer.

The attention uses ReLU normalisation as in ibex/isolde/sw/radar_attention:

    A = ReLU(Q K^T * scale) * post_scale

`scale` and `post_scale` are left *unfolded* in the model on purpose: the
ONNX -> AISLE pass has to fold them into Wq / Wv.

Next to the model, <out>.npz holds an input and two references:

    y_redmule  bit-exact FP16 result of the RedMulE schedule emitted by
               AISLEToAISMEM (rounding after each of the 16 reduction steps,
               scales folded into f16 weights, ReLU as a sign-bit mask)
    y_float    float64 evaluation of the same graph (for a loose sanity check)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper

DOMAIN = "com.isolde"
L, D = 12, 16          # one native RedMulE tile: Y[12x16] += X[12x16] W[16x16]


# ---------------------------------------------------------------------------
# FunctionProto bodies (reference semantics)
# ---------------------------------------------------------------------------
def _ref(name, attr_type, ref=None):
    return helper.make_attribute_ref(name, attr_type, ref_attr_name=ref or name)


def _node(op, inputs, outputs, domain="", refs=(), **attrs):
    node = helper.make_node(op, inputs, outputs, domain=domain, **attrs)
    node.attribute.extend(refs)
    return node


def _opsets(onnx_opset, *domains):
    return [helper.make_opsetid("", onnx_opset)] + [
        helper.make_opsetid(d, 1) for d in domains]


def sdpa_function(opset):
    """Attention(Q, K, V) = ReLU(Q K^T * scale) * post_scale V; Q,K,V 4-D."""
    nodes = [
        _node("Transpose", ["K"], ["Kt"], perm=[0, 1, 3, 2]),
        _node("MatMul", ["Q", "Kt"], ["QKt"]),
        _node("Constant", [], ["s32"],
              refs=[_ref("value_float", AttributeProto.FLOAT, "scale")]),
        _node("CastLike", ["s32", "QKt"], ["s"]),
        _node("Mul", ["QKt", "s"], ["S"]),
        _node("Relu", ["S"], ["Sr"]),
        _node("Constant", [], ["p32"],
              refs=[_ref("value_float", AttributeProto.FLOAT, "post_scale")]),
        _node("CastLike", ["p32", "Sr"], ["p"]),
        _node("Mul", ["Sr", "p"], ["A"]),
        _node("MatMul", ["A", "V"], ["O"]),
    ]
    return helper.make_function(
        DOMAIN, "ScaledDotProductAttention", ["Q", "K", "V"], ["O"], nodes,
        _opsets(opset), attributes=["scale"],
        attribute_protos=[helper.make_attribute("normalization", "relu"),
                          helper.make_attribute("post_scale", 1.0)])


def mha_function(opset):
    """Concat(head_i) Wo, head_i = Attention(Xq Wq_i, Xkv Wk_i, Xkv Wv_i)."""
    nodes = [
        _node("MatMul", ["Xq", "Wq"], ["q"]),
        _node("MatMul", ["Xkv", "Wk"], ["k"]),
        _node("MatMul", ["Xkv", "Wv"], ["v"]),
        _node("Constant", [], ["h"],
              refs=[_ref("value_int", AttributeProto.INT, "num_heads")]),
        _node("Constant", [], ["axis0"], value_ints=[0]),
        _node("Unsqueeze", ["h", "axis0"], ["h1"]),
        _node("Constant", [], ["keep2"], value_ints=[0, 0]),
        _node("Constant", [], ["rest"], value_ints=[-1]),
        _node("Concat", ["keep2", "h1", "rest"], ["split"], axis=0),
    ]
    for t in "qkv":
        nodes += [_node("Reshape", [t, "split"], [t + "_s"]),
                  _node("Transpose", [t + "_s"], [t + "_h"], perm=[0, 2, 1, 3])]
    nodes += [
        _node("ScaledDotProductAttention", ["q_h", "k_h", "v_h"], ["o_h"],
              domain=DOMAIN,
              refs=[_ref("scale", AttributeProto.FLOAT),
                    _ref("post_scale", AttributeProto.FLOAT),
                    _ref("normalization", AttributeProto.STRING)]),
        _node("Transpose", ["o_h"], ["o_t"], perm=[0, 2, 1, 3]),
        _node("Constant", [], ["merge"], value_ints=[0, 0, -1]),
        _node("Reshape", ["o_t", "merge"], ["o"]),
        _node("MatMul", ["o", "Wo"], ["Y"]),
    ]
    return helper.make_function(
        DOMAIN, "MultiHeadAttention", ["Xq", "Xkv", "Wq", "Wk", "Wv", "Wo"],
        ["Y"], nodes, _opsets(opset, DOMAIN), attributes=["scale"],
        attribute_protos=[helper.make_attribute("num_heads", 1),
                          helper.make_attribute("normalization", "relu"),
                          helper.make_attribute("post_scale", 1.0)])


def ffn_function(opset):
    """FFN(X) = ReLU(X W1) W2 (no biases)."""
    nodes = [_node("MatMul", ["X", "W1"], ["U"]),
             _node("Relu", ["U"], ["R"]),
             _node("MatMul", ["R", "W2"], ["Y"])]
    return helper.make_function(DOMAIN, "PositionwiseFeedForward",
                                ["X", "W1", "W2"], ["Y"], nodes, _opsets(opset))


def encoder_layer_function(opset):
    """h1 = h + MHA(h, h); y = h1 + FFN(h1)   ("Add & Norm", Norm = identity)."""
    refs = [_ref("num_heads", AttributeProto.INT),
            _ref("scale", AttributeProto.FLOAT),
            _ref("post_scale", AttributeProto.FLOAT),
            _ref("normalization", AttributeProto.STRING)]
    nodes = [
        _node("MultiHeadAttention", ["H", "H", "Wq", "Wk", "Wv", "Wo"],
              ["attn"], domain=DOMAIN, refs=refs),
        _node("Add", ["H", "attn"], ["H1"]),
        _node("PositionwiseFeedForward", ["H1", "W1", "W2"], ["ffn"],
              domain=DOMAIN),
        _node("Add", ["H1", "ffn"], ["Y"]),
    ]
    return helper.make_function(
        DOMAIN, "EncoderLayer", ["H", "Wq", "Wk", "Wv", "Wo", "W1", "W2"],
        ["Y"], nodes, _opsets(opset, DOMAIN), attributes=["scale"],
        attribute_protos=[helper.make_attribute("num_heads", 1),
                          helper.make_attribute("normalization", "relu"),
                          helper.make_attribute("post_scale", 1.0)])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def make_weights(seed, d_ff):
    rng = np.random.default_rng(seed)

    def w(rows, cols):
        return (rng.standard_normal((rows, cols)) / np.sqrt(rows)).astype(
            np.float16)

    return dict(wq=w(D, D), wk=w(D, D), wv=w(D, D), wo=w(D, D),
                w1=w(D, d_ff), w2=w(d_ff, D),
                scale=1.0 / np.sqrt(D),
                post_scale=float(np.exp(rng.uniform(1.5, 3.0)) / L))


def build_model(block, p, onnx_opset):
    attrs = dict(num_heads=1, normalization="relu",
                 scale=float(p["scale"]), post_scale=float(p["post_scale"]))
    names = dict(wq="Wq", wk="Wk", wv="Wv", wo="Wo", w1="W1", w2="W2")
    inits = {k: numpy_helper.from_array(p[k], names[k]) for k in names}

    if block == "layer":
        nodes = [helper.make_node(
            "EncoderLayer", ["h", "Wq", "Wk", "Wv", "Wo", "W1", "W2"], ["y"],
            domain=DOMAIN, name="encoder.layer0", **attrs)]
        used = list(names.values())
    elif block == "mha":
        # self-attention plus the residual "Add" of the paper's sub-layer
        nodes = [helper.make_node(
                     "MultiHeadAttention", ["h", "h", "Wq", "Wk", "Wv", "Wo"],
                     ["attn"], domain=DOMAIN, name="self_attention", **attrs),
                 helper.make_node("Add", ["h", "attn"], ["y"],
                                  name="add_norm")]
        used = ["Wq", "Wk", "Wv", "Wo"]
    elif block == "ffn":
        nodes = [helper.make_node(
                     "PositionwiseFeedForward", ["h", "W1", "W2"], ["ffn"],
                     domain=DOMAIN, name="feed_forward"),
                 helper.make_node("Add", ["h", "ffn"], ["y"],
                                  name="add_norm")]
        used = ["W1", "W2"]
    else:
        raise ValueError(block)

    graph = helper.make_graph(
        nodes, f"transformer_{block}",
        [helper.make_tensor_value_info("h", TensorProto.FLOAT16, [1, L, D])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, L, D])],
        initializer=[inits[k] for k in names if names[k] in used])
    model = helper.make_model(
        graph, producer_name="ISOLDE test-isolde/transformer", ir_version=9,
        opset_imports=_opsets(onnx_opset, DOMAIN),
        functions=[encoder_layer_function(onnx_opset),
                   mha_function(onnx_opset), sdpa_function(onnx_opset),
                   ffn_function(onnx_opset)])
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    return model


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------
def gemm16(x, w, y=None):
    """RedMulE: Z = X W (+ Y), FP16 rounding after each reduction step."""
    x = np.asarray(x, np.float16)
    w = np.asarray(w, np.float16)
    out = (np.zeros((x.shape[0], w.shape[1]), np.float16) if y is None
           else np.array(y, np.float16))
    for n in range(x.shape[1]):
        out = (x[:, n, None].astype(np.float32) * w[None, n, :].astype(np.float32)
               + out.astype(np.float32)).astype(np.float16)
    return out


def relu16(a):
    bits = np.ascontiguousarray(a, np.float16).view(np.uint16)
    return np.where(bits & 0x8000, 0, bits).astype(np.uint16).view(np.float16)


def pad(a, rows=16, cols=16):
    out = np.zeros((rows, cols), np.float16)
    out[:a.shape[0], :a.shape[1]] = a
    return out


def fold(w, factor):
    """What ONNXToAISLE does: f16(f64(w) * factor)."""
    return (w.astype(np.float64) * factor).astype(np.float16)


def redmule_reference(block, p, h):
    h = h.reshape(L, D).astype(np.float16)
    wq, wv = fold(p["wq"], p["scale"]), fold(p["wv"], p["post_scale"])

    def mha(x):
        q, k, v = gemm16(x, wq), gemm16(x, p["wk"]), gemm16(x, wv)
        s = gemm16(q, pad(k.T))
        o = gemm16(relu16(s), pad(v))
        return gemm16(o, p["wo"], x)             # residual preloaded in Y

    def ffn(x):
        d_ff = p["w1"].shape[1]
        u = [gemm16(x, p["w1"][:, j:j + 16]) for j in range(0, d_ff, 16)]
        y = x
        for i, j in enumerate(range(0, d_ff, 16)):
            y = gemm16(relu16(u[i]), p["w2"][j:j + 16], y)
        return y

    y = {"mha": mha, "ffn": ffn, "layer": lambda x: ffn(mha(x))}[block](h)
    return y.reshape(1, L, D)


def float_reference(block, p, h):
    f = {k: v.astype(np.float64) for k, v in p.items() if isinstance(v, np.ndarray)}
    x = h.reshape(L, D).astype(np.float64)

    def mha(x):
        s = (x @ f["wq"]) @ (x @ f["wk"]).T * p["scale"]
        return x + (np.maximum(s, 0) * p["post_scale"]) @ (x @ f["wv"]) @ f["wo"]

    def ffn(x):
        return x + np.maximum(x @ f["w1"], 0) @ f["w2"]

    y = {"mha": mha, "ffn": ffn, "layer": lambda x: ffn(mha(x))}[block](x)
    return y.reshape(1, L, D)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--block", choices=["layer", "mha", "ffn"], default="layer")
    ap.add_argument("--d-ff", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--onnx-opset", type=int, default=18)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    p = make_weights(a.seed, a.d_ff)
    model = build_model(a.block, p, a.onnx_opset)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, a.out)

    h = (np.random.default_rng(a.seed + 1).standard_normal((1, L, D))
         .astype(np.float16))
    np.savez(a.out.with_suffix(".npz"), h=h,
             y_redmule=redmule_reference(a.block, p, h),
             y_float=float_reference(a.block, p, h))
    print(f"wrote {a.out} (+ .npz)  block={a.block} d_ff={a.d_ff} "
          f"scale={p['scale']:.4f} post_scale={p['post_scale']:.4f}")


if __name__ == "__main__":
    main()
