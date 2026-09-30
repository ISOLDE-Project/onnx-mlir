#!/usr/bin/env python3
"""The radar_attention Encoder (tformer.py) as an ONNX graph of ISOLDE ops.

The top-level graph uses the operators that onnx-mlir (ISOLDE fork, patch
"Add ISOLDE transformer blocks") imports as ONNX-dialect ops:

    com.isolde::MultiHeadAttention       -> onnx.MultiHeadAttention
    com.isolde::PositionwiseFeedForward  -> onnx.PositionwiseFeedForward

with exactly the inputs and attributes of those ops (AdditionalONNXOps.td):

    MultiHeadAttention(Xq, Xkv, Wq, Wk, Wv, Wo [, C])
        num_heads: int, scale: float, post_scale: float, normalization: str
    PositionwiseFeedForward(X, W1, W2 [, C])
        activation: str

The graph, in the vocabulary of "Attention Is All You Need" (Vaswani et al.
2017, arXiv:1706.03762):

    x[1,12,32]
      MatMul                              input_embedding       (sec. 3.4)
      Add                                 positional_encoding   (sec. 3.5)
      for each layer i:                                         (Fig. 1, left)
        com.isolde.MultiHeadAttention     encoder.layer{i}.self_attention
        Add                               encoder.layer{i}.add_norm1
        com.isolde.PositionwiseFeedForward encoder.layer{i}.feed_forward
        Add                               encoder.layer{i}.add_norm2
      ReduceMean                          sequence_mean_pool    (not in paper)
      MatMul                              classifier_head       (not in paper)
    logits[1,4]

"Add & Norm" is a plain Add: the model has no LayerNorm.  The residual stays
an explicit Add in the file; onnx-mlir's ONNX->AISLE pass folds it into the
block's accumulator C (a free Y preload on RedMulE).

By default the model also carries a FunctionProto per com.isolde op (bodies
in standard ai.onnx ops), so onnxruntime, onnx.checker and shape inference
work on it unchanged; onnx-mlir ignores the bodies unless
--functions-to-decompose names the op.  --no-functions writes the bare custom
ops only (onnx-mlir still imports it; onnxruntime cannot run it).

Differences from the paper, all taken from tformer.Encoder:
  * ReLU attention: A = ReLU(Q K^T * scale) * post_scale,
    post_scale = exp(gate_i) / L   (Wortsman et al. 2023);
  * one head, d_model = d_k = d_v = 16; no LayerNorm, biases or dropout;
  * learned positional embedding; mean pool + linear classifier.

Weight sources (the number of layers comes from the source):
  --from-header H   trained weights, already *folded* as the firmware uses
        them (1/sqrt(d) in Wq, s/L in Wv) -> scale = 1, post_scale = 1 on the
        attention nodes.  Default: inc/tformer_weights.h, the 2-layer
        encoder written by `make model` (not in git); if it is missing,
        inc_l1/tformer_weights.h, the committed 1-layer chain-mode export.
  --from-torch   a seeded (untrained) tformer.Encoder with --layers layers
        (default 2), *unfolded*: scale and post_scale stay visible as
        attributes.  Needs torch.

--golden V (with a header) also writes <out>.npz for
test-isolde/transformer/check_aismem.py: the firmware's input window
tf_features from V (tformer_vectors.h, default inc/tformer_vectors.h) as
`h` [1,12,32], its tf_logits_golden (RedMulE FP16 arithmetic, bit-exact
target) as `y_redmule` [1,C], and a float64 forward pass as `y_float`.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import onnx
import onnx.inliner
from onnx import AttributeProto, TensorProto, helper, numpy_helper

DOMAIN = 'com.isolde'
ISOLDE_OPSET = 1
ONNX_OPSET = 18           # same as radar_beamforming / complex_gemm scripts
IR_VERSION = 9            # FunctionProto.attribute_proto needs IR >= 9

_DTYPES = {'f16': (np.float16, TensorProto.FLOAT16),
           'f32': (np.float32, TensorProto.FLOAT)}


def _ref(name, attr_type, ref=None):
    """Attribute of a body node that forwards the caller's attribute `ref`."""
    return helper.make_attribute_ref(name, attr_type, ref_attr_name=ref or name)


def _node(op, inputs, outputs, domain='', refs=(), **attrs):
    node = helper.make_node(op, inputs, outputs, domain=domain, **attrs)
    node.attribute.extend(refs)
    return node


def _opsets(*domains):
    return [helper.make_opsetid('', ONNX_OPSET)] + [
        helper.make_opsetid(d, ISOLDE_OPSET) for d in domains]


# ---------------------------------------------------------------------------
# Reference semantics of the com.isolde ops (model-local FunctionProtos)
# ---------------------------------------------------------------------------
def scaled_dot_product_attention_function(normalization):
    """Attention(Q, K, V) = norm(Q K^T * scale) V -- eq. (1); Q,K,V 4-D."""
    nodes = [
        _node('Transpose', ['K'], ['Kt'], perm=[0, 1, 3, 2]),
        _node('MatMul', ['Q', 'Kt'], ['QKt']),
        _node('Constant', [], ['scale_f32'],
              refs=[_ref('value_float', AttributeProto.FLOAT, 'scale')]),
        _node('CastLike', ['scale_f32', 'QKt'], ['scale_t']),
        _node('Mul', ['QKt', 'scale_t'], ['S']),
    ]
    if normalization == 'softmax':
        nodes.append(_node('Softmax', ['S'], ['A'], axis=-1))
    elif normalization == 'relu':
        nodes += [
            _node('Relu', ['S'], ['S_relu']),
            _node('Constant', [], ['post_f32'],
                  refs=[_ref('value_float', AttributeProto.FLOAT,
                             'post_scale')]),
            _node('CastLike', ['post_f32', 'S_relu'], ['post_t']),
            _node('Mul', ['S_relu', 'post_t'], ['A']),
        ]
    else:
        raise ValueError(normalization)
    nodes.append(_node('MatMul', ['A', 'V'], ['O']))
    return helper.make_function(
        DOMAIN, 'ScaledDotProductAttention', ['Q', 'K', 'V'], ['O'], nodes,
        _opsets(), attributes=['scale'],
        attribute_protos=[
            helper.make_attribute('normalization', normalization),
            helper.make_attribute('post_scale', 1.0)])


def multi_head_attention_function(normalization):
    """Concat(head_1..head_h) Wo, head_i = Attention(Xq Wq_i, Xkv Wk_i,
    Xkv Wv_i) -- sec. 3.2.2.  Signature of onnx.MultiHeadAttention without
    the optional accumulator C."""
    nodes = [
        _node('MatMul', ['Xq', 'Wq'], ['q']),
        _node('MatMul', ['Xkv', 'Wk'], ['k']),
        _node('MatMul', ['Xkv', 'Wv'], ['v']),
        # split heads: [B, L, D] -> [B, L, h, D/h] -> [B, h, L, D/h]
        _node('Constant', [], ['h'],
              refs=[_ref('value_int', AttributeProto.INT, 'num_heads')]),
        _node('Constant', [], ['zero_axis'], value_ints=[0]),
        _node('Unsqueeze', ['h', 'zero_axis'], ['h1']),
        _node('Constant', [], ['keep2'], value_ints=[0, 0]),
        _node('Constant', [], ['rest'], value_ints=[-1]),
        _node('Concat', ['keep2', 'h1', 'rest'], ['split_shape'], axis=0),
    ]
    for t in 'qkv':
        nodes += [_node('Reshape', [t, 'split_shape'], [t + '_s']),
                  _node('Transpose', [t + '_s'], [t + '_h'], perm=[0, 2, 1, 3])]
    nodes += [
        _node('ScaledDotProductAttention', ['q_h', 'k_h', 'v_h'], ['o_h'],
              domain=DOMAIN,
              refs=[_ref('scale', AttributeProto.FLOAT),
                    _ref('post_scale', AttributeProto.FLOAT),
                    _ref('normalization', AttributeProto.STRING)]),
        # merge heads: [B, h, L, d_v] -> [B, L, h, d_v] -> [B, L, h*d_v]
        _node('Transpose', ['o_h'], ['o_t'], perm=[0, 2, 1, 3]),
        _node('Constant', [], ['merge_shape'], value_ints=[0, 0, -1]),
        _node('Reshape', ['o_t', 'merge_shape'], ['o']),
        _node('MatMul', ['o', 'Wo'], ['Y']),
    ]
    return helper.make_function(
        DOMAIN, 'MultiHeadAttention',
        ['Xq', 'Xkv', 'Wq', 'Wk', 'Wv', 'Wo'], ['Y'], nodes,
        _opsets(DOMAIN), attributes=['scale'],
        attribute_protos=[helper.make_attribute('num_heads', 1),
                          helper.make_attribute('normalization', normalization),
                          helper.make_attribute('post_scale', 1.0)])


def positionwise_feed_forward_function():
    """FFN(X) = ReLU(X W1) W2 -- eq. (2), no biases.  Signature of
    onnx.PositionwiseFeedForward without the optional accumulator C."""
    nodes = [_node('MatMul', ['X', 'W1'], ['U']),
             _node('Relu', ['U'], ['R']),
             _node('MatMul', ['R', 'W2'], ['Y'])]
    return helper.make_function(
        DOMAIN, 'PositionwiseFeedForward', ['X', 'W1', 'W2'], ['Y'], nodes,
        _opsets(), attribute_protos=[helper.make_attribute('activation',
                                                           'relu')])


# ---------------------------------------------------------------------------
# Encoder graph
# ---------------------------------------------------------------------------
def build_model(params, dtype='f16', batch=1, normalization='relu',
                num_heads=1, with_functions=True, name='Encoder'):
    """params: proj[F,D], pos[L,D], head[D,C],
    layers[i]{wq,wk,wv,wo[D,D], w1[D,Dff], w2[Dff,D], scale, post_scale}.
    Returns an onnx.ModelProto whose top-level graph uses the com.isolde
    MultiHeadAttention / PositionwiseFeedForward ops directly."""
    np_t, onnx_t = _DTYPES[dtype]
    frames, d_model = params['pos'].shape
    n_features, n_classes = params['proj'].shape[0], params['head'].shape[1]
    assert d_model % num_heads == 0

    inits, value_info = [], []

    def init(tensor_name, value):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(value, dtype=np_t), tensor_name))
        return tensor_name

    def act(tensor_name, shape):
        # Static shapes on every edge: shape inference cannot see through a
        # custom op when the functions are omitted, and onnx-mlir benefits.
        value_info.append(
            helper.make_tensor_value_info(tensor_name, onnx_t, shape))
        return tensor_name

    seq = [batch, frames, d_model]
    nodes = [
        _node('MatMul', ['x', init('input_embedding.W', params['proj'])],
              [act('embedded', seq)], name='input_embedding'),
        _node('Add', ['embedded', init('positional_encoding.P', params['pos'])],
              [act('h0', seq)], name='positional_encoding'),
    ]
    h = 'h0'
    for i, layer in enumerate(params['layers']):
        p = f'encoder.layer{i}.'
        wq, wk, wv, wo = (init(p + 'self_attention.' + n, layer[k])
                          for n, k in (('Wq', 'wq'), ('Wk', 'wk'),
                                       ('Wv', 'wv'), ('Wo', 'wo')))
        w1 = init(p + 'feed_forward.W1', layer['w1'])
        w2 = init(p + 'feed_forward.W2', layer['w2'])
        nodes += [
            _node('MultiHeadAttention', [h, h, wq, wk, wv, wo],
                  [act(p + 'attn', seq)], domain=DOMAIN,
                  name=p + 'self_attention', num_heads=num_heads,
                  scale=float(layer['scale']),
                  post_scale=float(layer['post_scale']),
                  normalization=normalization),
            _node('Add', [h, p + 'attn'], [act(p + 'h1', seq)],
                  name=p + 'add_norm1'),
            _node('PositionwiseFeedForward', [p + 'h1', w1, w2],
                  [act(p + 'ffn', seq)], domain=DOMAIN,
                  name=p + 'feed_forward', activation='relu'),
            _node('Add', [p + 'h1', p + 'ffn'], [act(p + 'h2', seq)],
                  name=p + 'add_norm2'),
        ]
        h = p + 'h2'

    inits.append(numpy_helper.from_array(np.array([1], dtype=np.int64),
                                         'sequence_mean_pool.axes'))
    nodes += [
        _node('ReduceMean', [h, 'sequence_mean_pool.axes'],
              [act('pooled', [batch, d_model])], keepdims=0,
              name='sequence_mean_pool'),
        _node('MatMul', ['pooled', init('classifier_head.W', params['head'])],
              ['logits'], name='classifier_head'),
    ]
    graph = helper.make_graph(
        nodes, name,
        [helper.make_tensor_value_info('x', onnx_t,
                                       [batch, frames, n_features])],
        [helper.make_tensor_value_info('logits', onnx_t, [batch, n_classes])],
        initializer=inits, value_info=value_info)
    functions = []
    if with_functions:
        functions = [multi_head_attention_function(normalization),
                     scaled_dot_product_attention_function(normalization),
                     positionwise_feed_forward_function()]
    return helper.make_model(
        graph, producer_name='ISOLDE radar_attention tformer_onnx.py',
        ir_version=IR_VERSION, opset_imports=_opsets(DOMAIN),
        functions=functions)


def check(model):
    """Checker + strict shape inference when the bodies are present; with bare
    custom ops only the structural checker applies."""
    if model.functions:
        onnx.checker.check_model(model, full_check=True)
        onnx.shape_inference.infer_shapes(model, check_type=True,
                                          strict_mode=True, data_prop=True)
    else:
        onnx.checker.check_model(model)


def inline(model):
    """Standard-ops-only twin (what onnx-mlir produces with
    --functions-to-decompose for both ops)."""
    flat = onnx.inliner.inline_local_functions(model)
    return onnx.shape_inference.infer_shapes(flat, strict_mode=True)


# ---------------------------------------------------------------------------
# Weight sources
# ---------------------------------------------------------------------------
def params_from_torch(model, frames=12, fold=False):
    """Unfolded by default: scale=1/sqrt(d_model), post_scale=exp(gate)/L."""
    def np32(t):
        return t.detach().cpu().numpy().astype(np.float32)

    d_model = model.proj.weight.shape[0]
    layers = []
    for i, (attn, mlp) in enumerate(zip(model.attn, model.mlp)):
        scale = 1.0 / np.sqrt(d_model)
        post = float(model.gate[i].detach().exp()) / frames
        layer = dict(wq=np32(attn['q'].weight.T), wk=np32(attn['k'].weight.T),
                     wv=np32(attn['v'].weight.T), wo=np32(attn['o'].weight.T),
                     w1=np32(mlp[0].weight.T), w2=np32(mlp[2].weight.T),
                     scale=scale, post_scale=post)
        if fold:
            layer['wq'] = layer['wq'] * scale
            layer['wv'] = layer['wv'] * post
            layer['scale'] = layer['post_scale'] = 1.0
        layers.append(layer)
    return dict(proj=np32(model.proj.weight.T), pos=np32(model.pos),
                head=np32(model.head.weight.T), layers=layers)


def _header_arrays(path):
    text = Path(path).read_text()
    arrays = {}
    for m in re.finditer(r'uint16_t (\w+)\[(\d+)\][^{]*= \{([^}]*)\}', text):
        raw = np.array([int(v, 16) for v in
                        re.findall(r'0x([0-9a-f]{4})', m.group(3))],
                       dtype=np.uint16)
        arrays[m.group(1)] = raw.view(np.float16)
    defines = {k: int(v) for k, v in
               re.findall(r'#define (TF_\w+) (\d+)u', text)}
    return arrays, defines


def params_from_header(path):
    """Trained, folded weights from tformer_weights*.h (tile layouts undone)."""
    a, d = _header_arrays(path)
    D, F, L = d['TF_DMODEL'], d['TF_FEATURES'], d['TF_FRAMES']
    dff, C, tiles = d['TF_DFF'], d['TF_CLASSES'], d['TF_DFF'] // 16
    layers = []
    for i in range(d['TF_LAYERS']):
        w1 = a[f'tf_l{i}_w1'].reshape(tiles, D, 16).transpose(1, 0, 2)
        layers.append(dict(
            wq=a[f'tf_l{i}_wq'].reshape(D, D), wk=a[f'tf_l{i}_wk'].reshape(D, D),
            wv=a[f'tf_l{i}_wv'].reshape(D, D), wo=a[f'tf_l{i}_wo'].reshape(D, D),
            w1=w1.reshape(D, dff), w2=a[f'tf_l{i}_w2'].reshape(dff, D),
            scale=1.0, post_scale=1.0))        # folded into wq / wv
    return dict(proj=a['tf_proj'].reshape(F, D), pos=a['tf_pos'].reshape(L, D),
                head=a['tf_head'].reshape(D, 16)[:, :C], layers=layers)


def features_from_header(path):
    """tformer_vectors.h tf_features [2][12][16] -> window [12, 32]."""
    a, _ = _header_arrays(path)
    tiles = a['tf_features'].reshape(2, 12, 16)
    return np.concatenate([tiles[0], tiles[1]], axis=1)


def logits_golden_from_header(path, n_classes):
    """tf_logits_golden of tformer_weights.h: the firmware's FP16 logits."""
    a, _ = _header_arrays(path)
    return a['tf_logits_golden'][:n_classes].reshape(1, n_classes)


def float_reference(params, window):
    """float64 forward pass of the encoder (A = ReLU(Q K^T s) p V)."""
    f = lambda v: np.asarray(v, dtype=np.float64)
    h = f(window) @ f(params['proj']) + f(params['pos'])
    for layer in params['layers']:
        q, k, v = (h @ f(layer[w]) for w in ('wq', 'wk', 'wv'))
        a = np.maximum(q @ k.T * layer['scale'], 0) * layer['post_scale']
        h = h + a @ v @ f(layer['wo'])
        h = h + np.maximum(h @ f(layer['w1']), 0) @ f(layer['w2'])
    return (h.mean(axis=0) @ f(params['head'])).reshape(1, -1)


# ---------------------------------------------------------------------------
def default_header(here):
    """`make model` output (2 layers) if present, else the committed 1-layer
    chain-mode export."""
    for candidate in (here / 'inc/tformer_weights.h',
                      here / 'inc_l1/tformer_weights.h'):
        if candidate.exists():
            return candidate
    raise SystemExit('no tformer_weights.h found: run `make model` or pass '
                     '--from-header / --from-torch')


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument('--from-header', type=Path, default=None,
                     help='trained weights header (default: inc/, else inc_l1/)')
    src.add_argument('--from-torch', action='store_true',
                     help='seeded, untrained tformer.Encoder (unfolded)')
    ap.add_argument('--layers', type=int, default=None,
                    help='--from-torch: number of layers (default 2); with a '
                         'header: expected TF_LAYERS, checked')
    ap.add_argument('--d-ff', type=int, default=48)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--dtype', choices=sorted(_DTYPES), default='f16')
    ap.add_argument('--out', type=Path, default=Path('models/tformer_encoder.onnx'))
    ap.add_argument('--no-functions', action='store_true',
                    help='omit the FunctionProto bodies (onnx-mlir only)')
    ap.add_argument('--golden', type=Path, nargs='?', default=None,
                    const=Path(__file__).resolve().parent / 'inc/tformer_vectors.h',
                    metavar='VECTORS_H',
                    help='with a header: also write <out>.npz (input window, '
                         'firmware logits golden) for check_aismem.py')
    ap.add_argument('--inlined', action='store_true',
                    help='also write the standard-ops-only twin (*_inlined.onnx)')
    args = ap.parse_args()

    if args.from_torch:
        import torch
        import tformer
        torch.manual_seed(args.seed)
        params = params_from_torch(
            tformer.Encoder(layers=args.layers or 2, d_ff=args.d_ff))
        source = f'untrained tformer.Encoder (seed {args.seed})'
    else:
        header = args.from_header or default_header(here)
        params = params_from_header(header)
        source = str(header)
        n = len(params['layers'])
        if args.layers is not None and args.layers != n:
            ap.error(f'{header} holds {n} layer(s), not {args.layers}; '
                     f'`make model LAYERS={args.layers}` writes a matching '
                     f'inc/tformer_weights.h')

    model = build_model(params, dtype=args.dtype,
                        with_functions=not args.no_functions)
    check(model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.out)
    ops = [f'{n.domain + "." if n.domain else ""}{n.op_type}'
           for n in model.graph.node]
    print(f'wrote {args.out}  ({len(params["layers"])} layer(s), {args.dtype}, '
          f'from {source})')
    print('  graph: ' + ', '.join(ops))
    if args.golden is not None:
        if args.from_torch:
            ap.error('--golden needs --from-header (the firmware goldens)')
        window = features_from_header(args.golden)
        n_classes = params['head'].shape[1]
        npz = args.out.with_suffix('.npz')
        np.savez(npz, h=window.reshape(1, *window.shape).astype(np.float16),
                 y_redmule=logits_golden_from_header(header, n_classes),
                 y_float=float_reference(params, window))
        print(f'wrote {npz}  (input {args.golden.name}:tf_features, '
              f'y_redmule = tf_logits_golden)')
    if args.inlined:
        if args.no_functions:
            ap.error('--inlined needs the function bodies')
        flat = inline(model)
        onnx.checker.check_model(flat, full_check=True)
        out = args.out.with_name(args.out.stem + '_inlined.onnx')
        onnx.save(flat, out)
        print(f'wrote {out}  ({len(flat.graph.node)} standard nodes)')


if __name__ == '__main__':
    main()