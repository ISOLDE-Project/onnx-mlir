#!/usr/bin/env python3
"""Standalone ONNX graph for the two-K-tile projection stage of the ISOLDE
radar_attention encoder (tformer_runtime3), i.e. the C fragment:

    launch_bias(0, window, tf_proj, tf_pos);   // Y  = X0 . We0 + pos
    omrm_wait(1u);
    launch_accumulate(0, window + X_ELEMENTS,   // Y += X1 . We1
                      tf_proj + W_ELEMENTS);
    omrm_wait(1u);
    collect(0, h);                             // h  = Y

Mathematically this is the input embedding + learned positional encoding:

    h[12,16] = X[12,32] . We[32,16] + pos[12,16]

The RedMulE runtime splits the K=32 contraction into two 16-wide tiles
(launch_bias then launch_accumulate); a single MatMul over K=32 is the exact
same value, so the graph uses one MatMul + one Add.  The two-tile form is kept
available behind --split for a literal 1:1 mirror of the two launches.

Standard ai.onnx ops only (no com.isolde domain); runnable in onnxruntime.
IEEE binary16 to match the firmware storage type.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ONNX_OPSET = 18
IR_VERSION = 9

FRAMES = 12          # TF_FRAMES  (rows of the window / one tile tall)
D_MODEL = 16         # TF_DMODEL  (COLS, one tile wide)
FEATURES = 32        # TF_FEATURES (2 * COLS, two K-tiles)


def build_model(proj, pos, split=False):
    """proj: We[FEATURES, D_MODEL]; pos: [FRAMES, D_MODEL] (f16 arrays).
    Returns a ModelProto computing h = X.We + pos, X = [FRAMES, FEATURES]."""
    inits, nodes = [], []

    def init(name, value):
        t = numpy_helper.from_array(np.ascontiguousarray(value, np.float16), name)
        inits.append(t)
        return name

    x = helper.make_tensor_value_info('window', TensorProto.FLOAT16,
                                      [FRAMES, FEATURES])
    h = helper.make_tensor_value_info('h', TensorProto.FLOAT16,
                                      [FRAMES, D_MODEL])

    if not split:
        # One MatMul over the full K=32 contraction, then + pos.
        nodes.append(helper.make_node('MatMul', ['window', init('tf_proj', proj)],
                                      ['embedded'], name='input_embedding'))
        nodes.append(helper.make_node('Add', ['embedded', init('tf_pos', pos)],
                                      ['h'], name='positional_encoding'))
    else:
        # Literal two-K-tile mirror of launch_bias + launch_accumulate.
        # X0 = window[:, 0:16], X1 = window[:, 16:32]; We0/We1 the K-tiles.
        starts0 = init('starts0', np.array([0, 0], np.int64))
        # int64 initialisers (Slice indices) are exact, not f16.
        inits[-1] = numpy_helper.from_array(np.array([0, 0], np.int64), 'starts0')
        ends0 = numpy_helper.from_array(np.array([FRAMES, D_MODEL], np.int64), 'ends0')
        starts1 = numpy_helper.from_array(np.array([0, D_MODEL], np.int64), 'starts1')
        ends1 = numpy_helper.from_array(np.array([FRAMES, FEATURES], np.int64), 'ends1')
        axes = numpy_helper.from_array(np.array([0, 1], np.int64), 'axes')
        inits += [ends0, starts1, ends1, axes]

        we0 = init('tf_proj_tile0', proj[0:D_MODEL, :])
        we1 = init('tf_proj_tile1', proj[D_MODEL:FEATURES, :])

        nodes += [
            helper.make_node('Slice', ['window', 'starts0', 'ends0', 'axes'],
                             ['X0'], name='window_tile0'),
            helper.make_node('Slice', ['window', 'starts1', 'ends1', 'axes'],
                             ['X1'], name='window_tile1'),
            # launch_bias: Y = X0.We0 + pos
            helper.make_node('MatMul', ['X0', we0], ['Y0'], name='launch_bias_matmul'),
            helper.make_node('Add', ['Y0', init('tf_pos', pos)], ['Ybias'],
                             name='launch_bias_add'),
            # launch_accumulate: Y += X1.We1
            helper.make_node('MatMul', ['X1', we1], ['Y1'],
                             name='launch_accumulate_matmul'),
            helper.make_node('Add', ['Ybias', 'Y1'], ['h'],
                             name='launch_accumulate_add'),
        ]

    graph = helper.make_graph(nodes, 'projection', [x], [h], initializer=inits)
    model = helper.make_model(
        graph, producer_name='ISOLDE proj_onnx.py',
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid('', ONNX_OPSET)])
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, check_type=True, strict_mode=True)
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--out', type=Path, default=Path('models/proj.onnx'))
    ap.add_argument('--split', action='store_true',
                    help='emit the literal two-K-tile form (two MatMul+Add, '
                         'mirroring launch_bias + launch_accumulate)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--verify', action='store_true',
                    help='run onnxruntime and compare against a numpy reference')
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    proj = rng.standard_normal((FEATURES, D_MODEL)).astype(np.float16)
    pos = rng.standard_normal((FRAMES, D_MODEL)).astype(np.float16)

    model = build_model(proj, pos, split=args.split)
    onnx.save(model, args.out)
    ops = ', '.join(n.op_type for n in model.graph.node)
    print(f'wrote {args.out}  (split={args.split})  ops: {ops}')

    if args.verify:
        import onnxruntime as ort
        window = rng.standard_normal((FRAMES, FEATURES)).astype(np.float16)
        ref = (window.astype(np.float32) @ proj.astype(np.float32)
               + pos.astype(np.float32))
        sess = ort.InferenceSession(args.out.as_posix(),
                                    providers=['CPUExecutionProvider'])
        got = sess.run(['h'], {'window': window})[0].astype(np.float32)
        max_abs = float(np.max(np.abs(got - ref)))
        print(f'onnxruntime vs numpy f32 ref: max_abs_diff = {max_abs:.4g}')
        assert max_abs < 1e-1, 'projection mismatch'
        print('OK')


if __name__ == '__main__':
    main()
