#!/usr/bin/env python3
"""Generate tiled RedMulEComplexGemm ONNX model.

For M > 12, splits into multiple nodes with M=12 each.
Each tile has its own Ar/Ai inputs and Cr/Ci outputs.
B (Br/Bi) is shared across all tiles.
"""

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

import onnx
from onnx import TensorProto, helper

DTYPES = {
    "f16": TensorProto.FLOAT16,
    "f32": TensorProto.FLOAT,
}

HW_MAX_M = 12

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate tiled RedMulEComplexGemm ONNX model"
    )
    p.add_argument("--m", type=int, default=24)
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--hw-max-m", type=int, default=HW_MAX_M)
    p.add_argument("--dtype", choices=sorted(DTYPES), default="f16")
    p.add_argument("--onnx-opset", type=int, default=18)
    p.add_argument("--isolde-opset", type=int, default=1)
    p.add_argument("--out", type=Path, default=Path("redmule_gemm_24x16x16.onnx"))
    return p.parse_args()

def make_model(m, n, k, hw_max_m, dtype, onnx_opset, isolde_opset):
    if min(m, n, k) <= 0:
        raise ValueError("M, N and K must all be positive")

    elem_type = DTYPES[dtype]
    np_dtype = np.float16 if dtype == "f16" else np.float32

    # Compute tiles
    tiles = []
    row = 0
    while row < m:
        tile_m = min(hw_max_m, m - row)
        tiles.append((row, tile_m))
        row += tile_m

    print(f"  M={m} -> {len(tiles)} tile(s): {[t[1] for t in tiles]}")

    # Br is a runtime input, shared by all tiles
    # Bi is zero initializer, shared by all tiles
    inputs = [
        helper.make_tensor_value_info("Br", elem_type, [n, k]),
    ]
    outputs = []
    nodes = []
    initializers = [
        numpy_helper.from_array(
            np.zeros((n, k), dtype=np_dtype), name="Bi"
        ),
    ]

    for i, (row_start, tile_m) in enumerate(tiles):
        ar_name = f"Ar_{i}"
        ai_name = f"Ai_{i}"
        cr_name = f"Cr_{i}"
        ci_name = f"Ci_{i}"

        # Ar_i is a runtime input
        inputs.append(
            helper.make_tensor_value_info(ar_name, elem_type, [tile_m, n])
        )

        # Ai_i is a zero initializer
        initializers.append(
            numpy_helper.from_array(
                np.zeros((tile_m, n), dtype=np_dtype), name=ai_name
            )
        )

        # Cr_i, Ci_i are outputs
        outputs.append(
            helper.make_tensor_value_info(cr_name, elem_type, [tile_m, k])
        )
        outputs.append(
            helper.make_tensor_value_info(ci_name, elem_type, [tile_m, k])
        )

        # RedMulEComplexGemm node for this tile
        nodes.append(helper.make_node(
            "RedMulEComplexGemm",
            inputs=[ar_name, ai_name, "Br", "Bi"],
            outputs=[cr_name, ci_name],
            domain="com.isolde",
            name=f"RedMulEComplexGemm_{i}",
        ))

    graph = helper.make_graph(
        nodes,
        f"redmule_gemm_{m}x{n}x{k}_{dtype}_tiled",
        inputs,
        outputs,
        initializer=initializers,
    )

    model = helper.make_model(
        graph,
        producer_name="ISOLDE-Project/onnx-mlir",
        opset_imports=[
            helper.make_opsetid("", onnx_opset),
            helper.make_opsetid("com.isolde", isolde_opset),
        ],
    )

    return model

def main():
    a = parse_args()
    model = make_model(
        a.m, a.n, a.k, a.hw_max_m, a.dtype, a.onnx_opset, a.isolde_opset
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(a.out))

    print(f"Generated: {a.out}")
    print(f"  hw_max_m : {a.hw_max_m}")
    print(f"  Br       : [{a.n}, {a.k}] {a.dtype} (shared)")
    for i in range(0, a.m, a.hw_max_m):
        tile_m = min(a.hw_max_m, a.m - i)
        print(f"  tile {i//a.hw_max_m}  : Ar[{tile_m},{a.n}] -> Cr[{tile_m},{a.k}]")

if __name__ == "__main__":
    main()
