#!/usr/bin/env python3
"""Generate an ONNX model containing one ISOLDE RedMulEComplexGemm node.

The custom operator has split-complex operands:

    A = Ar + j*Ai
    B = Br + j*Bi

and produces:

    Cr = Ar @ Br - Ai @ Bi
    Ci = Ar @ Bi + Ai @ Br

The serialized ONNX model uses the custom domain ``com.isolde``.  Both F16 and
F32 tensor interfaces are supported by the ISOLDE ONNX dialect op.  F32 is the
default for the first bring-up; the RedMulE lowering may subsequently select
FP16 execution explicitly.
"""

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper


DTYPES = {
    "f16": TensorProto.FLOAT16,
    "f32": TensorProto.FLOAT,
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate an ISOLDE RedMulE split-complex GEMM ONNX model"
    )
    p.add_argument("--m", type=int, default=12)
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--k", type=int, default=16)
    p.add_argument(
        "--dtype",
        choices=sorted(DTYPES),
        default="f16",
        help="External tensor element type (default: f16)",
    )
    p.add_argument(
        "--onnx-opset",
        type=int,
        default=18,
        help="Standard ai.onnx opset imported by the model (default: 18)",
    )
    p.add_argument(
        "--isolde-opset",
        type=int,
        default=1,
        help="com.isolde custom-domain opset (default: 1)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("redmule_complex_gemm_12x16x16.onnx"),
    )
    return p.parse_args()


def make_model(m, n, k, dtype, onnx_opset, isolde_opset):
    if min(m, n, k) <= 0:
        raise ValueError("M, N and K must all be positive")

    elem_type = DTYPES[dtype]

    inputs = [
        helper.make_tensor_value_info("Ar", elem_type, [m, n]),
        helper.make_tensor_value_info("Ai", elem_type, [m, n]),
        helper.make_tensor_value_info("Br", elem_type, [n, k]),
        helper.make_tensor_value_info("Bi", elem_type, [n, k]),
    ]
    outputs = [
        helper.make_tensor_value_info("Cr", elem_type, [m, k]),
        helper.make_tensor_value_info("Ci", elem_type, [m, k]),
    ]

    node = helper.make_node(
        "RedMulEComplexGemm",
        inputs=["Ar", "Ai", "Br", "Bi"],
        outputs=["Cr", "Ci"],
        domain="com.isolde",
        name="RedMulEComplexGemm_0",
    )

    graph = helper.make_graph(
        [node],
        f"redmule_complex_gemm_{m}x{n}x{k}_{dtype}",
        inputs,
        outputs,
    )

    model = helper.make_model(
        graph,
        producer_name="ISOLDE-Project/onnx-mlir",
        opset_imports=[
            helper.make_opsetid("", onnx_opset),
            helper.make_opsetid("com.isolde", isolde_opset),
        ],
    )

    # The generic ONNX checker cannot validate the semantics of an unregistered
    # vendor-domain operator. Structural graph/type correctness is provided by
    # the explicit input/output ValueInfo above; ONNX-MLIR owns the custom op.
    return model


def main():
    a = parse_args()
    model = make_model(
        a.m, a.n, a.k, a.dtype, a.onnx_opset, a.isolde_opset
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, a.out)

    print(f"Generated: {a.out}")
    print(f"  operator : com.isolde::RedMulEComplexGemm")
    print(f"  Ar/Ai    : [{a.m}, {a.n}] {a.dtype}")
    print(f"  Br/Bi    : [{a.n}, {a.k}] {a.dtype}")
    print(f"  Cr/Ci    : [{a.m}, {a.k}] {a.dtype}")
    print(f"  domains  : ai.onnx={a.onnx_opset}, com.isolde={a.isolde_opset}")


if __name__ == "__main__":
    main()
