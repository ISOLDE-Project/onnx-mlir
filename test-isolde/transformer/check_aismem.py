#!/usr/bin/env python3
"""Run an emitted AISMEM RedMulE schedule on a host model and check it.

Usage: check_aismem.py graph.spade.mlir model.npz

The input is the IR written by `onnx-mlir --EmitSPADEMLIR`.  Its entry
function is interpreted straight-line:

    aismem.RedMulEAddrStart / RedMulEUpload / RedMulEUploadTile /
    RedMulEZero / RedMulEGEMM / RedMulEWait / RedMulEDownload
    memref.alloc, krnl.global, builtin.unrealized_conversion_cast, return

against a model of the tile-private SPMs (one 16-element FP16 row per 64-byte
SPM row, the spm_write layout of onnx_redmule_runtime.c).  Every RedMulE GEMM
rounds to FP16 after each of its 16 reduction steps, like the RTL reference.
The result is compared *bit-exactly* with `y_redmule` from the .npz written
by models/generate_transformer.py, and loosely with the float64 `y_float`.
Any other operation in the entry function is reported and fails the check:
the RedMulE path is expected to leave no host arithmetic behind.
"""
from __future__ import annotations

import re
import sys

import numpy as np

ROW_BYTES = 64
ROW_ELEMS = 16

IGNORED = ("aismem.qconstant", "memref.dealloc", "krnl.entry_point",
           "onnx.EntryPoint", "func.return", "return")


def parse_type(text):
    m = re.search(r"memref<([0-9x]+)xf16>", text)
    if not m:
        return None
    return tuple(int(d) for d in m.group(1).split("x"))


def parse_dense(body, shape):
    """dense<"0x...">, dense<[..]> or a splat dense<1.0>."""
    body = body.strip()
    if body.startswith('"0x'):
        raw = bytes.fromhex(body[3:-1])
        return np.frombuffer(raw, dtype="<f2").reshape(shape).copy()
    nums = [float(v) for v in re.findall(
        r"[-+]?(?:\d+\.?\d*(?:[eE][-+]?\d+)?|inf|nan|0x[0-9A-Fa-f]+)", body)]
    arr = np.array(nums, dtype=np.float16)
    if arr.size == 1:
        return np.full(shape, arr[0], dtype=np.float16)
    return arr.reshape(shape)


def attrs_of(line):
    out = {}
    for key, val in re.findall(r"(\w+) = (true|false|-?\d+)(?: : i\d+)?", line):
        out[key] = {"true": 1, "false": 0}.get(val, None)
        if out[key] is None:
            out[key] = int(val)
    return out


def operands_of(line):
    m = re.search(r'"\(([^)]*)\)', line) or re.search(r"\w\(([^)]*)\)", line)
    if not m:
        return []
    return [v.strip() for v in m.group(1).split(",") if v.strip()]


def results_of(line):
    m = re.match(r"\s*((?:%[\w#:]+(?:, )?)+) = ", line)
    if not m:
        return []
    names = []
    for r in m.group(1).split(","):
        r = r.strip()
        if ":" in r:           # %0:2 packs two results
            base, n = r.split(":")
            names += [f"{base}#{i}" for i in range(int(n))]
        else:
            names.append(r)
    return names


def gemm16(x, w, y):
    out = y.copy()
    for n in range(x.shape[1]):
        out = (x[:, n, None].astype(np.float32) * w[None, n, :].astype(np.float32)
               + out.astype(np.float32)).astype(np.float16)
    return out


class Machine:
    def __init__(self, inputs):
        self.values = {}
        self.spm = {}                    # tile -> {row index: 16 x f16}
        self.inputs = list(inputs)
        self.base = {}

    # --- SPM model --------------------------------------------------------
    def rows(self, tile, addr, count):
        bank = self.spm.setdefault(tile, {})
        start = addr // ROW_BYTES
        return np.stack([bank.get(start + i, np.zeros(ROW_ELEMS, np.float16))
                         for i in range(count)])

    def store(self, tile, addr, matrix):
        bank = self.spm.setdefault(tile, {})
        start = addr // ROW_BYTES
        for i, row in enumerate(matrix.reshape(-1, ROW_ELEMS)):
            bank[start + i] = row.astype(np.float16).copy()
        return addr + matrix.size // ROW_ELEMS * ROW_BYTES

    # --- ops --------------------------------------------------------------
    def get(self, name):
        v = self.values[name]
        return self.values[v] if isinstance(v, str) else v

    def run(self, op, line, res, ops):
        a = attrs_of(line)
        if op == "aismem.RedMulEAddrStart":
            # distinct, non-overlapping windows per tile; any base works
            self.values[res[0]] = 0
        elif op in ("aismem.RedMulEUpload", "aismem.RedMulEUploadTile"):
            src = self.get(ops[0])
            addr = self.get(ops[2] if op == "aismem.RedMulEUpload" else ops[1])
            if op == "aismem.RedMulEUpload":
                mat = src.reshape(-1, ROW_ELEMS)
                if a.get("negate"):
                    mat = (mat.view(np.uint16) ^ 0x8000).view(np.float16)
            else:
                ld = src.shape[-1]
                s2 = src.reshape(-1, ld)
                win = s2[a["row_offset"]:a["row_offset"] + a["rows"],
                         a["col_offset"]:a["col_offset"] + a["cols"]]
                if a.get("transpose"):
                    win = win.T
                bits = win.view(np.uint16).copy()
                if a.get("relu"):
                    bits[bits & 0x8000 != 0] = 0
                if a.get("negate"):
                    bits ^= 0x8000
                mat = np.zeros((a["dst_rows"], a.get("dst_cols", 16)), np.float16)
                mat[:bits.shape[0], :bits.shape[1]] = bits.view(np.float16)
            nxt = self.store(a["tile"], addr, mat)
            self.values[res[0]] = nxt
            self.values[res[1]] = None
        elif op == "aismem.RedMulEZero":
            n = a["elements"]
            self.store(a["tile"], self.get(ops[0]),
                       np.zeros((n // ROW_ELEMS, ROW_ELEMS), np.float16))
            self.values[res[0]] = None
        elif op == "aismem.RedMulEGEMM":
            t, m, n, k = a["tile"], a["m"], a["n"], a["k"]
            x = self.rows(t, self.get(ops[0]), m)[:, :n]
            w = self.rows(t, self.get(ops[1]), n)[:, :k]
            y = self.rows(t, self.get(ops[2]), m)[:, :k]
            self.store(t, self.get(ops[2]), gemm16(x, w, y))
            self.values[res[0]] = None
        elif op == "aismem.RedMulEWait":
            self.values[res[0]] = None
        elif op == "aismem.RedMulEDownload":
            dst = self.get(ops[1])
            n = a["elements"]
            dst.reshape(-1)[:] = self.rows(a["tile"], self.get(ops[0]),
                                           n // ROW_ELEMS).reshape(-1)[:dst.size]
            self.values[res[0]] = None
        else:
            raise NotImplementedError(op)


def main():
    mlir_path, npz_path = sys.argv[1], sys.argv[2]
    ref = np.load(npz_path)
    text = open(mlir_path).read()

    fm = re.search(r"func\.func @main_graph\((.*?)\)(.*?)\{\n(.*?)\n  \}",
                   text, re.S)
    if not fm:
        sys.exit("no main_graph function found")
    args = re.findall(r"(%arg\d+): (memref<[^>]*>)", fm.group(1))
    body = fm.group(3).splitlines()

    mach = Machine([])
    mach.values[args[0][0]] = ref["h"].astype(np.float16).copy()
    unknown, result = [], None
    for line in body:
        s = line.strip()
        if not s or s.startswith("//"):
            continue
        res = results_of(line)
        m = re.search(r'"([\w.]+)"\(', s) or re.search(r"= ([\w.]+)", s) \
            or re.match(r"([\w.]+)", s)
        op = m.group(1)
        if op.startswith("aismem.RedMulE"):
            mach.run(op, s, res, operands_of(s))
        elif op == "memref.alloc":
            mach.values[res[0]] = np.zeros(parse_type(s), np.float16)
        elif op == "krnl.global":
            shape = parse_type(s)
            dm = re.search(r"value = dense<(.*)> : tensor", s)
            if shape and dm:
                mach.values[res[0]] = parse_dense(dm.group(1), shape)
        elif op == "builtin.unrealized_conversion_cast":
            src = re.search(r"cast (%[\w#]+)", s).group(1)
            mach.values[res[0]] = src if src in mach.values else None
        elif op in ("return", "func.return", "onnx.Return"):
            result = re.search(r"return (%[\w#]+)", s).group(1)
        elif op in IGNORED:
            continue
        else:
            unknown.append(s)

    if unknown:
        print("host operations left in the entry function:")
        print("\n".join("  " + u for u in unknown))
    y = mach.get(result).reshape(ref["y_redmule"].shape)
    exact = np.array_equal(y.view(np.uint16), ref["y_redmule"].view(np.uint16))
    err = np.abs(y.astype(np.float64) - ref["y_float"]).max()
    print(f"bit-exact vs RedMulE FP16 reference: {exact}")
    print(f"max |y - float64 reference| = {err:.4g}")
    sys.exit(0 if exact and not unknown else 1)


if __name__ == "__main__":
    main()
