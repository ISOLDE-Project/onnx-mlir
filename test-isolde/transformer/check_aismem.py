#!/usr/bin/env python3
"""Run an emitted AISMEM RedMulE schedule on a host model and check it.

Usage: check_aismem.py graph.spade.mlir model.npz

The input is the IR written by `onnx-mlir --EmitSPADEMLIR`.  Its entry
function is interpreted straight-line:

    aismem.RedMulEAddrStart / RedMulEUpload / RedMulEUploadTile /
    RedMulEZero / RedMulEGEMM / RedMulEWait / RedMulEDownload
    aismem.SPMAlloc (placed: address = row * 64) / SPMRelu / SPMTranspose /
    SPMCopy
    memref.alloc, krnl.global, builtin.unrealized_conversion_cast, return

If the module has a `main_graph_preload` function (resident weights), it is
run first, on the same SPM state, exactly as firmware calls it once at boot.
The report counts the data moved between data memory and SPM.

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
# SPM rows per tile the RTL addresses (tmp/cluster: TCDM_AW = 10 -> 256; a
# row beyond aliases row - 256).  SPM_ROWS=512 in the environment for an RTL
# with the whole 32 KiB window.
SPM_ROWS = int(__import__("os").environ.get("SPM_ROWS", "256"))

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
        self.uploaded = 0      # fp16 values DMEM -> SPM
        self.downloaded = 0    # fp16 values SPM -> DMEM
        self.core_spm = 0      # fp16 values touched in SPM by the core
        self.buf = {}          # SSA value -> SPM buffer name (SPMAlloc)
        self.owner = {}        # (tile, row) -> buffer that last wrote it
        self.violations = []

    # --- SPM model --------------------------------------------------------
    def rows(self, tile, addr, count, who=None):
        """Read; with `who`, check that no other buffer overwrote the rows."""
        bank = self.spm.setdefault(tile, {})
        start = addr // ROW_BYTES
        self.check_rows(tile, start, count)
        if who is not None:
            for i in range(count):
                o = self.owner.get((tile, start + i), who)
                if o != who:
                    self.violations.append(
                        f"{who}: row {start + i} of tile {tile} was "
                        f"overwritten by {o}")
        return np.stack([bank.get(start + i, np.zeros(ROW_ELEMS, np.float16))
                         for i in range(count)])

    def check_rows(self, tile, start, count):
        if start + count > SPM_ROWS:
            self.violations.append(
                f"tile {tile}: rows [{start}, {start + count}) beyond the "
                f"{SPM_ROWS} rows of the SPM")

    def store(self, tile, addr, matrix, who=None):
        bank = self.spm.setdefault(tile, {})
        start = addr // ROW_BYTES
        self.check_rows(tile, start, matrix.size // ROW_ELEMS)
        for i, row in enumerate(matrix.reshape(-1, ROW_ELEMS)):
            bank[start + i] = row.astype(np.float16).copy()
            self.owner[(tile, start + i)] = who
        return addr + matrix.size // ROW_ELEMS * ROW_BYTES

    # --- ops --------------------------------------------------------------
    def get(self, name):
        v = self.values[name]
        return self.values[v] if isinstance(v, str) else v

    def run(self, op, line, res, ops):
        a = attrs_of(line)
        b = [self.buf.get(o) for o in ops]
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
            self.uploaded += mat.size
            nxt = self.store(a["tile"], addr, mat,
                             b[2] if op == "aismem.RedMulEUpload" else b[1])
            self.values[res[0]] = nxt
            self.values[res[1]] = None
        elif op == "aismem.RedMulEZero":
            n = a["elements"]
            self.store(a["tile"], self.get(ops[0]),
                       np.zeros((n // ROW_ELEMS, ROW_ELEMS), np.float16), b[0])
            self.values[res[0]] = None
        elif op == "aismem.RedMulEGEMM":
            t, m, n, k = a["tile"], a["m"], a["n"], a["k"]
            x = self.rows(t, self.get(ops[0]), m, b[0])[:, :n]
            w = self.rows(t, self.get(ops[1]), n, b[1])[:, :k]
            y = self.rows(t, self.get(ops[2]), m, b[2])[:, :k]
            self.store(t, self.get(ops[2]), gemm16(x, w, y), b[2])
            self.values[res[0]] = None
        elif op == "aismem.RedMulEWait":
            self.values[res[0]] = None
        elif op == "aismem.RedMulEDownload":
            dst = self.get(ops[1])
            n = a["elements"]
            self.downloaded += n
            cols = a.get("cols", ROW_ELEMS)
            rows = self.rows(a["tile"], self.get(ops[0]), n // cols,
                             b[0])[:, :cols]
            off, ld = a.get("dst_offset", 0), a.get("dst_ld", ROW_ELEMS)
            flat = dst.reshape(-1)
            if off == 0 and ld == ROW_ELEMS and cols == ROW_ELEMS:
                flat[:] = rows.reshape(-1)[:dst.size]
            else:  # a tile of a wider result, or a partial tile
                for r, row in enumerate(rows):
                    flat[off + r * ld:off + r * ld + cols] = row
            self.values[res[0]] = None
        elif op == "aismem.SPMRelu":
            t, addr, n = a["tile"], self.get(ops[0]), a["rows"]
            bits = self.rows(t, addr, n, b[0]).view(np.uint16).copy()
            bits[bits & 0x8000 != 0] = 0
            self.store(t, addr, bits.view(np.float16), b[0])
            self.core_spm += n * ROW_ELEMS
            self.values[res[0]] = None
        elif op == "aismem.SPMTranspose":
            t, n, dn = a["tile"], a["rows"], a.get("dst_rows", 16)
            src = self.rows(t, self.get(ops[0]), n, b[0])
            dst = np.zeros((dn, ROW_ELEMS), np.float16)
            dst[:, :n] = src.T[:dn, :]
            self.store(t, self.get(ops[1]), dst, b[1])
            self.core_spm += (n + dn) * ROW_ELEMS
            self.values[res[0]] = None
        elif op == "aismem.SPMMoveTile":
            # tile -> data memory -> tile (omrm_spm_move_f16)
            n, dn = a["rows"], a["dst_rows"]
            src = self.rows(a["src_tile"], self.get(ops[0]), n, b[0])
            if a.get("transpose"):
                src = src.T
            dst = np.zeros((dn, ROW_ELEMS), np.float16)
            dst[:src.shape[0], :src.shape[1]] = src
            self.store(a["dst_tile"], self.get(ops[1]), dst, b[1])
            self.downloaded += n * ROW_ELEMS
            self.uploaded += dn * ROW_ELEMS
            self.values[res[0]] = None
        elif op == "aismem.SPMCopy":
            t, n = a["tile"], a["rows"]
            self.store(t, self.get(ops[1]),
                       self.rows(t, self.get(ops[0]), n, b[0]), b[1])
            self.core_spm += 2 * n * ROW_ELEMS
            self.values[res[0]] = None
        else:
            raise NotImplementedError(op)


def run_function(mach, text, name, args_values):
    fm = re.search(r"func\.func @" + re.escape(name) + r"\((.*?)\)(.*?)\{\n(.*?)\n  \}",
                   text, re.S)
    if not fm:
        return None, None
    mach.values = {}
    mach.buf = {}
    args = re.findall(r"(%arg\d+): (memref<[^>]*>)", fm.group(1))
    for (arg, _), value in zip(args, args_values):
        mach.values[arg] = value
    unknown, result = [], None
    for line in fm.group(3).splitlines():
        s = line.strip()
        if not s or s.startswith("//"):
            continue
        res = results_of(line)
        m = re.search(r'"([\w.]+)"\(', s) or re.search(r"= ([\w.]+)", s) \
            or re.match(r"([\w.]+)", s)
        op = m.group(1)
        if op == "aismem.SPMAlloc":
            a = attrs_of(s)
            if "row" not in a:
                sys.exit("unplaced aismem.SPMAlloc: run aismem-spm-allocate")
            mach.values[res[0]] = a["row"] * ROW_BYTES
            nm = re.search(r'name = "([^"]*)"', s)
            mach.buf[res[0]] = nm.group(1) if nm else res[0]
        elif op.startswith("aismem.RedMulE") or op.startswith("aismem.SPM"):
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
            r = re.search(r"return (%[\w#]+)", s)
            result = r.group(1) if r else None
        elif op in IGNORED:
            continue
        else:
            unknown.append(s)
    return result, unknown


def main():
    mlir_path, npz_path = sys.argv[1], sys.argv[2]
    ref = np.load(npz_path)
    text = open(mlir_path).read()

    mach = Machine([])
    unknown = []
    if "func.func @main_graph_preload" in text:
        _, u = run_function(mach, text, "main_graph_preload", [])
        unknown += u
        print(f"preload: {mach.uploaded} fp16 uploaded once (resident)")
        mach.uploaded = 0
    # Two inferences on the same SPM state: resident data clobbered by the
    # first one shows up in the second.
    outputs = []
    for run in range(2):
        if run == 1:
            mach.uploaded = mach.downloaded = mach.core_spm = 0
        result, u = run_function(mach, text, "main_graph",
                                 [ref["h"].astype(np.float16).copy()])
        if result is None:
            sys.exit("no main_graph function found")
        outputs.append(mach.get(result).copy())
    unknown += u

    if unknown:
        print("host operations left in the entry function:")
        print("\n".join("  " + x for x in unknown))
    for v in sorted(set(mach.violations)):
        print("SPM:", v)
    exact = True
    for y in outputs:
        y = y.reshape(ref["y_redmule"].shape)
        exact &= np.array_equal(y.view(np.uint16),
                                ref["y_redmule"].view(np.uint16))
    err = np.abs(y.astype(np.float64) - ref["y_float"]).max()
    print(f"per inference: {mach.uploaded} fp16 uploaded, {mach.downloaded} "
          f"downloaded, {mach.core_spm} touched in SPM by the core")
    print(f"bit-exact vs RedMulE FP16 reference: {exact}")
    if not exact and y.size <= 64:
        print("  got     ", y.reshape(-1))
        print("  expected", ref["y_redmule"].reshape(-1))
    print(f"max |y - float64 reference| = {err:.4g}")
    sys.exit(0 if exact and not unknown and not mach.violations else 1)


if __name__ == "__main__":
    main()
