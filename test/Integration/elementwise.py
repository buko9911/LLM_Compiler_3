"""VPU stream semantics and linalg.generic lowering against a PyTorch oracle.

Two cases. The hand-written target IR pins the instruction contract itself; the
graph source runs the same hardware through Stage 1..4 from a tensor matmul plus an
elementwise chain, which is the shape every decoder layer has.
"""
import argparse
import base64
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

import harness


def simulate(package, emulator, settings, address, rows, columns, l2_address=None):
    """Output at DRAM ``address``, read back through L2 (see harness.py)."""
    return harness.simulate(package, emulator, settings, address, rows, columns,
                            l2_address=l2_address)


def pack(manifest, arguments, tensors):
    """Build the DRAM image an entry point asks for.

    A weight the compiler folded a transpose into is written transposed; the
    package names those arguments, because nothing in the declared shape alone
    says which orientation the bytes are in.
    """
    flipped = set(manifest.get("pretransposed", []))
    image = bytearray(manifest["dram_bytes"])
    for descriptor, tensor in zip(arguments, tensors):
        if descriptor["index"] in flipped:
            tensor = tensor.T.contiguous()
        # A weight the graph reads panel by panel is stored that way, so one
        # panel is one transfer instead of one short row per K.
        if panel := descriptor.get("packed"):
            rows, columns = descriptor["rows"], descriptor["cols"]
            tensor = tensor.reshape(rows, columns // panel, panel)
            tensor = tensor.permute(1, 0, 2).contiguous().reshape(rows, columns)
        raw = tensor.numpy().astype("<f2").tobytes()
        assert len(raw) == descriptor["rows"] * descriptor["cols"] * 2, \
            f"argument {descriptor['index']} is not the declared shape"
        image[descriptor["address"]:descriptor["address"] + len(raw)] = raw
    return bytes(image)


def main():
    parser = argparse.ArgumentParser()
    for name in ("compiler", "emulator", "settings", "source", "graph", "softmax", "loop", "attention", "carry", "flash", "divide", "constant", "transpose", "layer", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    emulator, settings = args.emulator.resolve(), args.settings.resolve()

    # |x| <= 1/2 on the 1/16 grid keeps every product on 1/256, exact in FP16.
    def grid(count, seed, rows, columns):
        return ((torch.arange(count) * seed % 17 - 8).float() / 16).half().reshape(rows, columns)

    target = out / "target"
    subprocess.run([str(args.compiler), "--from=target", str(args.source),
                    "--settings", str(args.settings), "-o", str(target)], check=True)
    a = grid(1024, 7, 32, 32)
    expected = (a.float() * a.float()).half().numpy().astype("<f2")
    (target / "hbm.bin").write_bytes(a.numpy().astype("<f2").tobytes() + bytes(2048))
    actual = simulate(target, emulator, settings, 2048, 32, 32, l2_address=4096)
    np.testing.assert_array_equal(actual.view("<u2"), expected.view("<u2"),
                                  err_msg="hand-written VPU square is not bit-exact")

    graph = out / "graph"
    subprocess.run([str(args.compiler), "--from=graph", "--fp16", "--readback", str(args.graph),
                    "--settings", str(args.settings), "-o", str(graph)], check=True)
    manifest = json.loads((graph / "metadata.json").read_text())
    arguments = sorted(manifest["arguments"], key=lambda d: d["index"])
    result = manifest["outputs"][0]
    rows, columns = result["rows"], result["cols"]
    tensors = [grid(rows * columns, 7 + 3 * i, rows, columns) for i in range(len(arguments))]
    # Every VPU operation rounds its own result, so the oracle rounds at each step.
    y = (tensors[0].float() @ tensors[1].float()).half().float()
    expected = ((y * y).half().float() + y).half().numpy().astype("<f2")
    image = pack(manifest, arguments, tensors)
    (graph / "hbm.bin").write_bytes(bytes(image))
    actual = simulate(graph, emulator, settings, result["address"], rows, columns)
    np.testing.assert_array_equal(actual.view("<u2"), expected.view("<u2"),
                                  err_msg="lowered matmul->elementwise chain is not bit-exact")
    # Reduction, broadcast and scalar-lane paths together. exp is an SFU
    # approximation, so softmax is held to FP16 precision rather than bit equality.
    softmax = out / "softmax"
    subprocess.run([str(args.compiler), "--from=graph", "--fp16", "--readback", str(args.softmax),
                    "--settings", str(args.settings), "-o", str(softmax)], check=True)
    manifest = json.loads((softmax / "metadata.json").read_text())
    argument, result = manifest["arguments"][0], manifest["outputs"][0]
    rows, columns = argument["rows"], argument["cols"]
    a = grid(rows * columns, 7, rows, columns)
    image = bytearray(manifest["dram_bytes"])
    raw = a.numpy().astype("<f2").tobytes()
    image[argument["address"]:argument["address"] + len(raw)] = raw
    (softmax / "hbm.bin").write_bytes(bytes(image))
    actual = simulate(softmax, emulator, settings, result["address"], rows, columns).astype("f4")
    reference = torch.softmax(a.float(), dim=1).half().numpy().astype("f4")
    error = float(np.abs(actual - reference).max())
    assert error < 1e-3, f"softmax differs from torch.softmax by {error}"
    sums = actual.sum(axis=1)
    assert np.abs(sums - 1.0).max() < 1e-3, f"softmax rows do not sum to one: {sums.min()}..{sums.max()}"
    # The same row sums a 242-word unrolled block produces, in 81 words.
    loop = out / "loop"
    subprocess.run([str(args.compiler), "--from=target", str(args.loop),
                    "--settings", str(args.settings), "-o", str(loop)], check=True)
    a = grid(1024, 7, 32, 32)
    expected = a.float().sum(dim=1).numpy().astype("<f4")
    (loop / "hbm.bin").write_bytes(a.numpy().astype("<f2").tobytes() + bytes(128))
    harness.run(loop, emulator, settings)
    sums = harness.read(loop, 2176, 1, 32, dtype="<f4").reshape(32)
    np.testing.assert_array_equal(sums.view("<u4"), expected.view("<u4"),
                                  err_msg="hardware-loop row sums are not bit-exact")

    # FlashAttention over two KV blocks: running max and sum carried across the
    # blocks, O rescaled and accumulated on the VPU. FP16 exp keeps this to FP16
    # precision rather than bit equality.
    attention = out / "attention"
    subprocess.run([str(args.compiler), "--from=target", str(args.attention),
                    "--settings", str(args.settings), "-o", str(attention)], check=True)
    br = d = bc = 32
    blocks = 2
    q = grid(br * d, 7, br, d)
    kt = [grid(d * bc, 3 + 2 * j, d, bc) for j in range(blocks)]
    v = [grid(bc * d, 5 + 2 * j, bc, d) for j in range(blocks)]
    image = bytearray(12288)

    def place(offset, tensor):
        raw = tensor.numpy().astype("<f2").tobytes()
        image[offset:offset + len(raw)] = raw

    place(0, q)
    for j in range(blocks):
        place(2048 + j * 2048, kt[j])
        place(6144 + j * 2048, v[j])
    (attention / "hbm.bin").write_bytes(bytes(image))
    actual = simulate(attention, emulator, settings, 10240, br, d, l2_address=12288).astype("f4")
    scores = torch.cat([q.float() @ kt[j].float() for j in range(blocks)], dim=1)
    values = torch.cat([t.float() for t in v], dim=0)
    reference = (torch.softmax(scores, dim=1) @ values).numpy().astype("f4")
    attention_error = float(np.abs(actual - reference).max())
    assert attention_error < 5e-3, f"FlashAttention differs from dense attention by {attention_error}"

    # scf.for carrying a running accumulator: the shape of a KV-block loop. The
    # zero init arrives as an argument because the compiler is value-free.
    carry = out / "carry"
    subprocess.run([str(args.compiler), "--from=graph", "--fp16", "--readback", str(args.carry),
                    "--settings", str(args.settings), "-o", str(carry)], check=True)
    manifest = json.loads((carry / "metadata.json").read_text())
    arguments = sorted(manifest["arguments"], key=lambda d: d["index"])
    result = manifest["outputs"][0]
    n = result["rows"]
    left, right = grid(n * n, 7, n, n), grid(n * n, 3, n, n)
    tensors = [left, right, torch.zeros(n, n).half()]
    # Two iterations, each adding one FP16-rounded product into the accumulator.
    expected = (2 * (left.float() @ right.float()).half().float()).half().numpy().astype("<f2")
    image = pack(manifest, arguments, tensors)
    (carry / "hbm.bin").write_bytes(bytes(image))
    actual = simulate(carry, emulator, settings, result["address"], n, n)
    np.testing.assert_array_equal(actual.view("<u2"), expected.view("<u2"),
                                  err_msg="loop-carried accumulation is not bit-exact")

    # The same FlashAttention, written as ordinary linalg and scf.for with no
    # dedicated operation, lowered by the general paths.
    flash = out / "flash"
    subprocess.run([str(args.compiler), "--from=graph", "--fp16", "--readback", str(args.flash),
                    "--settings", str(args.settings), "-o", str(flash)], check=True)
    manifest = json.loads((flash / "metadata.json").read_text())
    arguments = sorted(manifest["arguments"], key=lambda d: d["index"])
    result = manifest["outputs"][0]
    br, kv = 32, 64
    queries = grid(br * br, 7, br, br)
    keys = grid(br * kv, 3, br, kv)
    values = grid(kv * br, 5, kv, br)
    # m starts at a finite floor, not -inf: the rescale derives its zero as m - m.
    initial = [queries, keys, values, torch.full((br,), -65504.0),
               torch.zeros(br), torch.zeros(br, br).half()]
    image = bytearray(manifest["dram_bytes"])
    for descriptor, tensor in zip(arguments, initial):
        kind = "<f4" if descriptor["dtype"] == "float32" else "<f2"
        raw = tensor.numpy().astype(kind).tobytes()
        image[descriptor["address"]:descriptor["address"] + len(raw)] = raw
    (flash / "hbm.bin").write_bytes(bytes(image))
    actual = simulate(flash, emulator, settings, result["address"], br, br).astype("f4")
    reference = (torch.softmax(queries.float() @ keys.float(), dim=1) @ values.float()).numpy().astype("f4")
    flash_error = float(np.abs(actual - reference).max())
    assert flash_error < 5e-3, f"lowered FlashAttention differs from dense attention by {flash_error}"

    # a/b as a reciprocal and a product, behind its own opt-in.
    divide = out / "divide"
    subprocess.run([str(args.compiler), "--from=graph", "--fp16", "--readback", "--reciprocal-division",
                    str(args.divide), "--settings", str(args.settings), "-o", str(divide)], check=True)
    refused = subprocess.run([str(args.compiler), "--from=graph", "--fp16", "--readback", str(args.divide),
                              "--settings", str(args.settings), "-o", str(out / "refused")],
                             capture_output=True, text=True)
    assert refused.returncode, "division must not ride along on --fp16"
    manifest = json.loads((divide / "metadata.json").read_text())
    argument, result = manifest["arguments"][0], manifest["outputs"][0]
    side = argument["rows"]
    x = grid(side * side, 7, side, side)
    image = bytearray(manifest["dram_bytes"])
    raw = x.numpy().astype("<f2").tobytes()
    image[argument["address"]:argument["address"] + len(raw)] = raw
    (divide / "hbm.bin").write_bytes(bytes(image))
    actual = simulate(divide, emulator, settings, result["address"], side, side)
    # Every step rounds to FP16, and the quotient is a product with the reciprocal.
    exponential = (-x.float()).half().float().exp().half().float()
    denominator = (exponential + x.float()).half().float()
    expected = (x.float() * (1.0 / denominator).half().float()).half().numpy().astype("<f2")
    np.testing.assert_array_equal(actual.view("<u2"), expected.view("<u2"),
                                  err_msg="reciprocal division is not bit-exact")

    # A constant tensor arrives as an argument whose contents the package names.
    constant = out / "constant"
    subprocess.run([str(args.compiler), "--from=graph", "--fp16", "--readback", str(args.constant),
                    "--settings", str(args.settings), "-o", str(constant)], check=True)
    manifest = json.loads((constant / "metadata.json").read_text())
    arguments = {d["index"]: d for d in manifest["arguments"]}
    assert manifest["constants"], "the package must name the constant's contents"
    result = manifest["outputs"][0]
    side = arguments[0]["rows"]
    x = grid(side * side, 7, side, side)
    image = bytearray(manifest["dram_bytes"])
    raw = x.numpy().astype("<f2").tobytes()
    image[arguments[0]["address"]:arguments[0]["address"] + len(raw)] = raw
    for entry in manifest["constants"]:
        payload = base64.b64decode(entry["base64"])
        assert len(payload) == entry["bytes"]
        at = arguments[entry["index"]]["address"]
        image[at:at + len(payload)] = payload
    (constant / "hbm.bin").write_bytes(bytes(image))
    actual = simulate(constant, emulator, settings, result["address"], side, side)
    expected = (x.float() * 2.0).half().numpy().astype("<f2")
    np.testing.assert_array_equal(actual.view("<u2"), expected.view("<u2"),
                                  err_msg="constant tensor is not bit-exact")

    # QK^T: the transpose materialises in L1, not over the DRAM tensor.
    transpose = out / "transpose"
    subprocess.run([str(args.compiler), "--from=graph", "--fp16", "--readback", str(args.transpose),
                    "--settings", str(args.settings), "-o", str(transpose)], check=True)
    manifest = json.loads((transpose / "metadata.json").read_text())
    arguments = sorted(manifest["arguments"], key=lambda d: d["index"])
    result = manifest["outputs"][0]
    queries = grid(32 * 32, 7, 32, 32)
    keys = grid(64 * 32, 3, 64, 32)
    image = pack(manifest, arguments, (queries, keys))
    (transpose / "hbm.bin").write_bytes(bytes(image))
    actual = simulate(transpose, emulator, settings, result["address"], 32, 64)
    expected = (queries.float() @ keys.float().T).half().numpy().astype("<f2")
    np.testing.assert_array_equal(actual.view("<u2"), expected.view("<u2"),
                                  err_msg="QK^T through a transpose is not bit-exact")

    # A whole decoder layer: RMSNorm, QKV, QK^T, softmax, PV, output projection,
    # residual, SwiGLU. Everything the general paths carry, in one graph.
    layer = out / "layer"
    subprocess.run([str(args.compiler), "--from=graph", "--fp16", "--readback", "--reciprocal-division",
                    str(args.layer), "--settings", str(args.settings), "-o", str(layer)], check=True)
    manifest = json.loads((layer / "metadata.json").read_text())
    arguments = sorted(manifest["arguments"], key=lambda d: d["index"])
    result = manifest["outputs"][0]
    tokens, hidden, ffn = 32, 64, 128
    shapes = [(tokens, hidden), (hidden, hidden), (hidden, hidden), (hidden, hidden),
              (hidden, hidden), (hidden, ffn), (hidden, ffn), (ffn, hidden)]
    seeds = [7, 3, 5, 9, 11, 13, 15, 17]
    weights = [grid(r * c, seed, r, c) for (r, c), seed in zip(shapes, seeds)]
    image = pack(manifest, arguments, weights)
    (layer / "hbm.bin").write_bytes(bytes(image))
    actual = simulate(layer, emulator, settings, result["address"], tokens, hidden).astype("f4")
    x, wq, wk, wv, wo, wg, wu, wd = (t.float() for t in weights)
    normalised = x * torch.rsqrt((x * x).sum(1, keepdim=True))
    scores = (normalised @ wq) @ (normalised @ wk).T
    context = torch.softmax(scores, dim=1) @ (normalised @ wv)
    residual = x + context @ wo
    gate, up = residual @ wg, residual @ wu
    swiglu = (gate / (torch.exp(-gate) + gate)) * up
    reference = (residual + swiglu @ wd).half().numpy().astype("f4")
    # FP16 end to end against an FP32 oracle, so this is a relative bound.
    layer_error = float(np.abs(actual - reference).max() / np.abs(reference).max())
    assert layer_error < 1e-2, f"decoder layer differs from the FP32 oracle by {layer_error}"

    print(f"VPU square (target IR), matmul->elementwise chain and {rows}x{columns} softmax "
          f"(graph IR), hardware-loop row sums, FlashAttention and scf.for carry: bit-exact where "
          f"claimed, softmax within {error:.1e} of torch, attention within {attention_error:.1e}, "
          f"loop-carried accumulation, reciprocal division, constants and QK^T exact, "
          f"lowered FlashAttention within {flash_error:.1e}, decoder layer within {layer_error:.1e}")


if __name__ == "__main__":
    main()
