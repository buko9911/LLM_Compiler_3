"""Driver contracts, hardware-bound restart, and assembler failure diagnostics."""
import argparse
import base64
import json
import re
import struct
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--assembler", required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    source = args.root / "Integration/matmul32.mlir"
    settings = args.root / "Inputs/hardware.toml"

    def call(command, success=True, input=None):
        result = subprocess.run(list(map(str, command)), input=input, capture_output=True, text=True)
        assert (result.returncode == 0) == success, result.stdout + result.stderr
        return result

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        base = [args.compiler, "--from=target", source, "--settings", settings]
        call([*base, "-o", root / "plain"])
        call([*base, "-o", root / "debug", "--reparse-stages"])
        restart = root / "debug/03-target.mlir"
        call([args.compiler, "--from=target", restart, "--settings", settings, "-o", root / "restart"])
        for name in ("program.bin", "system.json"):
            for mode in ("debug", "restart"):
                assert (root / mode / name).read_bytes() == (root / "plain" / name).read_bytes()
        changed = root / "changed.toml"
        changed.write_text(settings.read_text() + "\n# different hardware identity\n")
        result = call([args.compiler, "--from=target", restart, "--settings", changed,
                       "-o", root / "bad"], success=False)
        assert "fingerprint" in result.stderr
        assert not (root / "bad").exists()
        changed.write_text(settings.read_text().replace("l1_sram_size_bytes = 4194304", "l1_sram_size_bytes = 64"))
        result = call([args.compiler, "--from=target", source, "--settings", changed,
                       "-o", root / "bad"], success=False)
        assert "capacity" in result.stderr
        result = call([args.compiler, source, "-o", root / "bad"], success=False)
        assert "--from=target" in result.stderr
        # ISA ver 1.0: a matrix load is a four-word record with its shape inline.
        result = call([args.assembler], input="M_LOAD_ACT_F16 5, 4, 64, 128\nM_MMA_F16F16F32 ACC\n")
        assert result.stdout == "0x01c01437\n0x00000004\n0x00000040\n0x00000080\n0x04c0003b\n", result.stdout
        decoded = call([args.assembler, "--disassemble"], input=result.stdout)
        assert decoded.stdout == "M_LOAD_ACT_F16 5, 4, 64, 128\nM_MMA_F16F16F32 ACC\n", decoded.stdout
        result = call([args.assembler], input="C_LUI_U32 0, 3junk\n", success=False)
        assert "line 1" in result.stderr
        call([args.assembler, "--disassemble"], input="0x00000001\n", success=False)
        # A long broadcast fits in row tiles even when all FP32 lanes together
        # exceed half of L1. Previously the full lane was charged to every tile.
        broadcast = root / "broadcast.mlir"
        broadcast.write_text("""
func.func @main(%a: tensor<1024x32xf16>, %b: tensor<1024xf32>) -> tensor<1024x32xf16> {
  %e = tensor.empty() : tensor<1024x32xf16>
  %r = linalg.generic {indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i)>, affine_map<(i,j)->(i,j)>], iterator_types = ["parallel", "parallel"]}
    ins(%a, %b : tensor<1024x32xf16>, tensor<1024xf32>) outs(%e : tensor<1024x32xf16>) {
  ^bb0(%x: f16, %s: f32, %o: f16):
    %t = arith.truncf %s : f32 to f16
    %v = arith.mulf %x, %t : f16
    linalg.yield %v : f16
  } -> tensor<1024x32xf16>
  return %r : tensor<1024x32xf16>
}
""")
        changed.write_text(settings.read_text().replace("l1_sram_size_bytes = 4194304",
                                                        "l1_sram_size_bytes = 8192"))
        call([args.compiler, "--from=graph", broadcast, "--fp16", "--settings", changed,
              "--reparse-stages", "-o", root / "broadcast"])
        f32inputs = root / "f32inputs"
        f32inputs.mkdir()
        (f32inputs / "arg0.bin").write_bytes(bytes(1024 * 32 * 2))
        (f32inputs / "arg1.bin").write_bytes(struct.pack("<f", 1.0) * 1024)
        call([args.compiler, "--from=graph", broadcast, "--fp16", "--settings", changed,
              "--inputs", f32inputs, "-o", root / "broadcast-inputs"])
        # Force multiple weight panels, then verify image bytes independently.
        # Nonuniform constants expose the old bypass of the packing path.
        packed = root / "packed.mlir"
        values = ", ".join("[" + ", ".join(str((r * 7 + c) % 19) + ".0"
                          for c in range(256)) + "]" for r in range(256))
        packed.write_text("""
func.func @main(%a: tensor<32x256xf16>) -> tensor<32x256xf16> {
  %w = arith.constant dense<[VALUES]> : tensor<256x256xf16>
  %z = arith.constant 0.0 : f16
  %e = tensor.empty() : tensor<32x256xf16>
  %i = linalg.fill ins(%z : f16) outs(%e : tensor<32x256xf16>) -> tensor<32x256xf16>
  %r = linalg.matmul ins(%a, %w : tensor<32x256xf16>, tensor<256x256xf16>)
       outs(%i : tensor<32x256xf16>) -> tensor<32x256xf16>
  return %r : tensor<32x256xf16>
}
""".replace("VALUES", values))
        changed.write_text(settings.read_text().replace("size_bytes = 8388608", "size_bytes = 262144"))
        inputs = root / "inputs"
        inputs.mkdir()
        (inputs / "arg0.bin").write_bytes(bytes(32 * 256 * 2))
        call([args.compiler, "--from=graph", packed, "--fp16", "--settings", changed,
              "--inputs", inputs, "--reparse-stages", "-o", root / "packed"])
        metadata = json.loads((root / "packed/metadata.json").read_text())
        weight = metadata["arguments"][1]
        panel = weight["packed"]
        assert panel < 256, "regression must exercise multiple panels"
        expected = b"".join(struct.pack("<e", (r * 7 + c) % 19)
                            for first in range(0, 256, panel)
                            for r in range(256) for c in range(first, first + panel))
        image = (root / "packed/hbm.bin").read_bytes()
        assert image[weight["address"]:weight["address"] + len(expected)] == expected
        # Metadata constants remain logical row-major; image creation owns packing.
        logical = base64.b64decode(metadata["constants"][0]["base64"])
        assert logical != expected
        # Local async transfers must preserve both L2 WAR and RAW hazards.
        header = (root / "packed/03-placed.mlir").read_text().splitlines()[0]
        hazards = root / "hazards.mlir"
        hazards.write_text(header + """
func.func @main() {
  %a = memref.alloc() {npu.address = 0 : i64} : memref<32xf16, 1>
  %b = memref.alloc() {npu.address = 0 : i64} : memref<32xf16, 2>
  %c = memref.alloc() {npu.address = 128 : i64} : memref<32xf16, 1>
  %d = memref.alloc() {npu.address = 128 : i64} : memref<32xf16, 2>
  memref.copy %a, %b : memref<32xf16, 1> to memref<32xf16, 2>
  memref.copy %b, %a : memref<32xf16, 2> to memref<32xf16, 1>
  memref.copy %a, %d : memref<32xf16, 1> to memref<32xf16, 2>
  memref.copy %d, %c : memref<32xf16, 2> to memref<32xf16, 1>
  %out = memref.alloc() {npu.address = 256 : i64} : memref<2x32xf16, 1>
  %tile = memref.alloc() {npu.address = 256 : i64} : memref<2x16xf16, 2>
  %left = memref.subview %out[0, 0] [2, 16] [1, 1] : memref<2x32xf16, 1> to memref<2x16xf16, strided<[32, 1]>, 1>
  %right = memref.subview %out[0, 16] [2, 16] [1, 1] : memref<2x32xf16, 1> to memref<2x16xf16, strided<[32, 1], offset: 16>, 1>
  memref.copy %tile, %left : memref<2x16xf16, 2> to memref<2x16xf16, strided<[32, 1]>, 1>
  memref.copy %tile, %right : memref<2x16xf16, 2> to memref<2x16xf16, strided<[32, 1], offset: 16>, 1>
  return
}
}
""")
        call([args.compiler, "--from=placed", hazards, "--settings", changed,
              "--save-stages", "-o", root / "hazards"])
        target = (root / "hazards/03-target.mlir").read_text()
        assert target.count("C_WAIT_LDMA") == 2
        assert "L2_LOAD_STRIDED_ASYNC" in target and "L2_STORE_STRIDED_ASYNC" in target
        # Residency follows contents, not reused L2 addresses. A DRAM write
        # invalidates the old key; a small L1 must retain the uncached path.
        fingerprint = re.search(r'npu.hardware = "([^"]+)"', header).group(1)
        cache = root / "cache.mlir"
        cache_source = """
module attributes {npu.stage = "placed", npu.schema = 1 : i64,
  npu.hardware = "FINGERPRINT", npu.cores = 1 : i64,
  npu.l1_bytes = 256 : i64, npu.l1_capacity = 4096 : i64,
  npu.l2_bytes = 512 : i64, npu.dram_bytes = 256 : i64} {
func.func @main(%a: memref<32xf16> {npu.address = 0 : i64},
                %b: memref<32xf16> {npu.address = 64 : i64}) {
  %l2 = memref.alloc() {npu.address = 0 : i64} : memref<32xf16, 1>
  %l1 = memref.alloc() {npu.address = 0 : i64} : memref<32xf16, 2>
  memref.copy %a, %l2 : memref<32xf16> to memref<32xf16, 1>
  linalg.copy {npu.cache_activation} ins(%l2 : memref<32xf16, 1>) outs(%l1 : memref<32xf16, 2>)
  memref.copy %a, %l2 : memref<32xf16> to memref<32xf16, 1>
  linalg.copy {npu.cache_activation} ins(%l2 : memref<32xf16, 1>) outs(%l1 : memref<32xf16, 2>)
  memref.copy %b, %l2 : memref<32xf16> to memref<32xf16, 1>
  memref.copy %l2, %a : memref<32xf16, 1> to memref<32xf16>
  memref.copy %a, %l2 : memref<32xf16> to memref<32xf16, 1>
  linalg.copy {npu.cache_activation} ins(%l2 : memref<32xf16, 1>) outs(%l1 : memref<32xf16, 2>)
  return
}
}
""".replace("FINGERPRINT", fingerprint)
        for mode, text, hits in [("resident", cache_source, 1),
                                 ("no-room", cache_source.replace("npu.l1_capacity = 4096", "npu.l1_capacity = 256"), 0)]:
            cache.write_text(text)
            call([args.compiler, "--from=placed", cache, "--settings", changed,
                  "--reparse-stages", "-o", root / mode])
            stats = json.loads((root / mode / "metadata.json").read_text())["local_optimizations"]
            assert stats["activation_cache_hits"] == hits, stats
    print("driver: restart, hardware, assembler, packed constants, FP32 inputs, async DMA hazards passed")


if __name__ == "__main__":
    main()
