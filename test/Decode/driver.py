"""Check input-contract routing, the prefill alias, and malformed selectors."""
import argparse
import subprocess
import tempfile
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--compiler", required=True)
a = p.parse_args()
for args, code, needle in [
    (["--help"], 0, "LLM_Compiler_3"),
    (["--pipeline=graph", "--help"], 0, "--from="),
    (["--pipeline=prefill", "--help"], 0, "--from="),
    (["--pipeline=graph-hybrid", "--help"], 0, "--split-every"),
    (["--pipeline=decode", "--help"], 0, "--npu-weight-prefetch"),
    (["--pipeline=decode", "--help"], 0, "--npu-phase-split"),
    (["--pipeline", "decode", "--help"], 0, "--pim-bankset-pingpong"),
    (["--manifest=unused.json", "--help"], 0, "--npu-weight-prefetch"),
    (["--pipeline=unknown"], 2, "must be prefill, graph, graph-hybrid or decode"),
    (["--pipeline"], 2, "needs a value"),
    (["--pipeline=decode", "--pipeline=graph"], 2, "only once"),
    (["--pipeline=graph", "--manifest=unused.json"], 2, "requires --pipeline=decode"),
]:
    r = subprocess.run([a.compiler, *args], text=True, capture_output=True)
    assert r.returncode == code and needle in r.stdout + r.stderr, (args, r.returncode, r.stdout, r.stderr)
with tempfile.TemporaryDirectory(prefix="npu-phase-options-") as output:
    base = [a.compiler, "--pipeline=decode", "unused.mlir", "--manifest=unused.json",
            "--spec=unused.json", "-o", output]
    for selectors in [
        ["--npu-phase-split=invalid"],
        ["--npu-phase-split=none"],
        ["--npu-phase-split=none", "--force=pim"],
        ["--npu-phase-split=none", "--force=npu", "--map=q_proj:npu"],
        ["--npu-phase-split=none", "--force=npu", "--pim-packet-npu=g4"],
    ]:
        r = subprocess.run([*base, *selectors], text=True, capture_output=True)
        assert r.returncode != 0 and "--npu-phase-split must be layer or none" in r.stderr, (
            selectors, r.returncode, r.stdout, r.stderr)
with tempfile.TemporaryDirectory(prefix="npu-reserved-init-") as output:
    for attr in ("hyb.elide_matrix_init", "hyb.matrix_init_elisions", "hyb.gemv_epilogue", "hyb.gemv_epilogue_fusions",
                 "hyb.elementwise_distributed", "hyb.elementwise_axis", "hyb.elementwise_distributions"):
        source = Path(output)/"reserved.mlir"
        source.write_text(f'module {{ func.func @decode_step() attributes {{{attr}}} {{ return }} }}')
        r = subprocess.run([a.compiler, "--pipeline=decode", str(source), "--manifest=unused.json",
                            "--spec=unused.json", "--force=npu", "-o", str(Path(output)/"build")],
                           text=True, capture_output=True)
        assert r.returncode and "reserved compiler IR attributes" in r.stderr, (attr,r.stderr)
print("24 compiler routing/diagnostic checks passed")
