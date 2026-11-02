"""Check decode routing and malformed selectors."""
import argparse
import subprocess
import tempfile
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--compiler", required=True)
a = p.parse_args()
for args, code, needle in [
    (["--help"], 0, "--pim-bankset-pingpong"),
    (["--pipeline=graph", "--help"], 2, "--pipeline must be decode"),
    (["--pipeline=prefill", "--help"], 2, "--pipeline must be decode"),
    (["--pipeline=graph-hybrid", "--help"], 2, "--pipeline must be decode"),
    (["--pipeline=decode", "--help"], 0, "--force"),
    (["--pipeline", "decode", "--help"], 0, "--pim-bankset-pingpong"),
    (["--manifest=unused.json", "--help"], 0, "--no-output-collect"),
    (["--pipeline=unknown"], 2, "must be decode"),
    (["--pipeline"], 2, "needs a value"),
    (["--pipeline=decode", "--pipeline=graph"], 2, "only once"),
    (["--pipeline=graph", "--manifest=unused.json"], 2, "--pipeline must be decode"),
]:
    r = subprocess.run([a.compiler, *args], text=True, capture_output=True)
    assert r.returncode == code and needle in r.stdout + r.stderr, (args, r.returncode, r.stdout, r.stderr)
# Optimizations have no switches: the removed options are rejected.
with tempfile.TemporaryDirectory(prefix="npu-removed-options-") as output:
    base = [a.compiler, "--pipeline=decode", "unused.mlir", "--manifest=unused.json",
            "--spec=unused.json", "-o", output]
    for selector in ["--npu-phase-split=none", "--npu-weight-prefetch=false"]:
        r = subprocess.run([*base, selector], text=True, capture_output=True)
        assert r.returncode != 0 and "Unknown command line argument" in r.stderr, (selector, r.stderr)
    # A target key that is not a hardware number (here a removed on/off switch) is an error.
    spec, source = Path(output)/"spec.json", Path(output)/"step.mlir"
    spec.write_text('{"npu": {"cores": 4, "projection_fusion": true}}')
    source.write_text('module { func.func @decode_step() { return } }')
    opt = Path(a.compiler).with_name("npu-opt")
    r = subprocess.run([str(opt), str(source), f"--hyb-npu-projection-fusion=spec={spec}"],
                       text=True, capture_output=True)
    assert r.returncode != 0 and "npu.projection_fusion is not a target parameter" in r.stderr, r.stderr
with tempfile.TemporaryDirectory(prefix="npu-reserved-init-") as output:
    for attr in ("hyb.elide_matrix_init", "hyb.matrix_init_elisions", "hyb.gemv_epilogue", "hyb.gemv_epilogue_fusions",
                 "hyb.elementwise_distributed", "hyb.elementwise_axis", "hyb.elementwise_distributions"):
        source = Path(output)/"reserved.mlir"
        source.write_text(f'module {{ func.func @decode_step() attributes {{{attr}}} {{ return }} }}')
        r = subprocess.run([a.compiler, "--pipeline=decode", str(source), "--manifest=unused.json",
                            "--spec=unused.json", "--force=npu", "-o", str(Path(output)/"build")],
                           text=True, capture_output=True)
        assert r.returncode and "reserved compiler IR attributes" in r.stderr, (attr,r.stderr)
print("21 compiler routing/diagnostic checks passed")
