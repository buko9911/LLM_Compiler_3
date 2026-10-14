"""Check input-contract routing, the prefill alias, and malformed selectors."""
import argparse
import subprocess

p = argparse.ArgumentParser()
p.add_argument("--compiler", required=True)
a = p.parse_args()
for args, code, needle in [
    (["--help"], 0, "LLM_Compiler_3"),
    (["--pipeline=graph", "--help"], 0, "--from="),
    (["--pipeline=prefill", "--help"], 0, "--from="),
    (["--pipeline=graph-hybrid", "--help"], 0, "--split-every"),
    (["--pipeline=decode", "--help"], 0, "--npu-weight-prefetch"),
    (["--pipeline", "decode", "--help"], 0, "--pim-bankset-pingpong"),
    (["--manifest=unused.json", "--help"], 0, "--npu-weight-prefetch"),
    (["--pipeline=unknown"], 2, "must be prefill, graph, graph-hybrid or decode"),
    (["--pipeline"], 2, "needs a value"),
    (["--pipeline=decode", "--pipeline=graph"], 2, "only once"),
    (["--pipeline=graph", "--manifest=unused.json"], 2, "requires --pipeline=decode"),
]:
    r = subprocess.run([a.compiler, *args], text=True, capture_output=True)
    assert r.returncode == code and needle in r.stdout + r.stderr, (args, r.returncode, r.stdout, r.stderr)
print("11 compiler routing/diagnostic checks passed")
