#!/usr/bin/env python3
"""Compile a model into its three NPU programs from the model and three sizes.

    compile_model.py --model DIR --max-length L --chunk C --verify V --out OUT [--target FILE] [--force npu|pim]

A step processes M new tokens against a KV cache of fixed capacity (static shapes, as NPU LLM runtimes
do): every program attends over exactly L positions, the P = L - M cached ones plus its M new ones, and
the host's additive mask hides the cache slots not filled yet. So one program per token count serves
every position of a sequence up to L:

  OUT/decode   M = 1           all logits (one row)
  OUT/verify   M = V           all logits (a speculative block is checked row by row)
  OUT/chunk    M = C           last-row logits (a prompt chunk; the first chunk sees an empty cache)

Verify is a prompt chunk of V tokens that keeps every logits row; with C == V both programs are built.
The compiler gets nothing else: every optimization decision follows from these shapes and the target's
hardware numbers. OUT/programs.json lists the programs; each OUT/<name> is an npu-compile output
directory (with its frontend IR in OUT/<name>/ir).
"""
import argparse, json, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--model", type=Path, required=True)
ap.add_argument("--max-length", type=int, required=True, help="L: positions every program attends over")
ap.add_argument("--chunk", type=int, required=True, help="C: tokens per prompt chunk")
ap.add_argument("--verify", type=int, required=True, help="V: tokens per verify block")
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--target", type=Path, default=ROOT / "configs/baselines/npu_only_lp6_4ch.json")
ap.add_argument("--force", choices=["npu", "pim"], help="place every GEMV group on one device")
ap.add_argument("--layers", type=int, default=0, help="truncate to N layers (0 = all; for fixtures)")
ap.add_argument("--programs", nargs="+", choices=["decode", "verify", "chunk"], default=["decode", "verify", "chunk"])
ap.add_argument("--compiler", type=Path, default=ROOT / "build/bin/npu-compile")
ap.add_argument("--no-image", action="store_true", help="do not write lp6.bin")
a = ap.parse_args()

sizes = dict(decode=(1, "all"), verify=(a.verify, "all"), chunk=(a.chunk, "last"))
for name in a.programs:
    tokens = sizes[name][0]
    if not 0 < tokens < a.max_length:
        ap.error(f"{name}: {tokens} tokens must be in [1, max length {a.max_length})")
a.out.mkdir(parents=True, exist_ok=True)
programs = []
for name in a.programs:
    tokens, logits = sizes[name]
    capacity = a.max_length - tokens
    build, ir = a.out / name, a.out / name / "ir"
    t0 = time.time()
    subprocess.run([sys.executable, str(ROOT / "frontend/export.py"), "--model", str(a.model), "--tokens", str(tokens),
                    "--capacity", str(capacity), "--layers", str(a.layers), "--logits", logits, "--out", str(ir)],
                   check=True)
    t1 = time.time()
    command = [str(a.compiler), str(ir / "step.mlir"), "--manifest", str(ir / "manifest.json"), "--spec", str(a.target),
               "-o", str(build)] + ([f"--force={a.force}"] if a.force else []) + (["--no-image"] if a.no_image else [])
    with open(a.out / f"{name}.compile.log", "w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        sys.exit(f"{name}: npu-compile failed, see {a.out / (name + '.compile.log')}")
    programs.append(dict(name=name, tokens=tokens, capacity=capacity, attention_positions=a.max_length,
                         logits=logits, build=str(build), export_seconds=round(t1 - t0, 1),
                         compile_seconds=round(time.time() - t1, 1)))
    print(f"{name}: M={tokens} cache={capacity} logits={logits} -> {build} "
          f"[export {t1 - t0:.0f}s, compile {time.time() - t1:.0f}s]", flush=True)
(a.out / "programs.json").write_text(json.dumps(dict(
    model=str(a.model), max_length=a.max_length, chunk=a.chunk, verify=a.verify, target=str(a.target),
    force=a.force, layers=a.layers or None, programs=programs), indent=1) + "\n")
