#!/usr/bin/env python3
"""Host runtime for a compiled decode step (the output directory of npu-compile --pipeline=decode).

Prefill is NOT compiled: the host runs the prompt through transformers (FP32, CPU) only to
build the KV cache. Every decode step runs on the simulator (PLENA NPU + LPDDR6-PIM, one
program.bin): the host writes x / cos / sin / mask / KV cache into the LP6 image, runs the
program, reads logits and the new K/V rows from the collect phase's L1 dump, appends the
K/V rows to its cache and takes the argmax token.

--check compares every step with two host references on the same inputs:
  fp32  explicit-state step (frontend/decoder.py), original weights, FP32
  w8a8  the same step with per-row INT8 weights (the image's) and token-wise INT8 activations
        at every linear, FP32 elsewhere
"""
from __future__ import annotations
import argparse, json, os, re, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "frontend"))
from decoder import DecoderDims, DecodeStep, weight_names, rope_tables, step_mask  # noqa: E402
from weights import SafetensorStore  # noqa: E402
from prefill import prefill_kv  # noqa: E402

LP6 = ROOT.parent
EMU = LP6 / "IESLAB_NPU_Simulator/transactional_emulator/target/release/transactional_emulator"
# The simulator's default settings: four LPDDR6 channels (102.4 GB/s), matching configs/target.json.
# Builds of configs/target_1ch.json (one channel) run with --lp6-config
# configs/LPDDR6_PIM_Mobile_2CH.ini; the simulator refuses a PIM program laid out for another
# subchannel count.
SETTINGS = LP6 / "IESLAB_NPU_Simulator/plena_settings.toml"
BACKEND = LP6 / "lp6-pim-main"
DT = {"f16": np.float16, "f32": np.float32, "i8": np.int8, "i32": np.int32}


def quantize_rows(W: torch.Tensor):
    """Per-row symmetric INT8, the image's weights (npu-compile --pipeline=decode quantizeRows): (q, scale)."""
    q = torch.empty(W.shape, dtype=torch.int8)
    s = torch.empty(W.shape[0], dtype=torch.float32)
    for r0 in range(0, W.shape[0], 8192):
        w = W[r0:r0 + 8192].float()
        sc = w.abs().amax(1) / 127.0
        sc = torch.where(sc > 0, sc, torch.full_like(sc, 1e-12))
        q[r0:r0 + 8192] = torch.clamp(torch.round(w / sc[:, None]), -127, 127).to(torch.int8)
        s[r0:r0 + 8192] = sc
    return q, s


class Device:
    """The compiled step on the simulator."""

    def __init__(self, out: Path, settings: Path = SETTINGS, lp6_config: Path | None = None,
                 emulator: Path = EMU):
        self.out = out
        self.emulator = emulator.resolve()
        self.settings = self._with_cores(settings)
        self.lp6_config = lp6_config.resolve() if lp6_config else None
        self.io = json.loads((out / "io.json").read_text())
        self.args = {a["name"]: a for a in self.io["args"]}
        self.outputs = {o["name"]: o for o in self.io["outputs"]}
        cores = len(json.loads((out / "system.json").read_text())["logical_to_physical"])
        self.dump_name = "sram_dump.bin" if cores == 1 else "sram_dump.core0.bin"

    def execution_env(self):
        """Match runtime lowering to the versioned, already packed weight image."""
        env = dict(os.environ, LP6_DRAMSIM3_ROOT=str(BACKEND))
        kernel = json.loads((self.out / "system.json").read_text()).get("pim_kernel")
        if not kernel or not self.io["pim_commands"]:
            return env
        layout = kernel.get("weight_layout", "even-v1")
        lead = kernel.get("bankset_pingpong_lead", 0)
        expected = "bankset-row-alternating-v1" if lead else "even-v1"
        if layout != expected or lead < 0 or lead > 63 or lead == 1:
            raise ValueError(f"unsupported PIM weight layout / lead: {layout}, {lead}")
        value = int(env.get("LP6PIM_BANKSET_PINGPONG", "0"))
        value = 12 if value == 1 else max(0, value)
        if "LP6PIM_BANKSET_PINGPONG" in env and value != lead:
            raise ValueError("LP6PIM_BANKSET_PINGPONG conflicts with the compiled weight layout; recompile or unset it")
        if int(env.get("LP6PIM_INLINE_SCALE_GROUP", "0")) > 0:
            raise ValueError("compiled W8A8 weight layouts do not support LP6PIM_INLINE_SCALE_GROUP")
        if lead and env.get("LP6PIM_GEMV_STREAM_MODE") in ("fixed", "output_fixed"):
            raise ValueError("bank-set ping-pong requires output-increment GEMV lowering")
        env["LP6PIM_BANKSET_PINGPONG"] = str(lead)
        return env

    def _with_cores(self, settings: Path) -> Path:
        """The settings with num_cores = the program's logical cores (a copy in the build when it differs)."""
        cores = len(json.loads((self.out / "system.json").read_text())["logical_to_physical"])
        text = settings.read_text()
        patched = re.sub(r"(?m)^num_cores\s*=\s*\d+", f"num_cores = {cores}", text)
        if patched == text:
            return settings.resolve()
        copy = (self.out / "settings.toml").resolve()
        copy.write_text(patched)
        return copy

    def write(self, name: str, array: np.ndarray):
        a = self.args[name]
        data = np.ascontiguousarray(array, dtype=DT[a["dtype"]]).reshape(a["shape"])
        assert data.nbytes == a["bytes"], (name, data.nbytes, a["bytes"])
        with open(self.out / "lp6.bin", "r+b") as f:
            f.seek(a["addr"])
            f.write(data.tobytes())

    def run(self, *, profile=False, timeline_max_events=12_000_000):
        if timeline_max_events <= 0:
            raise ValueError("timeline_max_events must be positive")
        t0 = time.time()
        for f in (self.dump_name, "commands.json"):
            (self.out / f).unlink(missing_ok=True)
        extra = ["--lp6-config", str(self.lp6_config)] if self.lp6_config else []
        if profile:
            extra += ["--sram-timing-out", "timing.json", "--timeline-out", "timeline.json",
                      "--timeline-merge-gap", "0", "--timeline-max-events", str(timeline_max_events),
                      "--vpu-profile-out", "vpu_profile.json"]
        res = subprocess.run([str(self.emulator), "--settings", str(self.settings), *extra, "--system-program", "system.json",
                              "--command-report", "commands.json", "--lp6-image", "lp6.bin", "--lp6-size", "16GiB",
                              "--log-level", "off"], cwd=self.out, capture_output=True, text=True,
                             env=self.execution_env())
        (self.out / "run.log").write_text(res.stdout + res.stderr)
        if res.returncode:
            print(res.stderr[-3000:])
            raise SystemExit(f"emulator failed ({res.returncode})")
        dump = (self.out / self.dump_name).read_bytes()
        outs = {}
        for name, o in self.outputs.items():
            outs[name] = np.frombuffer(dump[o["l1"]:o["l1"] + o["bytes"]], dtype=DT[o["dtype"]]).reshape(o["shape"]).copy()
        rep = json.loads((self.out / "commands.json").read_text())
        return outs, rep, time.time() - t0


def phase_times(rep):
    """Per-phase simulated time (us) from the command report of the single step program.

    Commands are named <phase>.<record> (a PIM record is named after its phase). Phases run in
    program order: a phase ends when its last record completes and starts when the previous one
    ended."""
    phases = {}
    for c in rep["commands"]:
        name = c["name"].split(".")[0]
        p = phases.setdefault(name, {"kind": "npu", "end": 0, "first": c["event_id"]})
        p["end"] = max(p["end"], c["end_cycle"])
        if c["kind"] == "pim":
            p["kind"] = "pim"
    rows, start = [], 0
    mhz = rep.get("npu_clock_mhz", 800)
    for name, p in sorted(phases.items(), key=lambda kv: kv[1]["first"]):
        rows.append((name, p["kind"], (p["end"] - start) / mhz))
        start = p["end"]
    return rows


def categorize(build: Path, times):
    """Sums phase times by role: PIM phases by the weights of their group (partition.json, same
    order), NPU phases by the PIM group they follow."""
    groups = json.loads((build / "partition.json").read_text())["groups"]
    def role(ws):
        w = ws[0]
        for key, name in (("q_proj", "PIM q/k/v"), ("o_proj", "PIM o"), ("gate_proj", "PIM gate/up"),
                          ("down_proj", "PIM down"), ("embed_tokens", "PIM lm_head"), ("lm_head", "PIM lm_head")):
            if key in w:
                return name
        return "PIM other"
    after = {"PIM q/k/v": "NPU dequant+qk-norm+RoPE+attention+quant", "PIM o": "NPU o dequant+residual+norm+quant",
             "PIM gate/up": "NPU dequant+SiLU*up+quant", "PIM down": "NPU down dequant+residual+norm+quant",
             "PIM lm_head": "NPU logits dequant"}
    out, gi, last = {}, 0, None
    if not any(kind == "pim" for _, kind, _ in times):
        return {"NPU (everything, GEMVs included)": sum(t for n, k, t in times if n != "collect")}
    for name, kind, t in times:
        if name == "collect":
            continue
        if kind == "pim":
            last = role(groups[gi]["weights"]) if gi < len(groups) else "PIM other"
            gi += 1
            key = last
        else:
            key = after.get(last, "NPU norm+quant (first layer)")
        out[key] = out.get(key, 0.0) + t
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("build", type=Path, help="npu-compile --pipeline=decode output directory")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--settings", type=Path, default=SETTINGS,
                    help="simulator settings (num_cores follows the program); default: four LPDDR6 channels")
    ap.add_argument("--lp6-config", type=Path, help="LP6 DRAMSim3 INI in place of the settings' one")
    a = ap.parse_args()

    dev = Device(a.build, a.settings, a.lp6_config)
    io = dev.io
    model = Path(io["model"])
    d = DecoderDims.from_config(model)
    d.layers = io["dims"]["layers"]
    M, P = io["tokens"], io["capacity"]
    assert M == 1, "the runtime drives M = 1 decode steps"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
    ids = tok(a.prompt, return_tensors="pt").input_ids[0].tolist()
    T = len(ids)
    assert T - 1 + a.steps <= P, "KV capacity too small"

    # Host prefill (FP32): KV cache of prompt[:-1]; the last prompt token is the first decode step.
    t0 = time.time()
    kt, vt = prefill_kv(model, d, ids[:-1], P)
    K, V = kt.numpy(), vt.numpy()
    print(f"host prefill of {T - 1} tokens ({d.layers} layers, FP32): {time.time() - t0:.1f}s", flush=True)

    store = SafetensorStore(model)
    embed = store.get("model.embed_tokens.weight")
    refs = None
    if a.check:
        refs = {"fp32": DecodeStep(d, 1, P), "w8a8": DecodeStep(d, 1, P, act_quant=True)}

    def reference_weights(quantized):
        # DecodeStep consumes one layer at a time; do not hold the full BF16 and
        # INT8 checkpoints simultaneously when checking an 8B model.
        for name, _ in weight_names(d):
            w = store.get(name)
            yield quantize_rows(w) if quantized and w.dim() == 2 else w

    token, past = ids[-1], T - 1
    generated = []
    for step in range(a.steps):
        x = embed[token].float().numpy()[None]
        cos, sin = rope_tables(d, [past])
        mask = step_mask(1, P, past).numpy()
        dev.write("x", x); dev.write("cos", cos.numpy()); dev.write("sin", sin.numpy()); dev.write("mask", mask)
        for i in range(d.layers):
            dev.write(f"past_k.{i}", K[i]); dev.write(f"past_v.{i}", V[i])
        outs, rep, wall = dev.run()
        np.savez(a.build / f"step{step}.npz", x=x, cos=cos.numpy(), sin=sin.numpy(), mask=mask, K=K, V=V,
                 **{k: v for k, v in outs.items()})
        logits = outs["logits"][0].astype(np.float32)
        nxt = int(logits.argmax())
        times = phase_times(rep)
        npu = sum(t for n, k, t in times if k == "npu" and n != "collect")
        pim = sum(t for n, k, t in times if k == "pim")
        print(f"step {step}: pos {past} token {token!r} -> {nxt} {tok.decode([nxt])!r}   "
              f"simulated {npu + pim:.0f} us (NPU {npu:.0f} + PIM {pim:.0f})   wall {wall:.0f}s", flush=True)
        if refs:
            xin = torch.tensor(x)
            cm = torch.tensor(mask)
            for name, stepm in refs.items():
                with torch.no_grad():
                    rl, rk, rv = stepm(xin, cos, sin, cm, [torch.tensor(K[i]) for i in range(d.layers)],
                                       [torch.tensor(V[i]) for i in range(d.layers)], reference_weights(name == "w8a8"))
                rl = rl[0].numpy()
                rel = np.linalg.norm(logits - rl) / np.linalg.norm(rl)
                kerr = max(np.abs(outs[f"new_k.{i}"][:, 0].astype(np.float32) - rk[i][:, 0].numpy()).max()
                           for i in range(d.layers))
                top5 = np.argsort(-rl)[:5]
                def rel_err(got, ref):
                    got = got.astype(np.float32); ref = ref.numpy()
                    return np.linalg.norm(got - ref) / np.linalg.norm(ref)
                kv = "  ".join(f"L{i} K {rel_err(outs[f'new_k.{i}'], rk[i]):.2e} V {rel_err(outs[f'new_v.{i}'], rv[i]):.2e}"
                               for i in range(min(d.layers, 3)))
                print(f"   vs {name}: new K/V rel_l2 {kv}")
                print(f"   vs {name}: logits rel_l2 {rel:.3e}  max|dlogit| {np.abs(logits - rl).max():.3f}  "
                      f"top-1 {int(rl.argmax())} ({'match' if int(rl.argmax()) == nxt else 'DIFF'})  "
                      f"device top-1 rank in ref top-5: {list(top5).index(nxt) if nxt in top5 else '-'}  "
                      f"new K max abs err {kerr:.3e}", flush=True)
        for i in range(d.layers):
            K[i, :, past] = outs[f"new_k.{i}"][:, 0].astype(np.float32)
            V[i, :, past] = outs[f"new_v.{i}"][:, 0].astype(np.float32)
        generated.append(nxt)
        token, past = nxt, past + 1
    print("prompt   :", repr(a.prompt))
    print("generated:", repr(tok.decode(generated)))
    (a.build / "phase_times.json").write_text(json.dumps(phase_times(rep), indent=1))
    cat = categorize(a.build, phase_times(rep))
    total = sum(cat.values())
    print(f"last step, simulated time by role (sum over layers), total {total:.0f} us:")
    for k, v in sorted(cat.items(), key=lambda kv: -kv[1]):
        print(f"   {k:45s} {v:9.0f} us  {100 * v / total:5.1f}%")


if __name__ == "__main__":
    main()
