#!/usr/bin/env python3
"""One-copy Up-projection experiment using executable ISA and timed NPU reduction.

W8A8 kernel scope: input INT8 in DRAM -> output INT32 in shared L2. Quantization,
scaling and the remaining FFN are intentionally outside *both* measured paths.
The simulator's optional read-view model is a hardware research configuration.
It is not enabled by npu-compile's production pipeline.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass, asdict
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LP6 = ROOT.parent
SIM = LP6 / "IESLAB_NPU_Simulator/transactional_emulator"
sys.path.insert(0, str(SIM / "tools"))
import isa_v2 as isa

X_BASE, W_BASE, O_BASE = 4096, 4 << 20, 768 << 20
X_L2, W_L2 = 0x1E0000, 2 << 20
WEIGHT_ROW = 256


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2) + "\n")


@dataclass
class Layout:
    n: int
    k: int
    tokens: int = 1
    tile_n: int = 256
    k_splits: int = 1
    n_order: str = "interleaved"
    weight_row: int = WEIGHT_ROW
    pingpong: bool = False

    def __post_init__(self):
        assert self.k_splits in (1,2,4,8) and self.tile_n in (64,128,256,512)
        assert self.n>0 and self.n%(self.groups*32)==0
        assert self.k>0 and self.k%(256*self.k_splits)==0 and 1<=self.tokens<=32

    @property
    def groups(self): return 8 // self.k_splits
    @property
    def accs(self): return self.tile_n // 32
    @property
    def streams(self): return -(-self.n // (self.groups * self.tile_n))
    @property
    def kp(self): return self.k // self.k_splits
    @property
    def per_pu(self): return self.n // (8 * self.groups)
    @property
    def weight_tiles(self): return self.streams * self.accs * self.kp // 8
    @property
    def act_row(self): return self.weight_row + -(-self.weight_tiles // 64)
    @property
    def out_row(self): return self.act_row + -(-(self.tokens * self.kp // 8) // 64)
    @property
    def image_size(self):
        return (self.out_row + -(-(self.tokens * 8 * self.streams * self.accs // 2) // 64)) * 262144

    def packet(self, row, tile, pu, sc, odd=0):
        bank = 2 * pu + odd
        return (row + tile // 64) * 262144 + (bank % 4) * 65536 + (tile % 64) * 1024 + (bank // 4) * 256 + sc * 32

    def logical_rows(self, sc, pu, stream, acc):
        local = pu * self.per_pu + (stream * self.accs + acc) * 4 + np.arange(4)
        valid = (stream * self.accs + acc) * 4 < self.per_pu
        if self.n_order == "joint":
            a = stream*self.accs+acc
            if self.groups >= 2:
                q = a*8*self.groups+(pu%2)*4*self.groups+((sc%self.groups)//2)*8+(pu//2)*2+sc%2
            else:
                q = a*8+(pu%2)*4+pu//2
            rows = q*4+np.arange(4)
        elif self.n_order == "interleaved":
            rows = (local // 4 * self.groups + sc % self.groups) * 4 + local % 4
        else:
            rows = sc % self.groups * (self.n // self.groups) + local
        return rows, valid

    def view(self, kind):
        return dict(logical_base=W_BASE if kind == "weight" else O_BASE, kind=kind, **asdict(self))

    def descriptor(self, *, buffer_bytes=65536, window_bytes=32768, reorder_bytes_per_cycle=128, fair_windows=False):
        cfg=dict(schema="lp6.experimental.dual_view.v1", buffer_bytes=buffer_bytes, window_bytes=window_bytes,
                    remap_latency_cycles=2, remap_blocks_per_cycle=4,
                    reorder_bytes_per_cycle=reorder_bytes_per_cycle,
                    views=[self.view("weight"), self.view("output")])
        if fair_windows: cfg["fair_windows"]=True
        return cfg

    def pack(self, path, w, x):
        # The only resident weight copy is here. W_BASE's NPU logical range is zero.
        assert W_BASE + w.size <= self.weight_row * 262144
        with Path(path).open("wb") as f: f.truncate(self.image_size)
        image = np.memmap(path, mode="r+", dtype=np.uint8, shape=(self.image_size,))
        image[X_BASE:X_BASE+x.size] = x.view(np.uint8).ravel()
        for sc in range(8):
            split = sc // self.groups
            for pu in range(8):
                for stream in range(self.streams):
                    for acc in range(self.accs):
                        rows, valid = self.logical_rows(sc, pu, stream, acc)
                        if not valid: continue
                        block = w[rows, split*self.kp:(split+1)*self.kp]
                        packets = block.reshape(4, self.kp//256, 32, 8).transpose(1, 2, 0, 3).reshape(-1, 32)
                        tiles = stream * self.accs * (self.kp//8) + np.arange(self.kp//8)//32 * self.accs * 32 + acc * 32 + np.arange(self.kp//8)%32
                        pa = self.packet(self.weight_row, tiles, pu, sc, (tiles//64%2 if self.pingpong else 0))
                        image[pa[:, None]+np.arange(32)] = packets.view(np.uint8)
        assert not np.any(image[W_BASE:W_BASE+w.size]), "duplicate weight in logical alias region"
        image.flush()
        del image


class Program:
    """Small final-ISA emitter, using the existing checked ISA encoder."""
    def __init__(self):
        self.words = [0]*5
        self.symbols = []
        self.core_words = 0

    def add(self, name, fn):
        ordinal = len(self.symbols)
        self.words += fn(ordinal)
        self.symbols.append(dict(id=ordinal, name=f"{name}.{ordinal}"))
        return ordinal

    def dma(self, name, deps, lp6, l2, size, *, load=True, rows=1, lp6_stride=None, l2_stride=None):
        return self.add(name, lambda i: isa.gdma_record(load, i, deps, lp6, l2, size, rows, lp6_stride, l2_stride))

    def core(self, name, deps, core, words):
        self.core_words += len(words)
        return self.add(name, lambda i: isa.core_begin_record(i, deps, len(words), core, 4 << 20) + words)

    def barrier(self, name, deps):
        while len(deps) > 255:
            deps = [self.add(name+"_join", lambda i, ds=deps[j:j+255]: isa.barrier_record(i, ds)) for j in range(0,len(deps),255)]
        return self.add(name, lambda i: isa.barrier_record(i, deps))

    def finish(self, path, kernel=None):
        self.words.append(isa.PROGRAM_END)
        self.words[:5] = [isa.PROGRAM_MAGIC, isa.PROGRAM_VERSION, len(self.words), len(self.symbols), self.core_words]
        path.mkdir(parents=True, exist_ok=True)
        binary = struct.pack(f"<{len(self.words)}I", *self.words)
        # Structural validation is independent of the emitter's event numbering.
        isa.decode_program(binary)
        (path/"program.bin").write_bytes(binary)
        manifest = dict(schema="plena.v2.unified_program.isa_v2.0", program="program.bin",
                        program_word_count=len(self.words), command_count=len(self.symbols),
                        core_instruction_count=self.core_words, command_symbols=self.symbols,
                        scheduling_policy="compiler_static", logical_to_physical=[0,1,2,3],
                        l2_bytes_required=8 << 20, l2_regions=[])
        if kernel: manifest["pim_kernel"] = kernel
        save(path/"system.json", manifest)


def constant(gp, value): return isa.load_constant(gp, int(value))


def ldma_strided(op, l1, l2, row_bytes):
    return constant(14,l1)+constant(15,l2)+[isa.c_set(4,row_bytes),isa.ldma_r(op,14,15)]


def npu_program(path, m, n, k, *, batch_tiles=4, native_order="linear"):
    assert n % 128 == 0 and 1 <= m <= 32
    assert batch_tiles in (1,2,4)
    assert native_order in ("linear","core_interleaved")
    assert k*32*8 <= 3<<19, "weight ring exceeds the experiment's per-core L2 window"
    assert m*k<=0x20000 and m*n*4<=X_L2, "input/output exceeds the experiment's L2 reservation"
    p = Program()
    xdep = p.dma("input.load", [], X_BASE, X_L2, m*k)
    tile_bytes, count = k*32, n//128
    history = [[] for _ in range(4)]
    prev = [xdep]*4
    for batch in range(-(-count//batch_tiles)):
        staged = []
        for core in range(4):
            items = []
            for j in range(batch*batch_tiles, min(count,batch*batch_tiles+batch_tiles)):
                n0 = (core*count+j)*32
                slot = W_L2 + core*(3<<19) + (j%8)*tile_bytes
                reuse_batches = 8//batch_tiles
                deps = [history[core][batch-reuse_batches]] if batch >= reuse_batches else []
                block = core*count+j if native_order=="linear" else j*4+core
                d = p.dma("weight.load", deps, W_BASE+block*tile_bytes, slot, tile_bytes)
                items.append((n0, slot, d))
            staged.append(items)
        for core, items in enumerate(staged):
            body = isa.ldma_i(0, 0, X_L2, m*k)
            body += isa.ldma_i(0, 0x40000, items[0][1], tile_bytes)
            for t, (n0, slot, _) in enumerate(items):
                if t: body += [isa.c_sync(4,1)]
                body += constant(0, 0x40000 + (t%2)*tile_bytes) + isa.m_ld(0,0,32,k)
                body += constant(1, 0) + isa.m_ld(2,1,m,k)
                body += [isa.m_mma(False, False, -(-k//32))]
                if t+1 < len(items):
                    body += isa.ldma_i(10, 0x40000+((t+1)%2)*tile_bytes, items[t+1][1], tile_bytes, event=1)
                body += constant(2,0xC0000) + isa.m_wb(2,2,m,32)
                if m == 1:
                    body += isa.ldma_i(1,0xC0000,n0*4,128)
                else:
                    body += [isa.c_set(0,m),isa.c_set(1,128),isa.c_set(2,n*4)]
                    body += ldma_strided(5,0xC0000,n0*4,128)
            event = p.core("npu.compute", [prev[core]]+[x[2] for x in items], core, body)
            history[core].append(event)
            prev[core] = event
    p.barrier("done",prev)
    p.finish(path)


def pim_program(path, layout, scatter_mode="wide"):
    l = layout
    assert l.tokens*l.k_splits*l.n*4<=8<<20, "partial outputs exceed physical L2"
    p = Program()
    xdep = p.dma("input.load", [], X_BASE, X_L2, l.tokens*l.k)
    writes = []
    # Timed L1 replication and wide stores, matching the compiler's all-SC
    # activation strategy. The K shards may hold different input slices.
    # No host-populated PIM input registers and no free activation broadcast.
    previous = xdep
    for token in range(l.tokens):
        if scatter_mode=="strided":
            for sc in range(8):
                split=sc//l.groups
                for pu in range(8):
                    for first in range(0,l.kp//32,64):
                        pk=token*l.kp//32+first
                        rows=min(l.kp//32-first,64-pk%64)
                        writes.append(p.dma("activation.scatter",[xdep],l.packet(l.act_row,pk,pu,sc),
                            X_L2+token*l.k+split*l.kp+first*32,32,load=False,rows=rows,lp6_stride=1024,l2_stride=32))
            continue
        body=isa.ldma_i(0,0,X_L2+token*l.k,l.k)
        body += [isa.c_set(0,l.kp//32),isa.c_set(1,32),isa.c_set(2,1024)]
        for bg in range(4):
            for sc in range(8):
                body += ldma_strided(9,0x40000+bg*256+sc*32,(sc//l.groups)*l.kp,32)
        body += isa.ldma_i(1,0x40000,0x100000,l.kp*32)
        packed=p.core("activation.pack",[previous],0,body)
        stores=[]
        first=0
        while first<l.kp//32:
            pk=token*l.kp//32+first
            count=min(l.kp//32-first,64-pk%64)
            for pu in (0,1):
                stores.append(p.dma("activation.scatter",[packed],l.packet(l.act_row,pk,pu,0),
                                    0x100000+first*1024,count*1024,load=False))
            first+=count
        previous=p.barrier("activation.token_ready",stores)
        writes.append(previous)
    adep = p.barrier("activation.ready",writes)
    cmds = []
    for token in range(l.tokens):
        for sc in range(8):
            emit = lambda op, **kw: isa.pim_primitive(op,sc,**kw)
            for stream in range(l.streams):
                cmds += [emit(isa.PIM_CLEAR,accumulator=i) for i in range(l.accs)]
                for sweep in range(l.kp//256):
                    a = token*l.kp//8+sweep*32
                    cmds += [emit(isa.PIM_LOAD_ACTIVATION,activation=a+i) for i in range(0,32,4)]
                    w = stream*l.accs*(l.kp//8)+sweep*l.accs*32
                    cmds += [emit(isa.PIM_GEMV_STREAM,weight=w,activation=a,accumulator=0,inputs=32,outputs=l.accs,
                                  weight_input_stride=1,weight_output_stride=32,activation_input_stride=1,accumulator_output_stride=1)]
                obase = ((token*8+sc)*l.streams+stream)*(l.accs//2)
                cmds += [emit(isa.PIM_STORE,accumulator=2*i,output=obase+i) for i in range(l.accs//2)]
    pd = p.add("pim.compute", lambda i: isa.pim_record(i,[adep],cmds))
    reads = []
    # Each K shard is returned in full INT32 precision; the reassembly hardware
    # changes byte order only. It performs no arithmetic reduction.
    for token in range(l.tokens):
        for split in range(l.k_splits):
            off = (token*l.k_splits+split)*l.n*4
            reads.append(p.dma("output.gather",[pd],O_BASE+off,off,l.n*4))
    ready = p.barrier("output.ready",reads)
    if l.k_splits > 1:
        done = []
        share = l.n//4
        for core in range(4):
            body = []
            for token in range(l.tokens):
                source = token*l.k_splits*l.n*4+core*share*4
                body += [isa.c_set(0,l.k_splits),isa.c_set(1,l.n*4),isa.c_set(2,share*4)]
                body += ldma_strided(4,0,source,share*4)
                for start in range(0,share,16):
                    length=min(16,share-start)  # 512-bit physical vector register
                    body += [isa.c_set(3,length)] + constant(0,start*4)
                    body += [isa.v_mem(False,isa.I32,0,0,0)]
                    for split in range(1,l.k_splits):
                        body += constant(1,(split*share+start)*4)
                        body += [isa.v_mem(False,isa.I32,1,1,0),isa.v_op(0,isa.I32,0,0,1)]
                    body += [isa.v_mem(True,isa.I32,0,0,0)]
                body += isa.ldma_i(1,0,source,share*4)
            done.append(p.core("npu.reduce",[ready],core,body))
        p.barrier("done",done)
    kernel=dict(channel=0,rank=0,row=l.weight_row,start_column=0,bank_set=1,weight_bits=8,activation_bits=8,
                scale_bits=0,input_register_count=8,output_register_count=8,subchannels=8,
                weight_layout="bankset-row-alternating-v1" if l.pingpong else "even-v1",
                bankset_pingpong_lead=12 if l.pingpong else 0)
    p.finish(path,kernel)


def execute(path, image, settings, emulator, expected, layout=None, *, engine, k, timeout=600,
            window_bytes=32768, buffer_bytes=65536, reorder_bytes_per_cycle=128, fair_windows=False,
            library=None):
    emulator_hash=sha(emulator)
    env = dict(os.environ,LD_PRELOAD=str(library or LP6/"lp6-pim-main/libdramsim3.so"),
               LP6_DRAMSIM3_ROOT=str(LP6/"lp6-pim-main"),LP6PIM_BANKSET_PINGPONG="12" if layout and layout.pingpong else "0")
    cmd=[str(emulator),"--settings",str(settings),"--system-program","system.json","--lp6-image",str(image),
         "--lp6-config",str(LP6/"lp6-pim-main/configs/LPDDR6_PIM_Mobile_4CH.ini"),"--lp6-size","16GiB",
         "--command-report","commands.json","--sram-timing-out","timing.json","--log-level","off"]
    if layout:
        save(path/"read_view.json",layout.descriptor(window_bytes=window_bytes,buffer_bytes=buffer_bytes,
                                                  reorder_bytes_per_cycle=reorder_bytes_per_cycle,fair_windows=fair_windows))
        cmd += ["--experimental-read-view","read_view.json"]
    save(path/"invocation.json",dict(argv=cmd,env={k:env[k] for k in ("LD_PRELOAD","LP6_DRAMSIM3_ROOT","LP6PIM_BANKSET_PINGPONG")}))
    start=time.monotonic()
    with (path/"run.log").open("w") as log:
        r=subprocess.run(cmd,cwd=path,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
    if r.returncode: raise RuntimeError((path/"run.log").read_text()[-6000:])
    dump=np.fromfile(path/"l2_sram_dump.bin",dtype=np.int32)
    m,n=expected.shape
    actual = (np.stack([dump[t*layout.k_splits*n:t*layout.k_splits*n+n] for t in range(m)])
              if layout and engine=="pim" else dump[:m*n].reshape(m,n))
    if not np.array_equal(actual,expected):
        bad=np.argwhere(actual!=expected)[0]
        raise AssertionError(f"{path}: {bad}: got {actual[tuple(bad)]}, expected {expected[tuple(bad)]}")
    report=json.loads((path/"commands.json").read_text())
    timing=json.loads((path/"timing.json").read_text())
    commands=report["commands"]
    end=max(c["end_cycle"] for c in commands)
    mhz=report["npu_clock_mhz"]
    def last(prefix): return max([c["end_cycle"] for c in commands if c["name"].startswith(prefix)] or [0])
    phases={}
    if engine=="pim":
        act=last("activation.ready"); pim=last("pim.compute"); gather=last("output.ready")
        phases=dict(input_scatter_us=act/mhz,pim_us=(pim-act)/mhz,gather_us=(gather-pim)/mhz,reduce_us=(end-gather)/mhz)
    result=dict(engine=engine,tokens=m,n=n,k=k,
                latency_us=end/mhz,phase_us=phases,all_outputs_int32_exact=True,
                output_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
                lp6_bytes_read=timing["lp6_bytes_read"],lp6_bytes_written=timing["lp6_bytes_written"],
                wall_seconds=time.monotonic()-start,program_sha256=sha(path/"program.bin"),
                emulator_sha256=emulator_hash,
                read_view=layout.descriptor(window_bytes=window_bytes,buffer_bytes=buffer_bytes,
                    reorder_bytes_per_cycle=reorder_bytes_per_cycle,fair_windows=fair_windows) if layout else None)
    save(path/"result.json",result)
    for dumpfile in path.glob("*sram_dump*.bin"): dumpfile.unlink()
    print(path.name,round(result["latency_us"],3),"us, INT32 PASS",flush=True)
    return result


def load_weight(model, output, synthetic=False, n=14336, k=4096):
    if synthetic:
        w=np.random.default_rng(20261002).integers(-127,128,(n,k),dtype=np.int8)
        meta=dict(source="deterministic synthetic INT8",shape=[n,k])
    else:
        import torch
        from safetensors import safe_open
        torch.set_num_threads(4)
        name="model.layers.0.mlp.up_proj.weight"
        idx=json.loads((model/"model.safetensors.index.json").read_text())
        shard=model/idx["weight_map"][name]
        with safe_open(shard,framework="pt") as f: raw=f.get_tensor(name).float().numpy()
        scales=np.maximum(np.max(np.abs(raw),axis=1)/np.float32(127),np.float32(1e-12))
        w=np.clip(np.rint(raw/scales[:,None]),-127,127).astype(np.int8)
        meta=dict(source=str(shard),tensor=name,shape=list(w.shape),quantization="symmetric INT8 per output row, round-to-nearest-even",
                  original_f32_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),scale_sha256=hashlib.sha256(scales.tobytes()).hexdigest())
    meta["int8_sha256"]=hashlib.sha256(w.tobytes()).hexdigest()
    save(output/"weight_provenance.json",meta)
    return w


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--model",type=Path,default=Path("/home/jongjip/models/llama31-8b"))
    ap.add_argument("--emulator",type=Path,default=SIM/"target/release/transactional_emulator")
    ap.add_argument("--synthetic",action="store_true")
    ap.add_argument("--n",type=int,default=14336); ap.add_argument("--k",type=int,default=4096)
    ap.add_argument("--tokens",type=int,nargs="+",default=[1])
    ap.add_argument("--cases",nargs="+",default=["npu_native","grouped256","n256","n128","n64","n512","k2","k4","k8"])
    ap.add_argument("--keep-images",action="store_true")
    ap.add_argument("--engines",nargs="+",choices=["pim","npu"],default=["pim","npu"])
    ap.add_argument("--scatter-mode",choices=["wide","strided"],default="wide")
    ap.add_argument("--fair-windows",action="store_true")
    ap.add_argument("--window-bytes",type=int,default=32768)
    ap.add_argument("--buffer-bytes",type=int,default=65536)
    ap.add_argument("--reorder-bytes-per-cycle",type=int,default=128)
    args=ap.parse_args(); out=args.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    if (out/"summary.json").exists(): raise SystemExit("choose a fresh output directory")
    settings=out/"settings.toml"
    settings.write_text(re.sub(r"(?m)^num_cores\s*=\s*\d+","num_cores = 4",(SIM.parent/"plena_settings.toml").read_text()))
    save(out/"provenance.json",dict(emulator=str(args.emulator),emulator_sha256=sha(args.emulator),
        library_sha256=sha(LP6/"lp6-pim-main/libdramsim3.so"),settings_sha256=sha(settings),
        dram_config_sha256=sha(LP6/"lp6-pim-main/configs/LPDDR6_PIM_Mobile_4CH.ini"),scope="INT8 input in LP6 to INT32 output in L2",
        codegen_sha256=sha(__file__),argv=sys.argv))
    w=load_weight(args.model,out,args.synthetic,args.n,args.k)
    n,k=w.shape
    result={}
    for m in args.tokens:
        x=np.random.default_rng(4096).integers(-127,128,(m,k),dtype=np.int8)
        expected=x.astype(np.int32) @ w.astype(np.int32).T
        for name in args.cases:
            key=f"m{m}_{name}"
            base=out/key; base.mkdir(exist_ok=False)
            image=base/"lp6.bin"
            kwargs=dict(k=k,window_bytes=args.window_bytes,buffer_bytes=args.buffer_bytes,reorder_bytes_per_cycle=args.reorder_bytes_per_cycle,
                        fair_windows=args.fair_windows)
            if name=="npu_native":
                with image.open("wb") as f:
                    f.seek(X_BASE); f.write(x.tobytes())
                    f.seek(W_BASE); f.write(w.reshape(n//32,32,k).transpose(0,2,1).copy().tobytes())
                npu_program(base,m,n,k)
                result[key]=execute(base,image,settings,args.emulator,expected,engine="npu",**kwargs)
            else:
                split=int(name[1:].removesuffix("pp")) if name.startswith("k") else 1
                tile=int(name[2:]) if name.startswith("pp") else (256 if name.startswith(("grouped","k","joint")) else int(name[1:]))
                order="joint" if name.startswith("joint") else ("grouped" if name.startswith("grouped") else "interleaved")
                l=Layout(n,k,m,tile,split,order,pingpong=name.startswith("pp") or name.endswith("pp"))
                l.pack(image,w,x)
                result[key]={"layout":asdict(l),"weight_payload_bytes":w.size,
                             "padded_weight_bytes":8*8*l.weight_tiles*32,"image_address_span":l.image_size}
                for engine in args.engines:
                    path=base/engine
                    path.mkdir()
                    (pim_program(path,l,args.scatter_mode) if engine=="pim" else npu_program(path,m,n,k))
                    result[key][engine]=execute(path,image,settings,args.emulator,expected,l,engine=engine,**kwargs)
            if not args.keep_images: image.unlink()
            save(out/"summary.partial.json",result)
    save(out/"summary.json",result)


if __name__=="__main__": main()
