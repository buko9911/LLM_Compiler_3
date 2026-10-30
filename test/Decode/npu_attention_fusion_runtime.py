#!/usr/bin/env python3
"""Native decode/Ramulator A/B for masked GQA attention producer fusion.

Uses tiny synthetic tensors, not pretrained model throughput. The NumPy value
reference follows the decode FP16 storage boundaries; all reported times come
from the existing cycle simulator. Fresh outputs and source hashes are required.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from safetensors.numpy import save_file

from npu_attention_fusion import attention_source


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command, cwd, log, env=None):
    with log.open("w") as stream:
        result = subprocess.run(list(map(str,command)), cwd=cwd, stdout=stream,
                                stderr=subprocess.STDOUT, env=env)
    if result.returncode:
        raise RuntimeError(f"Failed: {command}\n{log.read_text()[-6000:]}")


def value_reference(inputs, heads, groups, tokens, keys, depth):
    q,k,v = (inputs[name].astype(np.float32) for name in ("q","k","v"))
    scores = np.matmul(q,k).astype(np.float16).reshape(heads,groups,tokens,keys)
    scores = (scores.astype(np.float32)+inputs["mask"].astype(np.float32)[None,None,:,:]).astype(np.float16)
    shifted = scores.astype(np.float32)-scores.astype(np.float32).max(axis=-1,keepdims=True)
    exp = np.exp(shifted).astype(np.float16)
    probability = (exp.astype(np.float32)/exp.astype(np.float32).sum(axis=-1,keepdims=True)).astype(np.float16)
    return np.matmul(probability.reshape(heads,groups*tokens,keys).astype(np.float32),v).astype(np.float16)


def main():
    root = Path(__file__).resolve().parents[2]
    workspace = root.parent.parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-pipeline", action="store_true", help="Compare blocking, async-only, and common tiled input pipeline")
    parser.add_argument("--input-tile-columns", type=int, default=128)
    parser.add_argument("--input-scheduling", action="store_true", help="Compare common asynchronous input scheduling against blocking and immediate-wait controls")
    parser.add_argument("--compiler",type=Path,default=root/"build/bin/npu-compile")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--emulator",type=Path,default=workspace/"core/simulator/npu/transactional_emulator/target/release/transactional_emulator")
    parser.add_argument("--settings",type=Path,default=workspace/"research/checkpoint_race/hardware/kernel_probe/settings_4core.toml")
    parser.add_argument("--lp6-library",type=Path,default=workspace/"core/ramulator2/liblp6_ramulator.so")
    args=parser.parse_args()
    if args.input_pipeline and args.input_scheduling:
        parser.error("Choose input pipeline or input scheduling experiment")
    output=args.output.resolve(); output.mkdir(parents=True,exist_ok=False)
    model=output/"model"; model.mkdir()
    save_file({"unused":np.zeros((1,),dtype=np.float16)},model/"model.safetensors")
    sources=[args.compiler.resolve(),args.emulator.resolve(),args.settings.resolve(),args.lp6_library.resolve(),
             args.lp6_library.resolve().parent/"libramulator.so",Path(__file__).resolve(),
             Path(__file__).with_name("npu_attention_fusion.py").resolve()]
    provenance={str(path):digest(path) for path in sources if path.exists()}
    (output/"provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    cases=[("m1_causal",8,4,1,9,8,False),
           ("m8_causal_and_finite_masked_row",8,4,8,40,32,True),
           ("odd_keys_nonpower_heads",6,2,3,7,8,True)]
    if args.input_pipeline:
        cases += [("pipeline_column_tail",8,2,3,9,65,True),
                  ("pipeline_many_heads",12,2,2,17,128,False)]
    summary={"kind":"native_cycle_simulator_functional_microtests","cases":{}}
    for name,h,g,m,s,d,masked in cases:
        case=output/name;case.mkdir()
        source=case/"step.mlir"
        text=attention_source(heads=h,groups=g,tokens=m,keys=s,depth=d,with_mask=True)
        text=text.replace("@phase(","@decode_step(").replace('attributes {hyb.phase = "npu"}',"")
        if args.input_pipeline:
            # Model a KV append rather than a phase argument read: two disjoint
            # slices assemble V in shared L2 before the head-local consumer.
            # Values are unchanged; all three modes pay this same preparation.
            vt=f"tensor<{h}x{s}x{d}xf16>"
            head=f"tensor<{h}x{s-1}x{d}xf16>"
            tail=f"tensor<{h}x1x{d}xf16>"
            assembly=f"""%v_empty = tensor.empty() : {vt}
    %v_head = tensor.extract_slice %v[0,0,0][{h},{s-1},{d}][1,1,1] : {vt} to {head}
    %v_tail = tensor.extract_slice %v[0,{s-1},0][{h},1,{d}][1,1,1] : {vt} to {tail}
    %v_prefix = tensor.insert_slice %v_head into %v_empty[0,0,0][{h},{s-1},{d}][1,1,1] : {head} into {vt}
    %v_joined = tensor.insert_slice %v_tail into %v_prefix[0,{s-1},0][{h},1,{d}][1,1,1] : {tail} into {vt}
    """
            marker="%zero = arith.constant"
            assert marker in text
            text=text.replace(marker,assembly+marker,1)
            text=text.replace(",%v:",",%v_joined:").replace(",%v :",",%v_joined :").replace(", %v :",", %v_joined :")

        source.write_text(text)
        manifest=case/"manifest.json"
        manifest.write_text(json.dumps(dict(model=str(model),tokens=m,capacity=s-m,mask_min=-30000.,
            arguments=[dict(kind="input",name=n) for n in ("q","k","v","mask")],
            results=[dict(kind="output",name="y")]),indent=2))
        rng=np.random.default_rng(20261029)
        inputs={"q":rng.integers(-4,5,size=(h,g*m,d)).astype(np.float16)/8,
                "k":rng.integers(-4,5,size=(h,d,s)).astype(np.float16)/8,
                "v":rng.integers(-8,9,size=(h,s,d)).astype(np.float16)/8}
        mask=np.where(np.arange(s)[None,:]<=s-m+np.arange(m)[:,None],0,-30000).astype(np.float16)
        if masked:mask[-1,:]=-30000
        inputs["mask"]=mask
        np.savez(case/"inputs.npz",**inputs)
        reference=value_reference(inputs,h,g,m,s,d);np.save(case/"numpy_reference.npy",reference)
        runs={}
        previous=None
        modes = [("blocking",False,True,False), ("immediate",True,False,False), ("overlap",True,True,False)] if args.input_scheduling else [("off",False,True,False),("on",False,True,False)]
        if args.input_pipeline:
            modes = [("blocking",False,True,False), ("async",True,True,False), ("pipeline",True,True,True)]
        for mode,async_input,input_overlap,pipeline in modes:
            enabled = args.input_pipeline or args.input_scheduling or mode == "on"
            build=case/mode
            spec=json.loads((root/"configs/target.json").read_text())
            spec["npu"].update(attention_fusion=enabled,overlap_dequant_gemv=False,
                               l2_staging_bytes_per_core=0, async_input_loads=async_input,
                               async_input_load_overlap=input_overlap, input_tile_pipeline=pipeline,
                               input_tile_columns=args.input_tile_columns)
            target=case/f"{mode}.json";target.write_text(json.dumps(spec,indent=2))
            command=[args.compiler.resolve(),"--pipeline=decode",source,"--manifest",manifest,
                     "--spec",target,"--force=npu","-o",build]
            run(command,workspace,case/f"{mode}.compile.log")
            io=json.loads((build/"io.json").read_text())
            with (build/"lp6.bin").open("r+b") as image:
                for arg in io["args"]:
                    array=np.asarray(inputs[arg["name"]],dtype=np.float16).reshape(arg["shape"])
                    image.seek(arg["addr"]);image.write(array.tobytes())
            simulation=[args.emulator.resolve(),"--settings",args.settings.resolve(),
                "--lp6-config",workspace/"core/simulator/dram-pim/configs/LPDDR6_PIM_Mobile_4CH.ini",
                "--lp6-backend","ramulator","--lp6-library",args.lp6_library.resolve(),
                "--system-program","system.json","--lp6-image","lp6.bin","--lp6-size","16GiB",
                "--command-report","commands.json","--sram-timing-out","timing.json",
                "--timeline-out","timeline.json","--timeline-merge-gap","0",
                "--timeline-max-events","1000000","--log-level","off"]
            hashes={str(path):digest(path) for path in (source,manifest,target,case/"inputs.npz",build/"program.bin",build/"lp6.bin")}
            (build/"invocation.json").write_text(json.dumps(dict(command=list(map(str,simulation)),source_sha256=hashes),indent=2))
            begin=time.monotonic()
            run(simulation,build,build/"run.log",dict(os.environ,LP6_DRAMSIM3_ROOT=str(workspace/"core/simulator/dram-pim")))
            wall=time.monotonic()-begin
            output_info=io["outputs"][0]
            raw=(build/"sram_dump.core0.bin").read_bytes()
            array=np.frombuffer(raw[output_info["l1"]:output_info["l1"]+output_info["bytes"]],dtype=np.float16).reshape(output_info["shape"])
            np.save(build/"output.npy",array)
            assert np.isfinite(array).all(),name
            exact=previous is None or array.tobytes()==previous.tobytes()
            maximum=float(np.max(np.abs(array.astype(np.float32)-reference.astype(np.float32))))
            # The independent reference checks values, while byte comparison
            # checks the unchanged compiler numerical contract exactly.
            assert maximum<=0.002,(name,mode,maximum)
            assert exact,(name,"off/on numerical mismatch")
            timing=json.loads((build/"timing.json").read_text())
            runs[mode]=dict(total_us=timing["total_simulated_cycles"]/timing["npu_clock_mhz"],
                wall_seconds=wall,byte_exact_with_off=exact,max_abs_numpy_error=maximum,
                argmax_equal_to_off=previous is None or bool(np.array_equal(array.argmax(-1),previous.argmax(-1))),
                elements=int(array.size),lp6_read_bytes=timing["lp6_bytes_read"],lp6_write_bytes=timing["lp6_bytes_written"],
                phase_metadata=io["phases"], input_pipeline_groups=sum(map(int,__import__("re").findall(r"hyb.input_pipeline_groups = (\d+)",(build/"compiled.mlir").read_text()))))
            previous=array.copy()
        summary["cases"][name]=runs
        (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
        print(name,json.dumps({mode:{k:v for k,v in result.items() if k!="phase_metadata"} for mode,result in runs.items()}),flush=True)
    if args.input_pipeline:
        assert any(case["pipeline"]["input_pipeline_groups"] for case in summary["cases"].values()), "No pipeline exercised"
    for path in sources:
        if path.exists():assert digest(path)==provenance[str(path)],f"Changed while running: {path}"


if __name__=="__main__":
    main()
