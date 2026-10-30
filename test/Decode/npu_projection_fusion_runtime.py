#!/usr/bin/env python3
"""Native same-input projection and ring-reuse tests against integer NumPy.

Unequal Q/K/V widths, distinct row scales, multiple activation inputs, one core,
MMA row tails and small-ring fallback. No timing is synthesized.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from safetensors.numpy import save_file


def source(m, k, widths, different=False):
    types = [f'tensor<{m}x{n}xf32>' for n in widths]
    arguments = [f'%x:tensor<{m}x{k}xf32>', f'%x2:tensor<{m}x{k}xf32>']
    arguments += [f'%w{i}:tensor<{n}x{k}xf32>' for i,n in enumerate(widths)]
    lines = [f'module {{ func.func @decode_step({",".join(arguments)}) -> ({",".join(types)}) {{',
             '%zero = arith.constant 0.0 : f32']
    for i,n in enumerate(widths):
        x = '%x2' if different and i == 2 else '%x'
        lines += [
            f'%wt{i} = tensor.empty() : tensor<{k}x{n}xf32>',
            f'%t{i} = linalg.transpose ins(%w{i}:tensor<{n}x{k}xf32>) outs(%wt{i}:tensor<{k}x{n}xf32>) permutation=[1,0]',
            f'%e{i} = tensor.empty() : {types[i]}',
            f'%i{i} = linalg.fill ins(%zero:f32) outs(%e{i}:{types[i]}) -> {types[i]}',
            f'%y{i} = linalg.matmul ins({x},%t{i}:tensor<{m}x{k}xf32>,tensor<{k}x{n}xf32>) outs(%i{i}:{types[i]}) -> {types[i]}']
    lines += [f'return {",".join(f"%y{i}" for i in range(3))} : {",".join(types)}', '}}']
    return '\n'.join(lines)+'\n'


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[2]
    workspace = root.parents[1]
    sys.path.insert(0,str(root/'runtime'))
    from decode import Device
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--modes',nargs='+',choices=['off','fusion','release','both'],
                    default=['off','fusion','release','both'])
    a = ap.parse_args()
    out = a.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    compiler = root/'build/bin/npu-compile'
    settings = workspace/'research/checkpoint_race/hardware/kernel_probe/settings_4core.toml'
    library = workspace/'core/ramulator2/liblp6_ramulator.so'
    ini = workspace/'core/simulator/dram-pim/configs/LPDDR6_PIM_Mobile_4CH.ini'
    (out/'provenance.json').write_text(json.dumps({str(p):digest(p) for p in
        [Path(__file__),compiler,settings,library,ini]},indent=2)+'\n')
    results = {}
    cases = [('m1',1,64,[1152,384,384]),('m8',8,256,[768,256,256]),
             ('m33',33,64,[1152,384,384]),('different_input',8,64,[768,256,256]),
             ('one_core',8,64,[384,128,128]),('two_slots',8,64,[1152,384,384])]
    for name,m,k,widths in cases:
        case = out/name; (case/'model').mkdir(parents=True)
        different = name == 'different_input'
        rng = np.random.default_rng(1801)
        xq = rng.integers(-127,128,(m,k),dtype=np.int8); xq[:,0] = 127
        x2q = rng.integers(-127,128,(m,k),dtype=np.int8); x2q[:,0] = 127
        x, x2 = xq.astype(np.float16)/64, x2q.astype(np.float16)/64
        weights, expected = {}, {}
        for i,n in enumerate(widths):
            wq = rng.integers(-127,128,(n,k),dtype=np.int8); wq[:,0] = 127
            divisor = [32,64,128][i]
            weights[f'proj{i}.weight'] = wq.astype(np.float16)/divisor
            acc = (x2q if different and i == 2 else xq).astype(np.int32)@wq.astype(np.int32).T
            expected[f'y{i}'] = (acc.astype(np.float32)/(64*divisor)).astype(np.float16)
        save_file(weights,case/'model/model.safetensors')
        np.savez(case/'oracle.npz',x=x,x2=x2,**expected)
        (case/'step.mlir').write_text(source(m,k,widths,different))
        (case/'manifest.json').write_text(json.dumps(dict(model=str(case/'model'),tokens=m,capacity=1,
            mask_min=-30000.,arguments=[dict(kind='input',name='x'),dict(kind='input',name='x2')]+
            [dict(kind='weight',name=f'proj{i}.weight') for i in range(3)],
            results=[dict(kind='output',name=f'y{i}') for i in range(3)])))
        modes = {}
        for mode in a.modes:
            spec = json.loads((workspace/'research/npu_async_store_audit/configs/selected_m8.json').read_text())
            spec['npu'].update(projection_fusion=mode in ('fusion','both'),
                weight_stream_early_release=mode in ('release','both'),gemv_tile_bytes=k*32)
            if name == 'one_core': spec['npu']['cores'] = 1
            if name == 'two_slots': spec['npu']['l2_staging_bytes_per_core'] = k*32*2
            sp = case/f'{mode}.json'; sp.write_text(json.dumps(spec,indent=2))
            build = case/mode
            cmd = [str(compiler),'--pipeline=decode',str(case/'step.mlir'),'--manifest',str(case/'manifest.json'),
                   '--spec',str(sp),'--force=npu','-o',str(build)]
            p = subprocess.run(cmd,cwd=workspace,capture_output=True,text=True,
                               env=dict(os.environ,HYBRID_PROFILE_OPERATIONS='1'))
            (case/f'{mode}.compile.log').write_text(p.stdout+p.stderr)
            assert p.returncode == 0,(name,mode,p.stderr)
            io = json.loads((build/'io.json').read_text())
            fused = sum(ph.get('projection_fusion',{}).get('groups',0) for ph in io['phases'])
            assert fused == (1 if mode in ('fusion','both') else 0),(name,mode,fused)
            device = Device(build,settings=settings,lp6_config=ini,lp6_backend='ramulator',lp6_library=library)
            for arg in device.io['args']:
                if arg['kind']=='input': device.write(arg['name'],{'x':x,'x2':x2}[arg['name']])
            actual,commands,wall = device.run(profile=True)
            assert set(actual)==set(expected)
            for key in expected:
                assert actual[key].tobytes()==expected[key].tobytes(),(name,mode,key,'integer oracle mismatch')
            timing = json.loads((build/'timing.json').read_text())
            modes[mode] = dict(cycles=timing['total_simulated_cycles'],wall_seconds=wall,
                independent_reference_byte_exact=True,projection_fusions=fused,
                output_sha256={key:hashlib.sha256(v.tobytes()).hexdigest() for key,v in actual.items()},
                program_sha256=digest(build/'program.bin'),phases=io['phases'])
        results[name] = dict(shape=[m,k,*widths],modes=modes)
        print(name,{key:v['cycles'] for key,v in modes.items()},flush=True)
        (out/'summary.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__ == '__main__': main()
