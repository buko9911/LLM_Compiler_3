#!/usr/bin/env python3
"""Native Ramulator/NPU validation against an independent integer GEMM oracle.

The generated weights/activations have exact binary scales. Check selected
compiler policies, rectangular/tail MMA shapes, stream wrap and no-stream paths.
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


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source(m, k, n):
    return f'''module {{
 func.func @decode_step(%x:tensor<{m}x{k}xf32>,%w:tensor<{n}x{k}xf32>) -> tensor<{m}x{n}xf32> {{
  %zero = arith.constant 0.0 : f32
  %wt = tensor.empty() : tensor<{k}x{n}xf32>
  %t = linalg.transpose ins(%w:tensor<{n}x{k}xf32>) outs(%wt:tensor<{k}x{n}xf32>) permutation=[1,0]
  %e = tensor.empty() : tensor<{m}x{n}xf32>
  %i = linalg.fill ins(%zero:f32) outs(%e:tensor<{m}x{n}xf32>) -> tensor<{m}x{n}xf32>
  %y = linalg.matmul ins(%x,%t:tensor<{m}x{k}xf32>,tensor<{k}x{n}xf32>) outs(%i:tensor<{m}x{n}xf32>) -> tensor<{m}x{n}xf32>
  return %y : tensor<{m}x{n}xf32>
 }}
}}
'''


def main():
    root = Path(__file__).resolve().parents[2]
    workspace = root.parents[1]
    sys.path.insert(0, str(root/'runtime'))
    from decode import Device
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    policies = {'off': (False, False), 'init': (True, False), 'epilogue': (False, True),
                'both': (True, True), 'rotated': (True, True), 'batch1': (True, True),
                'async': (False, False), 'both_async': (True, True),
                'async_immediate': (False, False), 'release': (False, False),
                'pipeline': (True, True), 'pipeline_release': (True, True), 'round2': (True, True)}
    parser.add_argument('--modes', nargs='+', choices=list(policies),
                        default=['off', 'init', 'epilogue', 'both', 'rotated', 'batch1'])
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    compiler = root/'build/bin/npu-compile'
    settings = workspace/'research/checkpoint_race/hardware/kernel_probe/settings_4core.toml'
    library = workspace/'core/ramulator2/liblp6_ramulator.so'
    ini = workspace/'core/simulator/dram-pim/configs/LPDDR6_PIM_Mobile_4CH.ini'
    (output/'provenance.json').write_text(json.dumps({str(p):digest(p) for p in
        [Path(__file__),compiler,settings,library,ini]},indent=2)+'\n')
    results = {}
    for name,m,k,n,tile in [('m1',1,64,1152,4096),('m8',8,256,768,16384),
                           ('m33_rectangles',33,64,1152,6144),('single_tile',8,64,128,8192),
                           ('one_core',8,64,384,4096),('two_slots',8,64,1152,2048),
                           ('m8_k1024',8,1024,256,32768)]:
        case = output/name
        case.mkdir(); (case/'model').mkdir()
        rng = np.random.default_rng(917)
        xq = rng.integers(-127,128,size=(m,k),dtype=np.int8)
        wq = rng.integers(-127,128,size=(n,k),dtype=np.int8)
        xq[:,0] = wq[:,0] = 127
        x, w = xq.astype(np.float16)/np.float16(64), wq.astype(np.float16)/np.float16(64)
        expected = ((xq.astype(np.int32)@wq.astype(np.int32).T).astype(np.float32)/4096).astype(np.float16)
        save_file({'proj.weight':w},case/'model/model.safetensors')
        np.savez(case/'oracle.npz',x=x,xq=xq,wq=wq,expected=expected)
        (case/'step.mlir').write_text(source(m,k,n))
        (case/'manifest.json').write_text(json.dumps(dict(model=str(case/'model'),tokens=m,capacity=1,
            mask_min=-30000.,arguments=[dict(kind='input',name='x'),dict(kind='weight',name='proj.weight')],
            results=[dict(kind='output',name='y')]),indent=2))
        modes = {}
        for mode in args.modes:
            init, fuse = policies[mode]
            spec = json.loads((workspace/'research/npu_schedule_audit/configs/off.json').read_text())
            spec['npu'].update(matrix_init_elision=init,gemv_epilogue_fusion=fuse,gemv_tile_bytes=tile)
            if mode in ('release', 'pipeline_release'):
                spec['npu']['weight_stream_early_release'] = True
            if mode in ('pipeline', 'pipeline_release'):
                spec['npu']['gemv_epilogue_pipeline'] = True
            if mode == 'round2':   # the 2026-10-31 switches together with the pipelined epilogue
                spec['npu'].update(weight_stream_early_release=True, gemv_epilogue_pipeline=True,
                                   quant_fused_max=True, vector_loops=True, staging_misc_bytes=4096,
                                   gdma_lookahead=2, gdma_lookahead_streams_only=True, cross_phase_prefetch=True)
            if mode in ('async', 'both_async', 'async_immediate'):
                spec['npu']['async_output_stores'] = True
            if mode == 'async_immediate':
                spec['npu']['async_output_store_vector_overlap'] = False
            if name == 'one_core':
                spec['npu']['cores'] = 1
            if name == 'two_slots':
                spec['npu']['l2_staging_bytes_per_core'] = 4096
            if mode in ('rotated','batch1'):
                spec['npu']['weight_stream_rotate_slots'] = True
            if mode == 'batch1':
                spec['npu']['weight_prefetch_batch_tiles'] = 1
            sp=case/f'{mode}.json';sp.write_text(json.dumps(spec,indent=2))
            build=case/mode
            command=[str(compiler),'--pipeline=decode',str(case/'step.mlir'),'--manifest',str(case/'manifest.json'),
                     '--spec',str(sp),'--force=npu','-o',str(build)]
            run=subprocess.run(command,cwd=workspace,text=True,capture_output=True,
                               env=dict(os.environ,HYBRID_PROFILE_OPERATIONS='1'))
            (case/f'{mode}.compile.log').write_text(run.stdout+run.stderr)
            assert run.returncode == 0,run.stderr
            device=Device(build,settings=settings,lp6_config=ini,lp6_backend='ramulator',lp6_library=library)
            device.write('x',x)
            actual,commands,wall=device.run(profile=True)
            assert actual['y'].tobytes()==expected.tobytes(),(name,mode,'integer oracle mismatch')
            io=json.loads((build/'io.json').read_text())
            streams=[s for phase in io['phases'] for s in phase.get('weight_streams',[])]
            if name not in ('single_tile','two_slots') and (name != 'one_core' or fuse):
                assert streams and all(s['prefetch'] for s in streams),(name,mode,streams)
            elif name == 'two_slots':
                assert streams and all(not s['prefetch'] for s in streams),(name,mode,streams)
            timing=json.loads((build/'timing.json').read_text())
            modes[mode]=dict(cycles=timing['total_simulated_cycles'],wall_seconds=wall,
                independent_reference_byte_exact=True,output_sha256=hashlib.sha256(actual['y'].tobytes()).hexdigest(),
                program_sha256=digest(build/'program.bin'),weight_streams=streams)
        results[name]=dict(shape=[m,k,n],modes=modes)
        print(name,{key:value['cycles'] for key,value in modes.items()},flush=True)
        (output/'summary.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':
    main()
