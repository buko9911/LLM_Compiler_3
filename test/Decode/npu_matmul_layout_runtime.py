#!/usr/bin/env python3
"""Native transpose-B packing A/B. Existing simulator, exact outputs, real packing traffic."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np
from safetensors.numpy import save_file
from npu_attention_fusion_runtime import digest, run


def source(batch,m,n,k):
    prefix=f'{batch}x' if batch else ''
    a,b,c=(f'tensor<{prefix}{x}x{y}xf16>' for x,y in [(m,k),(n,k),(m,n)])
    maps=('affine_map<(b,m,n,k)->(b,m,k)>,affine_map<(b,m,n,k)->(b,n,k)>,affine_map<(b,m,n,k)->(b,m,n)>'
          if batch else 'affine_map<(m,n,k)->(m,k)>,affine_map<(m,n,k)->(n,k)>,affine_map<(m,n,k)->(m,n)>')
    return f'''module {{func.func @decode_step(%a:{a},%b:{b})->{c} {{
      %zero=arith.constant 0.0:f16
      %empty=tensor.empty():{c}
      %init=linalg.fill ins(%zero:f16) outs(%empty:{c})->{c}
      %result=linalg.{'batch_matmul' if batch else 'matmul'} indexing_maps=[{maps}]
        ins(%a,%b:{a},{b}) outs(%init:{c})->{c}
      return %result:{c}
    }} }}'''


def main():
    root=Path(__file__).resolve().parents[2];workspace=root.parents[1]
    sys.path.insert(0,str(root/'runtime'))
    from decode import Device
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--compiler',type=Path,default=root/'build/bin/npu-compile')
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    settings=workspace/'research/checkpoint_race/hardware/kernel_probe/settings_4core.toml'
    ini=workspace/'core/simulator/dram-pim/configs/LPDDR6_PIM_Mobile_4CH.ini'
    bridge=workspace/'core/ramulator2/liblp6_ramulator.so'
    model=out/'model';model.mkdir();save_file({'unused':np.zeros(1,dtype=np.float16)},model/'model.safetensors')
    sources=[args.compiler.resolve(),settings,ini,bridge,bridge.parent/'libramulator.so',Path(__file__).resolve(),root/'runtime/decode.py']
    provenance={str(p):digest(p) for p in sources}
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    cases=[('plain',0,8,64,32),('tail',0,5,37,19),('m_tail',0,33,37,65),
           ('batch',2,32,128,128),('batch_tail',3,5,37,19),
           ('fractional',0,17,35,129),('fractional_batch',2,7,65,33),
           ('gqa_ar',2,4,513,128),('gqa_verify',2,32,520,128),
           ('signed_zero',0,8,64,32),('subnormal',2,7,65,33),
           ('single_row_fallback',0,1,37,19)]
    summary={'kind':'native_cycle_simulator','cases':{}}
    for name,batch,m,n,k in cases:
        case=out/name;case.mkdir();src=case/'step.mlir';src.write_text(source(batch,m,n,k))
        manifest=case/'manifest.json';manifest.write_text(json.dumps(dict(model=str(model),tokens=m,capacity=0,mask_min=-30000.,
          arguments=[dict(kind='input',name=x) for x in ('a','b')],results=[dict(kind='output',name='y')]),indent=2))
        rng=np.random.default_rng(20261030)
        # Dyadic data gives an exact independent reference for non-power-of-two
        # K tails as well as ordinary tiles; native OFF/ON checks the full bytes.
        shape=lambda x,y:(batch,x,y) if batch else (x,y)
        inputs={'a':(rng.integers(-8,9,size=shape(m,k))/16).astype(np.float16),
                'b':(rng.integers(-8,9,size=shape(n,k))/16).astype(np.float16)}
        if name.startswith('fractional'):
            inputs={key:(rng.normal(size=array.shape)*0.2).astype(np.float16) for key,array in inputs.items()}
        if name=='signed_zero':
            for array in inputs.values():
                array.flat[::2]=np.float16(0.0);array.flat[1::2]=np.float16(-0.0)
        if name=='subnormal':
            for array in inputs.values():
                array.flat[::3]=np.nextafter(np.float16(0),np.float16(1));array.flat[1::3]=-np.nextafter(np.float16(0),np.float16(1))
        reference=np.matmul(inputs['a'].astype(np.float32),np.swapaxes(inputs['b'].astype(np.float32),-1,-2)).astype(np.float16)
        np.savez(case/'inputs.npz',**inputs);np.save(case/'reference.npy',reference)
        previous=None;runs={}
        for mode,enabled,swapped in [('off',False,False),('packed',True,False),('swapped',False,True)]:
            spec=json.loads((root/'configs/target.json').read_text());spec['npu'].update(matmul_rhs_packing=enabled,matmul_transpose_output=swapped,l2_staging_bytes_per_core=0)
            target=case/f'{mode}.json';target.write_text(json.dumps(spec,indent=2));build=case/mode
            run([args.compiler.resolve(),'--pipeline=decode',src,'--manifest',manifest,'--spec',target,'--force=npu','-o',build],workspace,case/f'{mode}.compile.log')
            compiled=(build/'compiled.mlir').read_text();applied=compiled.count('hyb.matmul_rhs_packed')
            swap_count=compiled.count('hyb.matmul_transpose_output')
            assert bool(applied)==(enabled and m>1),(name,mode,applied)
            assert bool(swap_count)==(swapped and m>1 and m*(k+n)<n*k),(name,mode,swap_count)
            device=Device(build,settings=settings,lp6_config=ini,lp6_backend='ramulator',lp6_library=bridge)
            for key,array in inputs.items():device.write(key,array)
            output,commands,wall=device.run(profile=True);actual=output['y'];np.save(build/'output.npy',actual)
            assert actual.shape==reference.shape,(name,actual.shape,reference.shape)
            assert np.isfinite(actual).all(),name
            error=float(np.max(np.abs(actual.astype(np.float32)-reference.astype(np.float32))))
            tolerance=0.002 if name.startswith('fractional') else 0.0
            assert error<=tolerance,(name,mode,error)
            exact=previous is None or actual.tobytes()==previous.tobytes()
            assert exact,(name,mode,'OFF/ON mismatch')
            timing=json.loads((build/'timing.json').read_text())
            runs[mode]=dict(total_us=timing['total_simulated_cycles']/timing['npu_clock_mhz'],packings=applied,output_transposes=swap_count,
                           byte_exact_with_off=exact,max_abs_numpy_error=error,numpy_abs_tolerance=tolerance,wall_seconds=wall,
                           read_bytes=timing['lp6_bytes_read'],write_bytes=timing['lp6_bytes_written'],
                           program_sha256=digest(build/'program.bin'),compiled_sha256=digest(build/'compiled.mlir'))
            if previous is None:previous=actual
        summary['cases'][name]=runs;(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(name,json.dumps(runs),flush=True)
    assert provenance=={str(p):digest(p) for p in sources},'Source changed during execution'
if __name__=='__main__':main()
