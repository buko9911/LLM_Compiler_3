#!/usr/bin/env python3
"""Safety boundaries for common FP16 transpose-B materialization."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile


def source(m=32, n=520, k=128, batch=2, phase='npu', dtype='f16', alias=False,
           canonical=False, zero=0.0, output_strided=False, read_after_fill=False):
    batch_dims = f'{batch}x' if batch else ''
    a=f'memref<{batch_dims}{m}x{k}x{dtype}, 1>'
    b=f'memref<{batch_dims}{n}x{k}x{dtype}, 1>'
    c=f'memref<{batch_dims}{m}x{n}x{dtype}, 1>'
    if output_strided:
        c=c.replace(', 1>', ', strided<[16640, 520, 0]>, 1>')
    maps = ('affine_map<(b,m,n,k)->(b,m,k)>, affine_map<(b,m,n,k)->(b,n,k)>, affine_map<(b,m,n,k)->(b,m,n)>'
            if batch else 'affine_map<(m,n,k)->(m,k)>, affine_map<(m,n,k)->(n,k)>, affine_map<(m,n,k)->(m,n)>')
    if canonical:
        b=f'memref<{batch_dims}{k}x{n}x{dtype}, 1>'
        maps=maps.replace('->(b,n,k)','->(b,k,n)').replace('->(n,k)','->(k,n)')
    output='%a' if alias else '%c'
    if alias:
        assert a==c
    read = f'%sink=memref.alloc():{c}\nmemref.copy {output},%sink:{c} to {c}' if read_after_fill else ''
    return f'''module {{ func.func @phase(%a:{a}, %b:{b}) attributes {{hyb.phase="{phase}"}} {{
      %zero = arith.constant {zero} : {dtype}
      %c = memref.alloc() : {c}
      linalg.fill ins(%zero:{dtype}) outs({output}:{c})
      {read}
      linalg.{'batch_matmul' if batch else 'matmul'} indexing_maps=[{maps}]
        ins(%a,%b:{a},{b}) outs({output}:{c})
      return
    }} }}'''


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--optimizer',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    out=Path(tempfile.mkdtemp(prefix='run-',dir=args.output))
    root=Path(__file__).resolve().parents[2]
    cases=[('batch',{},True,True,{}),('plain',{'batch':0},True,True,{}),
           ('tails',{'m':5,'n':37,'k':19},True,True,{}),
           ('off',{},False,False,{}),('single_row',{'m':1},True,False,{}),
           ('other_phase',{'phase':'pim'},True,False,{}),('f32',{'dtype':'f32'},True,False,{}),
           ('canonical',{'canonical':True},True,False,{}),
           ('rhs_output_overlap',{'output_strided':True},True,False,{}),
           ('alias',{'m':32,'n':128,'k':128,'alias':True},True,False,{}),
           ('capacity',{},True,False,{'l1_bytes':262144}),
           ('too_large',{'n':20000},True,False,{}),
           ('swap_batch',{},False,'swap',{'matmul_transpose_output':True}),
           ('swap_plain',{'batch':0},False,'swap',{'matmul_transpose_output':True}),
           ('swap_tail',{'m':5,'n':37,'k':19},False,'swap',{'matmul_transpose_output':True}),
           ('swap_intervening_read',{'read_after_fill':True},False,False,{'matmul_transpose_output':True}),
           ('swap_nonzero',{'zero':1.0},False,False,{'matmul_transpose_output':True}),
           ('swap_negative_zero',{'zero':-0.0},False,False,{'matmul_transpose_output':True}),
           ('swap_output_overlap',{'output_strided':True},False,False,{'matmul_transpose_output':True}),
           ('swap_m1',{'m':1},False,False,{'matmul_transpose_output':True}),
           ('swap_no_traffic_win',{'m':32,'n':8,'k':16},False,False,{'matmul_transpose_output':True}),
           ('swap_capacity',{},False,False,{'matmul_transpose_output':True,'l1_bytes':262144}),
           ('swap_alias',{'m':32,'n':128,'k':128,'alias':True},False,False,{'matmul_transpose_output':True})]
    result={}
    for name,kw,enabled,expected,options in cases:
        src=out/f'{name}.mlir';src.write_text(source(**kw))
        spec=json.loads((root/'configs/target.json').read_text())
        spec['npu'].update(matmul_rhs_packing=enabled,**options)
        path=out/f'{name}.json';path.write_text(json.dumps(spec))
        cmd=[str(args.optimizer.resolve()),str(src),f'--hyb-npu-matmul-layout=spec={path.resolve()}']
        run=subprocess.run(cmd,text=True,capture_output=True)
        (out/f'{name}.log').write_text(run.stdout+run.stderr)
        assert run.returncode==0,(name,run.stderr)
        packed='hyb.matmul_rhs_packed' in run.stdout
        swapped='hyb.matmul_transpose_output' in run.stdout
        assert packed==(expected is True),(name,packed,expected)
        assert swapped==(expected=='swap'),(name,swapped,expected)
        if packed or swapped:
            assert 'linalg.transpose' in run.stdout
            if swapped:assert run.stdout.count('hyb.elide_matrix_init')==2,name
            assert ('hyb.matmul_rhs_packings = 1' if packed else 'hyb.matmul_output_transposes = 1') in run.stdout
            # Idempotence: a packed canonical map must not be packed twice.
            again=subprocess.run(cmd[:1]+['-',cmd[-1]],input=run.stdout,text=True,capture_output=True)
            assert again.returncode==0,again.stderr
            assert again.stdout.count('linalg.transpose')==(1 if packed else 2),name
        result[name]={'passed':True,'packed':packed,'swapped':swapped}
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(out)
if __name__=='__main__':main()
