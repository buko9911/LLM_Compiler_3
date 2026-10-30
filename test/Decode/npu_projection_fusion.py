#!/usr/bin/env python3
"""Projection fusion rejection boundaries and module-wide partition groups."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import tempfile

from npu_gemv_optimizations import tensor_source


def source(shared=False, different_input=False, different_scale=False):
    args = ['%q:tensor<8x64xi8>','%s:tensor<8x1xf32>',
            '%q2:tensor<8x64xi8>','%s2:tensor<8x1xf32>']
    bodies, returns, types = [], [], []
    for i,nb in enumerate([12,4,4]):
        text = tensor_source(nb=nb)
        body = text[text.index('  %zero'):text.index('  return')]
        body = re.sub(r'%([a-zA-Z][a-zA-Z0-9_]*)',
            lambda m: '%'+m[1]+(str(i) if m[1] not in ('q','s') else ''),body)
        body = body.replace('proj.weight',f'proj{i}.weight')
        # Replace input-weight function arguments with immutable named weights.
        body = f'''%w{i} = hyb.weight "proj{i}.weight" layout "npu_nk32" : tensor<{nb}x64x32xi8>
%ws{i} = hyb.weight "proj{i}.weight" layout "row_scale" : tensor<{nb*32}xf32>
'''+body
        if i == 2:
            if different_input: body = re.sub(r'%q\b','%q2',body)
            if different_scale: body = re.sub(r'%s\b','%s2',body)
        bodies.append(body); returns.append(f'%y{i}'); types.append(f'tensor<8x{nb*32}xf16>')
    if shared:
        returns.append('%g0'); types.append('tensor<8x12x32xi32>')
    return 'module { func.func @phase('+','.join(args)+') -> ('+','.join(types)+') '+\
        'attributes {hyb.phase="npu"} {\n'+''.join(bodies)+'return '+','.join(returns)+' : '+','.join(types)+'\n}}\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--optimizer',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a = ap.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix='run-',dir=a.output))
    checks = {}
    cases = [
        ('qkv',source(),{},1,1),
        ('off',source(),{'projection_fusion':False},0,3),
        ('different_input',source(different_input=True),{},1,2),
        ('different_scale',source(different_scale=True),{},1,2),
        ('shared_accumulator',source(shared=True),{},1,2),
        ('nonzero',source().replace('constant 0 : i32','constant 1 : i32'),{},0,3),
        ('different_math',source().replace('arith.muli','arith.addi'),{},0,3),
        ('pim',source().replace('hyb.phase="npu"','hyb.phase="pim"'),{},0,3),
        ('small_sram',source(),{'l1_bytes':270336},0,3),
    ]
    for name,text,options,groups,contractions in cases:
        sp=out/f'{name}.json'; src=out/f'{name}.mlir'
        sp.write_text(json.dumps({'npu':{'projection_fusion':True,**options}}));src.write_text(text)
        cmd=[str(a.optimizer.resolve()),str(src),f'--hyb-npu-projection-fusion=spec={sp.resolve()}',
             '--canonicalize','--verify-each']
        p=subprocess.run(cmd,text=True,capture_output=True)
        (out/f'{name}.log').write_text(p.stdout+p.stderr)
        assert not p.returncode,(name,p.stderr)
        assert p.stdout.count('hyb.npu_gemv =')==contractions,(name,p.stdout)
        if groups: assert f'hyb.projection_fusions = {groups}' in p.stdout,(name,p.stdout)
        else: assert 'hyb.weight_segments' not in p.stdout
        checks[name]={'groups':groups,'contractions':contractions}
    for key in ['projection_fusion','weight_stream_early_release']:
        sp=out/'invalid.json';sp.write_text(json.dumps({'npu':{key:'true'}}))
        p=subprocess.run([str(a.optimizer.resolve()),str(out/'qkv.mlir'),
            f'--hyb-npu-projection-fusion=spec={sp.resolve()}'],text=True,capture_output=True)
        assert p.returncode and f'npu.{key} must be a boolean' in p.stderr
        checks[key]={'invalid_type_rejected':True}
    # Input identity, not function-local numeric group annotations, determines
    # partition membership. Two functions must not accidentally share group 0.
    src=out/'partition.mlir'
    function='''func.func @NAME(%q:tensor<1x64xi8>,%c:tensor<1x32xi32>) -> tensor<1x32xi32> {
      %y = hyb.qmatmul %q weight "NAME.weight" into %c {group=0:i64} : tensor<1x64xi8>,tensor<1x32xi32> -> tensor<1x32xi32>
      return %y : tensor<1x32xi32> }'''
    src.write_text('module {'+function.replace('NAME','a')+function.replace('NAME','b')+'}')
    p=subprocess.run([str(a.optimizer.resolve()),str(src),'--hyb-partition=force=npu'],text=True,capture_output=True)
    assert not p.returncode,p.stderr
    assert 'group = 0' in p.stdout and 'group = 1' in p.stdout,p.stdout
    checks['partition_function_scope']={'distinct_groups':True}
    (out/'summary.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2))


if __name__=='__main__': main()
