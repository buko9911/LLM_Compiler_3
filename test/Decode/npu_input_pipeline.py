#!/usr/bin/env python3
"""Input-copy/matrix pipeline IR legality and fallback tests, independent of op names."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import tempfile


def source(batch=0, n=65, shared=False, nonzero=False, alias=False, transposed=False, phase='npu'):
    prefix = f'{batch}x' if batch else ''
    a = prefix+'4x8xf16'
    b = prefix+(f'{n}x8xf16' if transposed else f'8x{n}xf16')
    c = prefix+f'4x{n}xf16'
    op = 'linalg.batch_matmul' if batch else 'linalg.matmul'
    maps = ''
    if transposed:
        assert not batch
        maps = 'indexing_maps = [affine_map<(m,n,k)->(m,k)>,affine_map<(m,n,k)->(n,k)>,affine_map<(m,n,k)->(m,n)>]'
    shared_op = f'memref.copy %b, %another : memref<{b}, 1> to memref<{b}, 1>' if shared else ''
    alias_op = f'%alias = memref.cast %c : memref<{c}, 1> to memref<*xf16, 1>' if alias else ''
    return f'''module {{
  func.func @phase(%src: memref<{b}, 2>) attributes {{hyb.phase = "{phase}"}} {{
    %zero = arith.constant {'1.0' if nonzero else '0.0'} : f16
    %a = memref.alloc() : memref<{a}, 1>
    %b = memref.alloc() : memref<{b}, 1>
    %c = memref.alloc() : memref<{c}, 1>
    %another = memref.alloc() : memref<{b}, 1>
    linalg.fill ins(%zero : f16) outs(%c : memref<{c}, 1>)
    {alias_op}
    memref.copy %src, %b : memref<{b}, 2> to memref<{b}, 1>
    {op} {maps} ins(%a, %b : memref<{a}, 1>, memref<{b}, 1>) outs(%c : memref<{c}, 1>)
    {shared_op}
    return
  }}
}}'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument('--opt', type=Path, default=root/'build/bin/npu-opt')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='npu-input-pipeline-') as directory:
        work = Path(directory)
        config = json.loads((root/'configs/target.json').read_text())
        config['npu'].update(input_tile_pipeline=True, input_tile_columns=32, async_input_loads=True)
        spec = work/'target.json';spec.write_text(json.dumps(config))
        cases = [('matmul_tail', source(), 3), ('batched_tail', source(batch=3), 9),
                 ('batch_only', source(batch=2,n=17), 2), ('one_tile', source(n=17), 0),
                 ('shared_B', source(shared=True), 0), ('nonzero_init', source(nonzero=True), 0),
                 ('output_alias', source(alias=True), 0), ('transpose_B', source(transposed=True), 0),
                 ('pim_phase', source(phase='pim'), 0),
                 ('overlapping_B', source().replace('memref<8x65xf16, 1>', 'memref<8x65xf16, strided<[32, 1]>, 1>'), 0),
                 ('overlapping_C', source().replace('memref<4x65xf16, 1>', 'memref<4x65xf16, strided<[32, 1]>, 1>'), 0)]
        for name,text,tiles in cases:
            path=work/f'{name}.mlir';path.write_text(text)
            result=subprocess.run([str(args.opt),str(path),f'--hyb-npu-input-pipeline=spec={spec}', '--verify-each'],text=True,capture_output=True)
            assert result.returncode==0,(name,result.stderr)
            output=result.stdout
            assert output.count('hyb.input_pipeline_copy')==tiles,(name,output)
            assert len(re.findall(r'hyb\.input_pipeline_tile\b', output))==tiles,(name,output)
            assert output.count('memref.alloc')==4,(name,'changed storage allocation')
            if tiles:
                assert output.count('linalg.fill')==tiles,(name,'missing tile init')
                # Full K=8 survives every tile; the first matrix is issued after
                # two copies, while subsequent tiles reuse the event/addresses.
                scheduled=[line for line in output.splitlines() if 'hyb.input_pipeline_' in line and not 'func.func' in line]
                assert 'input_pipeline_copy' in scheduled[0] and 'input_pipeline_copy' in scheduled[1]
                assert 'input_pipeline_tile' in scheduled[2],scheduled
            print(name,'PASS',tiles)
        config['npu']['input_tile_pipeline']=False;spec.write_text(json.dumps(config))
        path=work/'off.mlir';path.write_text(source())
        command=[str(args.opt),str(path)]
        before=subprocess.check_output(command,text=True)
        after=subprocess.check_output(command+[f'--hyb-npu-input-pipeline=spec={spec}'],text=True)
        assert before==after,'disabled pass changed IR'
        print('disabled_identity PASS')

if __name__=='__main__':main()
