#!/usr/bin/env python3
"""Safety boundaries for matrix-init elision and producer-local dequantization."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile


def tensor_source(shared=False, phase='npu', m=8, nb=12):
    n = nb * 32
    acc, weight = f'{m}x{nb}x32xi32', f'{nb}x64x32xi8'
    result = f'tensor<{m}x{n}xf16>'
    returned = f'%y : {result}'
    if shared:
        result = f'({result}, tensor<{acc}>)'
        returned = f'%y, %g : tensor<{m}x{n}xf16>, tensor<{acc}>'
    return f'''module {{
 func.func @phase(%q: tensor<{m}x64xi8>, %w: tensor<{weight}>,
                  %s: tensor<{m}x1xf32>, %ws: tensor<{n}xf32>) -> {result}
                  attributes {{hyb.phase = "{phase}"}} {{
  %zero = arith.constant 0 : i32
  %e = tensor.empty() : tensor<{acc}>
  %i = linalg.fill ins(%zero:i32) outs(%e:tensor<{acc}>) -> tensor<{acc}>
  %g = linalg.generic {{indexing_maps=[affine_map<(m,b,c,k)->(m,k)>,
       affine_map<(m,b,c,k)->(b,k,c)>,affine_map<(m,b,c,k)->(m,b,c)>],
       iterator_types=["parallel","parallel","parallel","reduction"]}}
       ins(%q,%w:tensor<{m}x64xi8>,tensor<{weight}>) outs(%i:tensor<{acc}>)
       attrs = {{hyb.npu_gemv = "proj.weight"}} {{
  ^bb0(%x:i8,%wv:i8,%cv:i32):
    %xx = arith.extsi %x : i8 to i32
    %ww = arith.extsi %wv : i8 to i32
    %p = arith.muli %xx,%ww : i32
    %r = arith.addi %cv,%p : i32
    linalg.yield %r : i32
  }} -> tensor<{acc}>
  %c = tensor.collapse_shape %g [[0],[1,2]] : tensor<{acc}> into tensor<{m}x{n}xi32>
  %o = tensor.empty() : tensor<{m}x{n}xf16>
  %y = hyb.dequant %c,%s,%ws weight "proj.weight" into %o : tensor<{m}x{n}xi32>,
       tensor<{m}x1xf32>,tensor<{n}xf32>,tensor<{m}x{n}xf16> -> tensor<{m}x{n}xf16>
  return {returned}
 }}
}}
'''


def memref_source(before='', after='', zero=0, phase='npu'):
    return f'''module {{
 func.func private @escape(memref<8x32xi32,1>)
 func.func @phase(%q:memref<8x64xi8,1>,%w:memref<64x32xi8,1>,%out:memref<8x32xi32,2>)
     attributes {{hyb.phase = "{phase}"}} {{
  %zero = arith.constant {zero} : i32
  %c = memref.alloc() : memref<8x32xi32,1>
  linalg.fill ins(%zero:i32) outs(%c:memref<8x32xi32,1>)
  {before}
  linalg.generic {{indexing_maps=[affine_map<(m,n,k)->(m,k)>,
    affine_map<(m,n,k)->(k,n)>,affine_map<(m,n,k)->(m,n)>],
    iterator_types=["parallel","parallel","reduction"]}}
    ins(%q,%w:memref<8x64xi8,1>,memref<64x32xi8,1>) outs(%c:memref<8x32xi32,1>)
    attrs = {{hyb.npu_gemv = "proj.weight"}} {{
   ^bb0(%x:i8,%wv:i8,%cv:i32):
    %xx = arith.extsi %x : i8 to i32
    %ww = arith.extsi %wv : i8 to i32
    %p = arith.muli %xx,%ww : i32
    %r = arith.addi %cv,%p : i32
    linalg.yield %r : i32
  }}
  {after}
  memref.copy %c,%out : memref<8x32xi32,1> to memref<8x32xi32,2>
  return
 }}
}}
'''


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--optimizer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='run-', dir=args.output))
    results = {}

    def check(name, source, option, enabled, expected, changes=None):
        target = json.loads((root/'configs/target.json').read_text())
        target['npu'].update({option: enabled, **(changes or {})})
        sp, src = output/f'{name}.json', output/f'{name}.mlir'
        sp.write_text(json.dumps(target)); src.write_text(source)
        epilogue = option == 'gemv_epilogue_fusion'
        flag = 'hyb-npu-gemv-epilogue' if epilogue else 'hyb-npu-matrix-init'
        command = [str(args.optimizer.resolve()), str(src), f'--{flag}=spec={sp.resolve()}']
        if epilogue:
            command += [f'--hyb-npu-tile=spec={sp.resolve()}', '--canonicalize']
        run = subprocess.run(command, text=True, capture_output=True)
        (output/f'{name}.log').write_text(run.stdout+run.stderr)
        assert run.returncode == 0, (name, run.stderr)
        marker = 'hyb.gemv_epilogue_fusions = 1' if epilogue else 'hyb.elide_matrix_init'
        assert (marker in run.stdout) == expected, (name, run.stdout)
        if epilogue and expected:
            assert 'hyb.stream' in run.stdout or changes and changes.get('cores') == 1
            assert run.stdout.count('hyb.dequant ') == 1
            assert run.stdout.count('hyb.npu_gemv = ') == 1
        results[name] = {'transformed': expected}

    for name, source, enabled, expected, changes in [
        ('epilogue', tensor_source(), True, True, None),
        ('epilogue_off', tensor_source(), False, False, None),
        ('shared_accumulator', tensor_source(shared=True), True, False, None),
        ('nonzero_accumulator', tensor_source().replace('constant 0 : i32', 'constant 1 : i32'), True, False, None),
        ('different_weight_scale', tensor_source().replace('weight "proj.weight"', 'weight "other.weight"'), True, False, None),
        ('pim_phase', tensor_source(phase='pim'), True, False, None),
        ('one_core', tensor_source(), True, True, {'cores': 1}),
        ('sram_admission', tensor_source(), True, False, {'l1_bytes': 270336}),
        ('m33', tensor_source(m=33), True, True, None),
    ]:
        # Force several weight tiles in small synthetic examples.
        check(name, source, 'gemv_epilogue_fusion', enabled, expected,
              {'gemv_tile_bytes': 2048, **(changes or {})})
    view = '%v = memref.cast %c : memref<8x32xi32,1> to memref<?x32xi32,1>'
    for name, source, enabled, expected in [
        ('init', memref_source(), True, True),
        ('init_off', memref_source(), False, False),
        ('nonzero', memref_source(zero=1), True, False),
        ('read_zero', memref_source(before='memref.copy %c,%out : memref<8x32xi32,1> to memref<8x32xi32,2>'), True, False),
        ('escape_before', memref_source(before='func.call @escape(%c) : (memref<8x32xi32,1>) -> ()'), True, False),
        ('view_before', memref_source(before=view), True, False),
        ('view_after', memref_source(after=view), True, True),
        ('init_pim', memref_source(phase='pim'), True, False),
    ]:
        check(name, source, 'matrix_init_elision', enabled, expected)
    for key in ('matrix_init_elision','gemv_epilogue_fusion','weight_stream_rotate_slots',
                'async_output_stores','async_output_store_vector_overlap',
                'async_input_loads','input_tile_pipeline','matmul_rhs_packing',
                'matmul_transpose_output'):
        spec = output/'invalid.json'
        spec.write_text(json.dumps({'npu':{key:'true'}}))
        flag = 'hyb-npu-gemv-epilogue' if key == 'gemv_epilogue_fusion' else 'hyb-npu-matrix-init'
        run = subprocess.run([str(args.optimizer.resolve()), str(output/'init.mlir'), f'--{flag}=spec={spec}'],
                             text=True,capture_output=True)
        assert run.returncode and f'npu.{key} must be a boolean' in run.stderr
        results[key+'_invalid_type'] = {'rejected': True}
    (output/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__ == '__main__':
    main()
