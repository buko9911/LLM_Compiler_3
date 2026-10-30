#!/usr/bin/env python3
"""Check overlap eligibility, protected lifetimes and bounded SRAM failure."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import tempfile


SOURCE = '''
#a = affine_map<(m,n,k)->(m,k)>
#b = affine_map<(m,n,k)->(k,n)>
#c = affine_map<(m,n,k)->(m,n)>
module {
  func.func private @unknown()
  func.func @phase() attributes {hyb.phase = "npu"} {
    %z = arith.constant 0 : i32
    %acc = memref.alloc() {test.name = "acc"} : memref<1x32xi32, 1>
    %scale = memref.alloc() {test.name = "scale"} : memref<1x1xf32, 1>
    %ws = memref.alloc() {test.name = "ws"} : memref<32xf32, 1>
    %out = memref.alloc() {test.name = "out"} : memref<1x32xf16, 1>
    hyb.dequant %acc, %scale, %ws weight "gate" into %out : memref<1x32xi32, 1>, memref<1x1xf32, 1>, memref<32xf32, 1>, memref<1x32xf16, 1>
    scf.forall (%i) in (4) {
      %q = memref.alloc() {test.name = "q"} : memref<1x32xi8, 1>
      %w = memref.alloc() {test.name = "w"} : memref<32x32xi8, 1>
      %c = memref.alloc() {test.name = "c"} : memref<1x32xi32, 1>
      EXTRA
      linalg.fill ins(%z : i32) outs(%c : memref<1x32xi32, 1>)
      linalg.generic {indexing_maps = [#a,#b,#c], iterator_types = ["parallel","parallel","reduction"]}
        ins(%q,%w : memref<1x32xi8, 1>,memref<32x32xi8, 1>) outs(%c : memref<1x32xi32, 1>) attrs = {hyb.npu_gemv = "up"} {
      ^bb0(%x:i8,%y:i8,%s:i32):
        %xx = arith.extsi %x : i8 to i32
        %yy = arith.extsi %y : i8 to i32
        %p = arith.muli %xx,%yy : i32
        %r = arith.addi %s,%p : i32
        linalg.yield %r : i32
      }
    } {hyb.cores = 4 : i64}
    return
  }
}
'''


def main():
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--optimizer', type=Path, default=root/'build/bin/npu-opt')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix='run-', dir=args.output))
    cases = {
        'independent': ('', True, False),
        'disabled': ('', False, False),
        'reads_dequant_output': ('%readback = memref.alloc() : memref<1x32xf16, 1>\n memref.copy %out, %readback : memref<1x32xf16, 1> to memref<1x32xf16, 1>', True, False),
        'overwrites_dequant_input': ('memref.copy %c, %acc : memref<1x32xi32, 1> to memref<1x32xi32, 1>', True, False),
        'unknown_effects': ('func.call @unknown() : () -> ()', True, False),
        'sram_exhaustion': ('', True, True),
    }
    results = {}
    for name, (extra, enabled, exhausted) in cases.items():
        spec = json.loads((root/'configs/target.json').read_text())
        spec['npu']['overlap_dequant_gemv'] = enabled
        if exhausted:
            spec['npu']['l1_bytes'] = 4096 + spec['npu']['l1_reserved_bytes'] + 1024
        sp = out/(name+'.json'); sp.write_text(json.dumps(spec))
        src = out/(name+'.mlir'); src.write_text(SOURCE.replace('EXTRA', extra))
        command = [str(args.optimizer.resolve()), str(src),
                   f'--hyb-npu-overlap=spec={sp.resolve()}', f'--hyb-npu-memplan=spec={sp.resolve()}']
        run = subprocess.run(command, capture_output=True, text=True)
        (out/(name+'.log')).write_text(run.stdout+run.stderr)
        if exhausted:
            assert run.returncode and 'L1 overflow' in run.stderr, run.stderr
            results[name] = {'explicit_capacity_rejection': True}
            continue
        assert not run.returncode, run.stderr
        expected = name == 'independent'
        assert ('hyb.overlap_id' in run.stdout) == expected, name
        if expected:
            spans = {}
            for line in run.stdout.splitlines():
                match = re.search(r'hyb.offset = (\d+).*test.name = "(\w+)".*memref<([^>]+)>', line)
                if match:
                    begin, label, _ = match.groups()
                    size = dict(acc=128, scale=4, ws=128, out=64, q=32, w=1024, c=128)[label]
                    spans[label] = (int(begin), int(begin)+size)
            assert len(spans) == 7, spans
            for protected in ('acc','scale','ws','out'):
                for matrix in ('q','w','c'):
                    a,b = spans[protected]; c,d = spans[matrix]
                    assert b <= c or d <= a, (protected,matrix,spans)
            results[name] = {'paired': True, 'all_four_lifetimes_disjoint': True, 'ranges': spans}
        else:
            results[name] = {'paired': False, 'unchanged_schedule': True}
    # A bounded vector issue budget is part of the policy, not a silently
    # truncated floating-point tuning knob. Check both legal endpoints and
    # out-of-range/nonintegral values.
    for chunk in (16, 32, 512, 1024, 0, 17, 1025, 16.5):
        spec = json.loads((out/'independent.json').read_text())
        spec['npu']['overlap_dequant_chunk_elements'] = chunk
        sp = out/f'chunk-{chunk}.json'; sp.write_text(json.dumps(spec))
        run = subprocess.run([str(args.optimizer.resolve()), str(out/'independent.mlir'),
                              f'--hyb-npu-overlap=spec={sp.resolve()}'],
                             capture_output=True, text=True)
        legal = isinstance(chunk, int) and 16 <= chunk <= 1024 and chunk % 16 == 0
        if legal:
            assert not run.returncode and 'hyb.overlap_id' in run.stdout, run.stderr
        else:
            assert run.returncode and 'integer multiple of 16' in run.stderr, run.stderr
        results[f'chunk_{chunk}'] = {'accepted': legal}
    # Elementwise dequant lowering uses contiguous vector loads/stores. A
    # non-unit inner stride must not become an overlap candidate.
    strided = SOURCE.replace('EXTRA', '').replace(
        '%out = memref.alloc() {test.name = "out"} : memref<1x32xf16, 1>',
        '%backing = memref.alloc() : memref<1x64xf16, 1>\n'
        '    %out = memref.subview %backing[0,0][1,32][1,2] : '
        'memref<1x64xf16, 1> to memref<1x32xf16, strided<[64,2]>, 1>')
    strided = strided.replace('memref<1x32xf16, 1>', 'memref<1x32xf16, strided<[64,2]>, 1>')
    src = out/'strided.mlir'; src.write_text(strided)
    run = subprocess.run([str(args.optimizer.resolve()), str(src),
                          f'--hyb-npu-overlap=spec={(out/"independent.json").resolve()}'],
                         capture_output=True, text=True)
    (out/'strided.log').write_text(run.stdout+run.stderr)
    assert not run.returncode and 'hyb.overlap_id' not in run.stdout, run.stderr
    results['strided_inner_dimension'] = {'paired': False}
    for name, before, after in (
            ('bad_scale_shape', 'memref<32xf32, 1>', 'memref<31xf32, 1>'),
            ('bad_output_dtype', 'memref<1x32xf16, 1>', 'memref<1x32xf32, 1>')):
        src = out/f'{name}.mlir'
        src.write_text(SOURCE.replace('EXTRA', '').replace(before, after))
        run = subprocess.run([str(args.optimizer.resolve()), str(src),
                              f'--hyb-npu-overlap=spec={(out/"independent.json").resolve()}'],
                             capture_output=True, text=True)
        (out/f'{name}.log').write_text(run.stdout+run.stderr)
        assert not run.returncode and 'hyb.overlap_id' not in run.stdout, run.stderr
        results[name] = {'paired': False}
    (out/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__ == '__main__':
    main()
