#!/usr/bin/env python3
"""Check escaping-result isolation and real L2 placement after bufferization.

The positive case shares one empty destination between a local producer and a
multi-core consumer's exported result. It reproduces the QKV / Gate-Up spill
mechanism without depending on a particular model's names or numerical values.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import tempfile


SOURCE = '''
#id = affine_map<(d0) -> (d0)>
#offset = affine_map<(d0) -> (d0 * 8)>
module {
  func.func private @npu_phase() -> tensor<32xf16> attributes {hyb.phase = "npu"} {
    %one = arith.constant 1.0 : f16
    %two = arith.constant 2.0 : f16
    %empty = tensor.empty() : tensor<32xf16>
    %local = linalg.fill ins(%one : f16) outs(%empty : tensor<32xf16>) -> tensor<32xf16>
    %middle_init = tensor.empty() : tensor<32xi32>
    %middle = scf.forall (%i) in (4) shared_outs(%shared = %middle_init) -> tensor<32xi32> {
      %off = affine.apply #offset(%i)
      %input = tensor.extract_slice %local[%off] [8] [1] : tensor<32xf16> to tensor<8xf16>
      %init = tensor.extract_slice %shared[%off] [8] [1] : tensor<32xi32> to tensor<8xi32>
      %value = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
        ins(%input : tensor<8xf16>) outs(%init : tensor<8xi32>) {
      ^bb0(%x : f16, %unused : i32):
        %integer = arith.fptosi %x : f16 to i32
        linalg.yield %integer : i32
      } -> tensor<8xi32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %value into %shared[%off] [8] [1] : tensor<8xi32> into tensor<32xi32>
      }
    } {hyb.cores = 4 : i64}
    %result = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
      ins(%middle : tensor<32xi32>) outs(%empty : tensor<32xf16>) {
    ^bb0(%x : i32, %unused : f16):
      %float = arith.sitofp %x : i32 to f16
      %sum = arith.addf %float, %two : f16
      linalg.yield %sum : f16
    } -> tensor<32xf16>
    return %result : tensor<32xf16>
  }
  func.func @decode_step() -> tensor<32xf16> {
    %r = func.call @npu_phase() : () -> tensor<32xf16>
    return %r : tensor<32xf16>
  }
}
'''


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--optimizer', type=Path, default=root/'build/bin/npu-opt')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix='run-', dir=args.output))
    optimizer = str(args.optimizer.resolve())
    for enabled in (False, True):
        target = json.loads((root/'configs/target.json').read_text())
        target['npu']['activation_residency'] = enabled
        (out/f'{enabled}.json').write_text(json.dumps(target))

    results = {}

    def run(name, source=SOURCE, enabled=True, extra=(), bare=False):
        src = out/(name+'.mlir'); src.write_text(source)
        command = [optimizer, str(src)]
        if not bare:
            command.append(f'--hyb-npu-activation-residency=spec={out / (str(enabled)+".json")}')
        command.extend(extra)
        result = subprocess.run(command, capture_output=True, text=True)
        (out/(name+'.log')).write_text(result.stdout+result.stderr)
        assert result.returncode == 0, (name, result.stderr)
        return result.stdout

    baseline = run('original', bare=True)
    disabled = run('disabled', enabled=False)
    assert disabled == baseline, 'disabled pass changed IR'
    results['disabled'] = {'byte_identical_ir': True}
    positive = run('shared_destination')
    assert 'hyb.activation_residency_private_results = 1' in positive
    assert positive.count('tensor.empty') == baseline.count('tensor.empty') + 1
    # Output and local producer now have different init values. Tensor values,
    # loop geometry and the scalar expression are unchanged.
    fill_init = re.search(r'linalg.fill.*outs\((%\w+) :', positive)[1]
    return_init = re.findall(r'ins\([^\n]+tensor<32xi32>\) outs\((%\w+) : tensor<32xf16>\)', positive)[-1]
    assert fill_init != return_init
    assert positive.count('arith.addf') == baseline.count('arith.addf') == 1
    results['shared_destination'] = {'private_results': 1, 'producer_and_result_distinct': True}

    repeated = run('idempotent', positive)
    assert repeated == positive, 'repeating pass changed IR or accounting'
    results['idempotent'] = {'byte_identical_ir': True}

    private = SOURCE.replace('%result = linalg.generic', '%private = tensor.empty() : tensor<32xf16>\n    %result = linalg.generic').replace(
        'ins(%middle : tensor<32xi32>) outs(%empty : tensor<32xf16>)',
        'ins(%middle : tensor<32xi32>) outs(%private : tensor<32xf16>)')
    existing = run('already_private', private)
    assert 'hyb.activation_residency_private_results = 0' in existing
    assert existing.count('tensor.empty') == 3
    results['already_private'] = {'private_results': 0}

    mixed = SOURCE.replace('  func.func @decode_step', '  func.func private @pim_phase() attributes {hyb.phase = "pim"} { return }\n  func.func @decode_step')
    assert run('mixed_npu_pim', mixed) == run('mixed_original', mixed, bare=True)
    results['mixed_npu_pim'] = {'byte_identical_ir': True}
    packet = SOURCE.replace('hyb.phase = "npu"', 'hyb.phase = "npu", hyb.npu_kernel = "packet"')
    assert run('packet', packet) == run('packet_original', packet, bare=True)
    results['packet_kernel'] = {'byte_identical_ir': True}

    # Returning an input has no phase-local empty destination. Privatization
    # must neither invent a copy nor claim that a destination was changed.
    passthrough = '''module {
      func.func private @npu_phase(%input: tensor<32xf16>) -> tensor<32xf16> attributes {hyb.phase = "npu"} {
        return %input : tensor<32xf16>
      }
    }'''
    passthrough_ir = run('returned_input', passthrough)
    assert 'hyb.activation_residency_private_results = 0' in passthrough_ir
    assert 'tensor.empty' not in passthrough_ir
    results['returned_input'] = {'private_results': 0, 'no_copy': True}

    opaque = '''module {
      func.func private @opaque() -> tensor<32xf16>
      func.func private @npu_phase() -> tensor<32xf16> attributes {hyb.phase = "npu"} {
        %result = func.call @opaque() : () -> tensor<32xf16>
        return %result : tensor<32xf16>
      }
    }'''
    opaque_ir = run('untied_producer', opaque)
    assert 'hyb.activation_residency_private_results = 0' in opaque_ir
    assert 'tensor.empty' not in opaque_ir
    results['untied_producer'] = {'private_results': 0, 'no_copy': True}

    multiblock = '''module {
      func.func private @npu_phase(%input: tensor<32xf16>) -> tensor<32xf16> attributes {hyb.phase = "npu"} {
        cf.br ^exit
      ^exit:
        return %input : tensor<32xf16>
      }
    }'''
    assert run('multiple_blocks', multiblock) == run('multiple_blocks_original', multiblock, bare=True)
    results['multiple_blocks'] = {'byte_identical_ir': True}

    reshaped = SOURCE.replace('return %result : tensor<32xf16>',
        '%shaped = tensor.expand_shape %result [[0,1]] output_shape [1,32] : tensor<32xf16> into tensor<1x32xf16>\n    return %shaped : tensor<1x32xf16>')
    reshaped = reshaped.replace('@npu_phase() -> tensor<32xf16>', '@npu_phase() -> tensor<1x32xf16>')
    # The test phase's caller is not needed for destination tracing.
    reshaped = reshaped[:reshaped.index('  func.func @decode_step')] + '}\n'
    reshaped_ir = run('reshaped_result', reshaped)
    assert 'hyb.activation_residency_private_results = 1' in reshaped_ir
    results['reshape'] = {'private_results': 1}

    # Check the actual storage consequence, not only the pass's annotation.
    # The private exported output remains a DRAM out-parameter, while the
    # original destination becomes a shared L2 allocation for the four cores.
    pipeline = ('--one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map',
                '--buffer-results-to-out-params=hoist-static-allocs',
                '--canonicalize', '--cse', '--hyb-npu-place', '--hyb-npu-promote')
    before_mem = run('before_bufferization', enabled=False, extra=pipeline)
    after_mem = run('after_bufferization', extra=pipeline)
    assert 'memref<32xf16, 2>' in after_mem, 'local shared activation was not placed in L2'
    # The initial fill must now write an on-chip value, never the DRAM output.
    fill_line = next(line for line in after_mem.splitlines() if 'linalg.fill' in line)
    assert 'memref<32xf16, 1>' in fill_line, fill_line
    # Prior to the forall, old code flushes the initial fill to the DRAM output;
    # the new version flushes it to shared L2 instead.
    before_prefix = before_mem.split('scf.forall',1)[0]
    after_prefix = after_mem.split('scf.forall',1)[0]
    dram_copy = r'memref.copy[^\n]+to memref<32xf16>'
    assert re.search(dram_copy, before_prefix), before_prefix
    assert not re.search(dram_copy, after_prefix), after_prefix
    assert re.search(r'memref.copy[^\n]+to memref<32xf16, 2>', after_prefix), after_prefix
    results['bufferized_residency'] = {'local_dram_flush_removed': True, 'shared_l2_used': True}

    # Policy errors must not silently fall back to the unoptimized program.
    for name, bad in (('integer', 1), ('string', 'true'), ('null', None)):
        target = json.loads((out/'True.json').read_text())
        target['npu']['activation_residency'] = bad
        path = out/f'invalid-{name}.json'; path.write_text(json.dumps(target))
        result = subprocess.run([optimizer, str(out/'original.mlir'),
            f'--hyb-npu-activation-residency=spec={path}'], capture_output=True, text=True)
        assert result.returncode and 'npu.activation_residency must be a boolean' in result.stderr
        results[f'invalid_{name}'] = {'rejected': True}
    path = out/'array.json'; path.write_text('[]')
    result = subprocess.run([optimizer, str(out/'original.mlir'),
        f'--hyb-npu-activation-residency=spec={path}'], capture_output=True, text=True)
    assert result.returncode and 'target spec must be a JSON object' in result.stderr
    results['invalid_root_array'] = {'rejected': True}

    (out/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__ == '__main__':
    main()
