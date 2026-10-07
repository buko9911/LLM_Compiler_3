#!/usr/bin/env python3
"""PIM M8 export -> M4 import: immutable PA/bytes, exact outputs, fail-closed imports.

Writes only a new --output directory. Native runs are sequential; no model
download or change to a resident model/checkpoint is involved.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
from safetensors.numpy import save_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'runtime'))
from decode import Device, phase_times  # noqa: E402
import workspace  # noqa: E402


class Digest:
    def __init__(self): self.h = hashlib.sha256()
    def integer(self, n): self.h.update(struct.pack('<Q', n))
    def text(self, s):
        b = s.encode(); self.integer(len(b)); self.h.update(b)
    def data(self, b): self.h.update(b)
    def final(self): return self.h.hexdigest()


def seal(s):
    h = Digest()
    for key in ['schema', 'mapping', 'quantization']: h.text(s[key])
    for key in ['channels', 'partitions', 'rank_bytes', 'packet_bytes', 'pus', 'lanes',
                'kernel_row', 'bankset_pingpong_lead', 'capacity_bytes']: h.integer(s[key])
    h.integer(len(s['weights']))
    for w in s['weights']:
        h.text(w['name'])
        for key in ['n', 'k', 'n_pad', 'scale_address', 'scale_bytes']: h.integer(w[key])
        h.integer(len(w['stream_part']))
        for p in w['stream_part']: h.integer(p)
        for b in w['stream_weight_base']: h.integer(b)
    s['layout_sha256'] = h.final()
    h = Digest(); h.text('lp6-pim-weight-manifest-digest-v1'); h.text(s['layout_sha256'])
    for w in s['weights']:
        for key in ['name', 'source_dtype', 'source_sha256', 'packet_sha256', 'scale_sha256']: h.text(w[key])
    s['manifest_sha256'] = h.final()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while b := f.read(1 << 20): h.update(b)
    return h.hexdigest()


def save(path, data): path.write_text(json.dumps(data, indent=2) + '\n')


def fixture(directory, m, model):
    directory.mkdir(parents=True)
    k, n = 512, 4096
    source = directory / 'step.mlir'
    source.write_text(f'''module {{
  func.func @decode_step(%x: tensor<{m}x{k}xf32>, %w: tensor<{n}x{k}xf32>) -> tensor<{m}x{n}xf32> {{
    %zero = arith.constant 0.0 : f32
    %wt = tensor.empty() : tensor<{k}x{n}xf32>
    %t = linalg.transpose ins(%w : tensor<{n}x{k}xf32>) outs(%wt : tensor<{k}x{n}xf32>) permutation = [1, 0]
    %empty = tensor.empty() : tensor<{m}x{n}xf32>
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<{m}x{n}xf32>) -> tensor<{m}x{n}xf32>
    %y = linalg.matmul ins(%x, %t : tensor<{m}x{k}xf32>, tensor<{k}x{n}xf32>) outs(%init : tensor<{m}x{n}xf32>) -> tensor<{m}x{n}xf32>
    return %y : tensor<{m}x{n}xf32>
  }}
}}
''')
    manifest = directory / 'frontend.json'
    save(manifest, dict(model=str(model), tokens=m, capacity=1, mask_min=-30000.0,
         arguments=[dict(kind='input', name='x'), dict(kind='weight', name='proj.weight')],
         results=[dict(kind='output', name='y')]))
    return source, manifest


def audit_images(s, paths, wq, source_values):
    """Independent scalar PA/index computation and direct reads of real images."""
    w = s['weights'][0]; n, k = w['n'], w['k']; channels = s['channels']
    row_bytes, bank_bytes = channels*32768, channels*8192
    col_bytes, bg_bytes = channels*128, channels*32
    h = Digest(); h.text('lp6-pim-packet-v1')
    count = 0
    with paths[0].open('rb') as a, paths[1].open('rb') as b:
        for stream, part in enumerate(w['stream_part']):
            for sweep in range(k//256):
                for acc in range(16):
                    for j in range(32):
                        tile = w['stream_weight_base'][stream] + (sweep*16+acc)*32+j
                        for pu in range(8):
                            bank = 2*pu
                            address = (part//channels)*(1 << 33) + (s['kernel_row']+tile//64)*row_bytes
                            address += (bank%4)*bank_bytes + (tile%64)*col_bytes + (bank//4)*bg_bytes + (part%channels)*32
                            if s['bankset_pingpong_lead'] and (tile//64)%2: address += bank_bytes
                            expected = bytearray()
                            for lane in range(4):
                                output = pu*(w['n_pad']//8)+(stream*8+acc//2)*8+2*lane+acc%2
                                k0 = (sweep*32+j)*8
                                expected += wq[output,k0:k0+8].tobytes() if output<n else bytes(8)
                            a.seek(address); b.seek(address); av, bv = a.read(32), b.read(32)
                            assert av == bv == expected, ('packet bytes', address)
                            h.integer(address); h.data(av); count += 1
        assert h.final() == w['packet_sha256']
        a.seek(w['scale_address']); b.seek(w['scale_address'])
        scales = a.read(w['scale_bytes'])
        assert scales == b.read(w['scale_bytes']) == np.full(n,1/64,dtype='<f4').tobytes()
    h = Digest(); h.text('lp6-pim-scale-v1'); h.integer(w['scale_address']); h.integer(w['scale_bytes']); h.data(scales)
    assert h.final() == w['scale_sha256']
    h = Digest(); h.text('lp6-pim-source-v1'); h.text('proj.weight'); h.text('F16')
    h.integer(n); h.integer(k); h.data(source_values.tobytes())
    assert h.final() == w['source_sha256']
    return dict(packets_checked=count, weight_payload_bytes=count*32, scale_bytes=len(scales),
                all_packet_PA_payload_scale_bytes_identical=True, independent_source_digest_matches=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--compiler', type=Path, default=ROOT/'build/bin/npu-compile')
    ap.add_argument('--old-compiler', type=Path)
    ap.add_argument('--emulator', type=Path, required=True)
    ap.add_argument('--dramsim-library', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args(); out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    spec = json.loads((ROOT/'configs/target.json').read_text())
    # The hardware still has two ranks; this small fixture stores in rank 0,
    # avoiding an unused 8GiB sparse image prefix solely for test I/O.
    spec['pim']['partitions'] = 8
    spec_path = out/'target.json'; save(spec_path,spec)
    rng = np.random.default_rng(195121)
    wq = rng.integers(-127,128,(4096,512),dtype=np.int8); wq[:,0] = 127
    source_values = wq.astype(np.float16)/np.float16(64)
    model = out/'model'; model.mkdir()
    save_file({'proj.weight':source_values},model/'model.safetensors')
    sources = {m: fixture(out/f'input_m{m}',m,model) for m in [4,8]}
    logs = out/'logs'; logs.mkdir()
    result = dict(scope='same PA/bytes across separately launched PIM M8/M4 variants, not dynamic runtime switching',
                  provenance={str(p):sha(p) for p in [a.compiler,a.emulator,a.dramsim_library,Path(__file__)]},
                  negatives=[], native=[])
    if a.old_compiler: result['provenance'][str(a.old_compiler)] = sha(a.old_compiler)

    def compile_case(name,m,options=(),compiler=None,spec_override=None,expect=None,source_override=None):
        build = out/name
        source, frontend = sources[m]
        if source_override is not None: source = source_override
        argv = [str((compiler or a.compiler).resolve()),'--pipeline=decode',str(source),'--manifest',str(frontend),
                '--spec',str(spec_override or spec_path),'--force=pim','--debug-collect','-o',str(build),*map(str,options)]
        run = subprocess.run(argv,cwd=ROOT,text=True,capture_output=True)
        (logs/f'{name}.log').write_text(run.stdout+run.stderr)
        if expect is None:
            assert run.returncode==0,(name,run.returncode,run.stderr[-2500:])
        else:
            assert run.returncode!=0 and expect in run.stdout+run.stderr,(name,run.returncode,run.stderr[-2500:])
            assert not (build/'pim_weight_storage_verified.json').exists(),name
            result['negatives'].append(dict(name=name,diagnostic=expect,exit_code=run.returncode))
        return build

    current_plain = compile_case('plain_current',8)
    if a.old_compiler:
        old = compile_case('plain_old',8,compiler=a.old_compiler)
        result['default_before_after'] = {name:dict(before=sha(old/name),after=sha(current_plain/name))
                                         for name in ['program.bin','lp6.bin']}
        assert all(x['before']==x['after'] for x in result['default_before_after'].values())
    exported = out/'weight_storage.json'
    m8 = compile_case('m8_export',8,['--pim-weight-storage-export',exported])
    for name in ['program.bin','lp6.bin']: assert sha(m8/name)==sha(current_plain/name),(name,'export changed default bytes')
    m4 = compile_case('m4_import',4,['--pim-weight-storage-import',exported])
    storage = json.loads(exported.read_text())
    sealed = copy.deepcopy(storage); seal(sealed); assert sealed==storage,'independent manifest digest'
    assert json.loads((m4/'pim_weight_storage_verified.json').read_text())==storage
    result['storage_manifest'] = storage
    result['image_audit'] = audit_images(storage,[m8/'lp6.bin',m4/'lp6.bin'],wq,source_values)
    reused = compile_case('reuse_sidecar',8,['--pim-weight-storage-export',out/'reuse_export.json'])
    compile_case('reuse_sidecar',4,['--pim-weight-storage-import',reused/'pim_weight_storage_verified.json'])
    assert json.loads((reused/'pim_weight_storage_verified.json').read_text())==storage
    for name in ['program.bin','lp6.bin']: assert sha(reused/name)==sha(m4/name),(name,'same-directory sidecar import')
    result['same_directory_sidecar_import_identical']=True

    os.environ['LD_PRELOAD'] = str(a.dramsim_library.resolve())
    os.environ.pop('LP6PIM_BANKSET_PINGPONG',None)
    for m,build in [(8,m8),(4,m4)]:
        xq = rng.integers(-127,128,(m,512),dtype=np.int8); xq[:,0] = 127
        x = xq.astype(np.float16)/np.float16(64)
        expected_i32 = xq.astype(np.int32)@wq.astype(np.int32).T
        expected_f16 = (expected_i32.astype(np.float32)/4096).astype(np.float16)
        dev = Device(build,emulator=a.emulator,lp6_config=workspace.LP6_CONFIG)
        dev.write('x',x); outputs,report,wall = dev.run()
        assert outputs['y'].tobytes()==expected_f16.tobytes(),(m,'FP16 output')
        partials = [outputs[name] for name,meta in dev.outputs.items() if meta['layout']=='pim_out' and meta['dtype']=='i32']
        assert len(partials)==1 and partials[0].tobytes()==expected_i32.tobytes(),(m,'INT32 output')
        np.savez(build/'reference.npz',x=x,expected_i32=expected_i32,expected_f16=expected_f16,actual=outputs['y'])
        result['native'].append(dict(M=m,all_INT32_outputs_byte_exact=True,all_FP16_outputs_byte_exact=True,
                                     wall_seconds=wall,phase_times=phase_times(report),output_sha256=hashlib.sha256(outputs['y'].tobytes()).hexdigest()))
        save(out/'summary.json',result)
        print('native',m,'INT32/FP16 byte exact',flush=True)

    mutations = out/'negative_manifests'; mutations.mkdir()
    def bad(name,modify,needle,reseal=True):
        s = copy.deepcopy(storage); modify(s)
        if reseal: seal(s)
        path = mutations/f'{name}.json'; save(path,s)
        compile_case('negative_'+name,4,['--pim-weight-storage-import',path],expect=needle)
    bad('version',lambda s:s.update(schema='lp6-pim-weight-storage-v9'),'unsupported schema',False)
    bad('partial',lambda s:s.pop('kernel_row'),'missing/invalid manifest field',False)
    bad('shape',lambda s:s['weights'][0].update(n=4095,scale_bytes=4095*4),'projection name/shape differs')
    bad('K_tail',lambda s:s['weights'][0].update(k=513),'K is not a multiple of 256')
    bad('unknown_name',lambda s:s['weights'][0].update(name='other.weight'),'projection name/shape differs')
    bad('layout_hash',lambda s:s.update(layout_sha256='0'*64),'layout SHA256 mismatch',False)
    bad('manifest_hash',lambda s:s.update(manifest_sha256='0'*64),'manifest SHA256 mismatch',False)
    bad('packet_content_hash',lambda s:s['weights'][0].update(packet_sha256='0'*64),'packet SHA256 mismatch')
    bad('scale_content_hash',lambda s:s['weights'][0].update(scale_sha256='0'*64),'scale SHA256 mismatch')
    bad('tile_overlap',lambda s:s['weights'][0]['stream_part'].__setitem__(1,0),'overlapping weight tile intervals')
    bad('scale_overlap',lambda s:s['weights'][0].update(scale_address=0x10000),'overlap with mutable dense buffers')
    def shrink_row(s):
        w=s['weights'][0];row_bytes=s['channels']*32768
        s['kernel_row']=(w['scale_address']+w['scale_bytes']+row_bytes-1)//row_bytes
    bad('constant_overlap',shrink_row,'constant reserve/dense buffers overlap')
    bad('rank_overflow',lambda s:s.update(kernel_row=(1<<33)//(s['channels']*32768)-1),'weight rows exceed rank/capacity')
    changed = copy.deepcopy(spec);changed['pim']['partitions']=16
    changed_path=out/'different_hardware.json';save(changed_path,changed)
    compile_case('negative_hardware',4,['--pim-weight-storage-import',exported],spec_override=changed_path,expect='target hardware/layout differs')
    changed_model=out/'changed_model';changed_model.mkdir();changed_values=source_values.copy();changed_values[0,0]=-changed_values[0,0]
    save_file({'proj.weight':changed_values},changed_model/'model.safetensors')
    compile_case('negative_source_weight',4,['--pim-weight-storage-import',exported,'--model',changed_model],expect='source SHA256 mismatch')
    stale=out/'negative_stale_verified';stale.mkdir()
    save(stale/'pim_weight_storage_verified.json',storage)
    compile_case('negative_stale_verified',4,['--pim-weight-storage-import',exported,'--model',changed_model],expect='source SHA256 mismatch')
    compile_case('negative_no_image',4,['--pim-weight-storage-import',exported,'--no-image'],expect='incompatible with --no-image')
    for label,opts,needle in [
        ('no_image',['--pim-weight-storage-import',exported,'--no-image'],'incompatible with --no-image'),
        ('malformed',['--pim-weight-storage-import',mutations/'version.json'],'unsupported schema')]:
        stale=out/f'negative_stale_{label}';stale.mkdir()
        save(stale/'pim_weight_storage_verified.json',storage)
        compile_case(f'negative_stale_{label}',4,opts,expect=needle)
    injected=out/'injected_storage_attribute.mlir'
    injected.write_text(sources[4][0].read_text().replace('module {','module attributes {hyb.pim_weight_storage_requested = unit} {',1))
    compile_case('negative_ir_attribute',4,['--no-image'],expect='reserved PIM weight storage IR attributes',source_override=injected)
    compile_case('negative_npu',4,['--pim-weight-storage-import',exported,'--force=npu'],expect='requires at least one PIM projection')
    compile_case('negative_explicit_bankset',4,['--pim-weight-storage-import',exported,'--pim-bankset-pingpong=12'],expect='bankset layout differs')
    result['all_checks_passed']=True
    save(out/'summary.json',result)
    print(json.dumps(dict(native_cases=len(result['native']),negative_cases=len(result['negatives']),all_checks_passed=True)),flush=True)


if __name__=='__main__': main()
