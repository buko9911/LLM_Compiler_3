#!/usr/bin/env python3
"""Effect-based transfer scheduling: dependency hazards and conservative boundaries."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import tempfile


def source(body, early_target=False, copy_source='%gamma', phase='npu'):
    target = '%dst = memref.alloc() : memref<16xf16, 1>'
    return f'''module {{
  func.func @phase(%src: memref<16xf16>, %gamma: memref<16xf16>, %other: memref<16xf16>)
      attributes {{hyb.phase = "{phase}"}} {{
    %c0 = arith.constant 0 : index
    %yes = arith.constant true
    %one = arith.constant 1.0 : f16
    %zero32 = arith.constant 0.0 : f32
    %x = memref.alloc() : memref<16xf16, 1>
    %out = memref.alloc() : memref<16xf16, 1>
    {target if early_target else ''}
    memref.copy %src, %x : memref<16xf16> to memref<16xf16, 1>
    {body}
    {'' if early_target else target}
    memref.copy {copy_source}, %dst : memref<16xf16> to memref<16xf16, 1>
    return
  }}
}}'''

ELEMENTWISE = '''linalg.generic {indexing_maps=[affine_map<(i)->(i)>,affine_map<(i)->(i)>],iterator_types=["parallel"]}
  ins(%x:memref<16xf16,1>) outs(%out:memref<16xf16,1>) {
  ^bb0(%a:f16,%b:f16):
    %v = arith.addf %a,%a:f16
    linalg.yield %v:f16
  }'''
REDUCTION = '''%sum = memref.alloc() : memref<f32, 1>
  linalg.fill ins(%zero32:f32) outs(%sum:memref<f32,1>)
  linalg.generic {indexing_maps=[affine_map<(i)->(i)>,affine_map<(i)->()>],iterator_types=["reduction"]}
  ins(%x:memref<16xf16,1>) outs(%sum:memref<f32,1>) {
  ^bb0(%a:f16,%acc:f32):
    %f = arith.extf %a:f16 to f32
    %v = arith.addf %f,%acc:f32
    linalg.yield %v:f32
  }'''
MATMUL = '''%a = memref.alloc():memref<2x4xf16,1>
  %b = memref.alloc():memref<4x4xf16,1>
  %c = memref.alloc():memref<2x4xf16,1>
  linalg.matmul ins(%a,%b:memref<2x4xf16,1>,memref<4x4xf16,1>) outs(%c:memref<2x4xf16,1>)'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    root=Path(__file__).resolve().parents[2]
    parser.add_argument('--opt',type=Path,default=root/'build/bin/npu-opt')
    args=parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='npu-transfer-schedule-') as directory:
        work=Path(directory);spec=work/'target.json'
        config=json.loads((root/'configs/target.json').read_text());config['npu']['transfer_schedule']=True
        spec.write_text(json.dumps(config))
        dynamic = source(ELEMENTWISE).replace('%c0 = arith.constant 0 : index', '%c0 = arith.constant 0 : index\n    %c16 = arith.constant 16 : index').replace('%dst = memref.alloc() : memref<16xf16, 1>', '%dst = memref.alloc(%c16) : memref<?xf16, 1>').replace('memref.copy %gamma, %dst : memref<16xf16> to memref<16xf16, 1>', 'memref.copy %gamma, %dst : memref<16xf16> to memref<?xf16, 1>')
        cases=[
          ('elementwise',source(ELEMENTWISE),True,'linalg.generic'),
          ('reduction',source(REDUCTION),True,'linalg.generic'),
          ('matmul',source(MATMUL),True,'linalg.matmul'),
          ('source_read_read',source('%v = memref.load %gamma[%c0] : memref<16xf16>'),True,'memref.load'),
          ('source_write_RAW',source('memref.store %one,%gamma[%c0] : memref<16xf16>'),False,''),
          ('target_read_WAR',source('%v = memref.load %dst[%c0] : memref<16xf16,1>',True),False,''),
          ('target_write_WAW',source('memref.store %one,%dst[%c0] : memref<16xf16,1>',True),False,''),
          ('may_alias_argument',source('memref.store %one,%other[%c0] : memref<16xf16>'),False,''),
          ('source_free',source('memref.dealloc %gamma : memref<16xf16>'),False,''),
          ('view_alias_target',source('%view=memref.subview %dst[0][16][1]:memref<16xf16,1> to memref<16xf16,1>\nmemref.store %one,%view[%c0]:memref<16xf16,1>',True),False,''),
          ('unknown_effects',source('"test.unknown"(%x) : (memref<16xf16,1>)->()'),False,''),
          ('control_region',source('scf.if %yes { memref.store %one,%out[%c0]:memref<16xf16,1> }'),False,''),
          ('unknown_nested_effect',source(ELEMENTWISE.replace('%v = arith.addf','"test.hidden"() : ()->()\n    %v = arith.addf')),False,''),
          ('SSA_producer',source(ELEMENTWISE+'\n%view=memref.subview %gamma[0][16][1]:memref<16xf16> to memref<16xf16>',copy_source='%view'),False,''),
          ('non_npu',source(ELEMENTWISE,phase='pim'),False,''),
          ('dynamic_target',dynamic,False,''),
          ('owned_pipeline',source(ELEMENTWISE).replace('memref.copy %gamma, %dst :', 'memref.copy %gamma, %dst {hyb.input_pipeline_copy} :'),False,''),
          ('scheduled_barrier',source(ELEMENTWISE.replace('linalg.generic {', 'linalg.generic {hyb.schedule_barrier,')),False,''),
        ]
        for name,text,moves,compute in cases:
            path=work/f'{name}.mlir';path.write_text(text)
            command=[str(args.opt),str(path),'--allow-unregistered-dialect',f'--hyb-npu-transfer-schedule=spec={spec}','--verify-each']
            result=subprocess.run(command,text=True,capture_output=True)
            assert result.returncode==0,(name,result.stderr)
            output=result.stdout
            origins=list(map(int,re.findall(r'hyb\.schedule_origin = (\d+)',output)))
            if name != 'non_npu':
                count=int(re.search(r'hyb\.schedule_origin_count = (\d+)',output).group(1))
                assert sorted(origins)==list(range(count)),(name,'origin mapping is not 1:1',origins)
            assert ('hyb.transfer_scheduled' in output)==moves,(name,output)
            if moves:
                assert output.index('hyb.transfer_scheduled')<output.index(compute),(name,output)
                # Input transfer order stays intact: the initial X copy is still
                # before the newly hoisted gamma copy, even across pure ops.
                assert output.index('memref.copy')<output.index('hyb.transfer_scheduled')
            # Structural verification after the scheduling mutation proves SSA
            # dominance, including the target allocation moved with the copy.
            print(name,'PASS',flush=True)
        config['npu']['transfer_schedule']=False;spec.write_text(json.dumps(config))
        path=work/'off.mlir';path.write_text(source(REDUCTION))
        before=subprocess.check_output([str(args.opt),str(path)],text=True)
        after=subprocess.check_output([str(args.opt),str(path),f'--hyb-npu-transfer-schedule=spec={spec}'],text=True)
        assert before==after,'OFF changed IR'
        print('disabled_identity PASS')

if __name__=='__main__':main()
