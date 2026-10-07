#!/usr/bin/env python3
"""Direct native NPU weights on PIM: one-hot selection plus timed 32-way NPU reduction."""
import argparse, json, re
from pathlib import Path
import numpy as np
import dual_view_experiment as e
isa=e.isa

class NativeView:
    pingpong=False
    k_splits=1
    def __init__(self,n): self.n=n
    def descriptor(self,**kw):
        return dict(schema='lp6.experimental.dual_view.v1',remap_latency_cycles=2,
                    remap_blocks_per_cycle=4,**kw,views=[dict(kind='native_partial',logical_base=e.O_BASE,
                    n=self.n,k=4096,tokens=1,tile_n=64,k_splits=1,n_order='interleaved',weight_row=16,pingpong=False)])

def packet(row,tile,pu,sc,odd=0):
    bank=2*pu+odd
    return (row+tile//64)*262144+(bank%4)*65536+(tile%64)*1024+(bank//4)*256+sc*32

def program(path,m,n):
    p=e.Program(); actrow=16+n//64
    zero=p.dma('zero.load',[],2<<20,4<<20,256<<10)
    prev=zero
    for token in range(m):
        xd=p.dma('input.load',[prev],e.X_BASE+token*4096,e.X_L2,4096)
        body=isa.ldma_i(0,0,e.X_L2,4096)+isa.ldma_i(0,0x40000,4<<20,256<<10)
        for r in range(8):
            for bs in range(2):
                for bg in range(4):
                    for sc in range(8):
                        for c in range(4):
                            src=bs*2048+bg*8+sc+c*32
                            dst=0x40000+(r*2+bs)*16384+bg*256+sc*32+c*8+r
                            body += [isa.c_set(0,16),isa.c_set(1,128),isa.c_set(2,1024)]
                            body += e.ldma_strided(9,dst,src,1)
        body+=isa.ldma_i(1,0x40000,5<<20,256<<10)
        packed=p.core('activation.pack',[xd],0,body)
        writes=[]
        for r in range(8):
            for bs in range(2):
                for parity in range(2):
                    writes.append(p.dma('activation.scatter',[packed],packet(actrow,r*32+bs*16,parity,0,bs),
                                        (5<<20)+(r*2+bs)*16384,16384,load=False))
        ready=p.barrier('activation.ready',writes)
        cmds=[]
        for j in range(n//64):
            for sc in range(8):
                def emit(op,**kw): return isa.pim_primitive(op,sc,**kw)
                cmds += [emit(isa.PIM_CLEAR,accumulator=r) for r in range(8)]
                for bs in range(2):
                    for r in range(8):
                        for chunk in (0,32):
                            a=r*128+bs*64+chunk
                            cmds += [emit(isa.PIM_LOAD_ACTIVATION,activation=a+i,bank_set=bs) for i in range(0,32,4)]
                            cmds += [emit(isa.PIM_GEMV_STREAM,weight=j*64+chunk,activation=a,accumulator=r,inputs=32,outputs=1,
                                          weight_input_stride=1,weight_output_stride=0,activation_input_stride=1,
                                          accumulator_output_stride=1,bank_set=bs)]
                cmds += [emit(isa.PIM_STORE,accumulator=r,output=j*8+r,bank_set=0) for r in range(8)]
        pd=p.add('pim.compute',lambda i:isa.pim_record(i,[ready],cmds))
        reads=[p.dma('output.gather',[pd],e.O_BASE+off,(2<<20)+off,min(524288,32*n*4-off)) for off in range(0,32*n*4,524288)]
        rd=p.barrier('output.ready',reads)
        done=[]; share=n//4
        for core in range(4):
            body=[isa.c_set(0,32),isa.c_set(1,n*4),isa.c_set(2,share*4)]
            body+=e.ldma_strided(4,0,(2<<20)+core*share*4,share*4)
            for start in range(0,share,16):
                body += [isa.c_set(3,min(16,share-start))]+e.constant(0,start*4)+[isa.v_mem(False,isa.I32,0,0,0)]
                for split in range(1,32):
                    body+=e.constant(1,(split*share+start)*4)+[isa.v_mem(False,isa.I32,1,1,0),isa.v_op(0,isa.I32,0,0,1)]
                body += [isa.v_mem(True,isa.I32,0,0,0)]
            body+=isa.ldma_i(1,0,(token*n+core*share)*4,share*4)
            done.append(p.core('npu.reduce',[rd],core,body))
        prev=p.barrier('token.done',done)
    p.finish(path,dict(channel=0,rank=0,row=16,start_column=0,bank_set=1,weight_bits=8,activation_bits=8,
                       scale_bits=0,input_register_count=8,output_register_count=8,subchannels=8))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--tokens',type=int,nargs='+',default=[1,2,4,8,16]);ap.add_argument('--synthetic',action='store_true')
    ap.add_argument('--n',type=int,default=14336);args=ap.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    settings=out/'settings.toml';settings.write_text(re.sub(r'(?m)^num_cores\s*=\s*\d+','num_cores = 4',(e.SIM.parent/'plena_settings.toml').read_text()))
    w=e.load_weight(Path('/home/jongjip/models/llama31-8b'),out,args.synthetic,args.n,4096);n,k=w.shape
    summary={}
    for m in args.tokens:
        path=out/f'm{m}';path.mkdir(exist_ok=False)
        x=np.random.default_rng(4096).integers(-127,128,(m,k),dtype=np.int8)
        expected=x.astype(np.int32)@w.astype(np.int32).T
        image=path/'lp6.bin'
        with image.open('wb') as f:
            f.truncate((16+n//64+16+(n//8+63)//64)*262144)
            f.seek(e.X_BASE);f.write(x.tobytes());f.seek(e.W_BASE);f.write(w.reshape(n//32,32,k).transpose(0,2,1).copy().tobytes())
        program(path,m,n)
        result=e.execute(path,image,settings,e.SIM/'target/release/transactional_emulator',expected,NativeView(n),
                         engine='native_pim',k=k,timeout=7200,window_bytes=8192,fair_windows=True)
        report=json.loads((path/'commands.json').read_text());mhz=report['npu_clock_mhz'];last=0
        phases=dict(input_scatter_us=0,pim_us=0,gather_us=0,reduce_us=0)
        for c in report['commands']:
            name=c['name']
            key=next((v for prefix,v in [('activation.ready','input_scatter_us'),('pim.compute','pim_us'),('output.ready','gather_us'),('token.done','reduce_us')] if name.startswith(prefix)),None)
            if key: phases[key]+=(c['end_cycle']-last)/mhz;last=c['end_cycle']
        result['phase_us']=phases;e.save(path/'result.json',result);summary[str(m)]=result
        e.save(out/'summary.json',summary);image.unlink()
if __name__=='__main__':main()
