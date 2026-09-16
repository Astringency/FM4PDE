"""One direct, unguided ODE sample and runtime profile per pretrained model."""
from __future__ import annotations
import argparse,hashlib,json,os,socket,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sampling.config import AblationConfig
from sampling.model_io import load_fm4pde_checkpoint_bundle
from sampling.runner import _scalar_conditioning_for_sampling
from sampling.sampler_wrappers import _call_velocity_model
from sampling.state import standardized_to_physical_state
from sampling.batching import combine_truths
from run_ablation_study import digest,write,PDES

def main(args):
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    args.output.mkdir(parents=True,exist_ok=True)
    results=[]
    for name in PDES+['ns_main']:
        pde='nsnonbounded' if name=='ns_main' else name
        source=args.inputs/pde
        protocol=json.loads((source/'protocol.json').read_text())
        weight=args.ns_main_weight if name=='ns_main' else source/'weights.pth'
        if name!='ns_main':assert digest(weight)==protocol['weights_sha256']
        target=args.output/name;target.mkdir(exist_ok=True)
        receipt=target/'profile.json'
        if receipt.exists():
            result=json.loads(receipt.read_text());assert digest(target/'sample.pt')==result['sample_sha256']
            results.append(result);continue
        free,_=torch.cuda.mem_get_info();assert free>40*2**30
        cfg=AblationConfig(**dict(protocol['base_configs']['both'],checkpoint_path=str(weight),
            device='cuda:0',batch_size=1,model_profile='auto',guidance_components='noguide'))
        net,normalizer,payload=load_fm4pde_checkpoint_bundle(str(weight),pde,'cuda:0',model_profile='auto')
        truths=torch.load(source/'truths.pt',map_location='cpu',weights_only=False)
        gt=combine_truths(truths,[0],'cuda:0')
        extra,conditioning=_scalar_conditioning_for_sampling(checkpoint_payload=payload,gt=gt,config=cfg,device='cuda:0')
        shape=tuple(gt.pair.shape);torch.manual_seed(20260910);torch.cuda.manual_seed_all(20260910)
        x=torch.randn(shape,device='cuda:0',dtype=torch.float32)
        noise_sha=hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest()
        start=time.monotonic();torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for k in range(100):
                t=torch.tensor(k/100,device=x.device,dtype=x.dtype)
                x=x+.01*_call_velocity_model(net,x,t,extra)
            fields=standardized_to_physical_state(x,pde,normalizer=normalizer)
        assert torch.isfinite(fields.coef).all() and torch.isfinite(fields.sol).all()
        torch.save(dict(coef=fields.coef.cpu(),sol=fields.sol.cpu(),
                        scalar_conditioning=conditioning),target/'sample.pt')
        model=getattr(net,'model',net)
        result=dict(name=name,pde=pde,checkpoint_path=str(weight),checkpoint_sha256=digest(weight),
            model_config=payload['model_config'],parameters=sum(p.numel() for p in model.parameters()),
            parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
            sample_sha256=digest(target/'sample.pt'),initial_noise_sha256=noise_sha,seed=20260910,
            guidance='none',sampler='direct uniform Euler ODE',steps=100,
            scalar_conditioning=conditioning,seconds=time.monotonic()-start,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            environment=dict(host=socket.gethostname(),python=sys.version,torch=torch.__version__,
                cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),device=os.environ.get('CUDA_VISIBLE_DEVICES')),
            script_sha256=digest(Path(__file__)))
        write(receipt,result);results.append(result)
        print('DONE',name,'parameters',result['parameters'],flush=True)
        del model,net,payload,truths,gt,fields,x;torch.cuda.empty_cache()
    write(args.output/'profiles.json',results)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--ns-main-weight',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
