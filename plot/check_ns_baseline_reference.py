"""Diagnose archived baseline differences across batch and arithmetic modes."""
import argparse,copy,importlib,json,sys
from pathlib import Path
from run_ns_spectral_baselines import make_batch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights',type=Path,required=True);p.add_argument('--baseline-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0')
    args=p.parse_args();sys.path.insert(0,str(args.baseline_root))
    import torch
    torch.set_num_threads(2)
    results=[]
    for task in ['forward','inverse','both']:
        for method,clsname in [('recfno','RecFNOBaseline'),('senseiver','SenseiverBaseline'),('voronoicnn','VoronoiCNNBaseline')]:
            d=torch.load(args.weights/task/(method+'_checkpoint.pt'),map_location='cpu',weights_only=False)
            t=torch.load(args.weights/task/(method+'_template.pt'),map_location='cpu',weights_only=False)
            cls=getattr(importlib.import_module('baselines.methods.'+method),clsname)
            model=cls().build(d['config'],d['data_spec']);model.load_state_dict(d['state_dict'],strict=True)
            model.load_payload(d);model.to(args.device).eval()
            mask=t['mask'][None].float();obs=t['input_fields'][None].float()*mask
            b=make_batch(obs,mask,t,method,0,args.device)
            ref=t['prediction'][None].float()
            for batch_size in ([1,8,16] if method=='recfno' else [1]):
                batch=copy.deepcopy(b)
                def expand(x):
                    if torch.is_tensor(x) and x.ndim and x.shape[0]==1:return x.expand(batch_size,*x.shape[1:]).contiguous()
                    if isinstance(x,dict):return {k:expand(v) for k,v in x.items()}
                    return x
                for key,value in vars(batch).items():setattr(batch,key,expand(value))
                for precision,tf32 in [('highest',False),('highest',True),('high',True)]:
                    torch.set_float32_matmul_precision(precision);torch.backends.cudnn.allow_tf32=tf32
                    with torch.no_grad():pred=model.predict_physical(batch)[:1].detach().cpu()
                    r=dict(task=task,method=method,batch_size=batch_size,precision=precision,cudnn_tf32=tf32,
                           relative_difference=float((pred-ref).norm()/ref.norm()),max_difference=float((pred-ref).abs().max()))
                    results.append(r);print(json.dumps(r),flush=True)
            del model
    args.output.write_text(json.dumps(dict(torch=torch.__version__,device=args.device,results=results),indent=2)+'\n')


if __name__=='__main__':main()
