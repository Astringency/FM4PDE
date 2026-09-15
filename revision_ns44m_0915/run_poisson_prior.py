"""One Poisson prior draw with the original direct Euler protocol and a new seed."""
import argparse, hashlib, json, os, pathlib, sys, time

def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=pathlib.Path,required=True)
    p.add_argument('--checkpoint-sha256',required=True)
    p.add_argument('--fm-code',type=pathlib.Path,required=True)
    p.add_argument('--output',type=pathlib.Path,required=True)
    p.add_argument('--seed',type=int,default=20260911)
    a=p.parse_args();assert sha(a.checkpoint)==a.checkpoint_sha256
    assert not a.output.exists(),'Choose a new output directory; existing results are preserved.'
    sys.path.insert(0,str(a.fm_code))
    import torch
    from sampling.model_io import load_fm4pde_checkpoint_bundle
    from sampling.sampler_wrappers import _call_velocity_model
    from sampling.state import standardized_to_physical_state
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,normalizer,payload=load_fm4pde_checkpoint_bundle(str(a.checkpoint),'poisson','cuda:0',model_profile='auto')
    # The recorded Poisson prior is unconditional, with no scalar conditioning.
    assert not payload['model_config'].get('scalar_conditioning',False)
    assert payload['model_config'].get('num_classes') is None
    shape=(1,2,128,128);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    x=torch.randn(shape,device='cuda:0',dtype=torch.float32)
    noise_sha=hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest()
    torch.cuda.synchronize();started=time.perf_counter()
    with torch.no_grad():
        for k in range(100):
            t=torch.tensor(k/100,device=x.device,dtype=x.dtype)
            x=x+.01*_call_velocity_model(model,x,t,None)
        fields=standardized_to_physical_state(x,'poisson',normalizer=normalizer)
    torch.cuda.synchronize();elapsed=time.perf_counter()-started
    assert torch.isfinite(fields.coef).all() and torch.isfinite(fields.sol).all()
    a.output.mkdir(parents=True)
    torch.save({'coef':fields.coef.cpu(),'sol':fields.sol.cpu(),'scalar_conditioning':None},a.output/'sample.pt')
    report={'status':'complete','pde':'poisson','seed':a.seed,'steps':100,'sampler':'direct uniform Euler ODE','guidance':'none','checkpoint':str(a.checkpoint),'checkpoint_sha256':a.checkpoint_sha256,'parameters':sum(p.numel() for p in model.model.parameters()),'initial_noise_sha256':noise_sha,'sample_sha256':sha(a.output/'sample.pt'),'elapsed_seconds':elapsed,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'script_sha256':sha(__file__),'change_from_existing_prior':'Only Gaussian-noise seed20260910 to20260911; same checkpoint, physical normalization and direct100-step Euler update.'}
    (a.output/'profile.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
