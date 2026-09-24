"""Compare U-Net and official OFM priors with the appendix OFM guidance."""
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import yaml
from experiments.paper.run import ROOT, digest, write_json


def run(spec,args):
    import torch
    from sampling.config import load_config
    from sampling.data import load_ground_truth
    from sampling.runner import run_single_ablation
    baseline=Path(os.environ.get('GENERATIVE_BASELINE_ROOT',ROOT.parent/'FunDPS_DDIS_ECI_OFM'))
    sys.path.insert(0,str(baseline/'adapters'))
    from shared_prior_runtime import load_prior, common_masks, native_noise_provider
    compact_root=Path(os.environ['OFM_DATA_ROOT'])
    checkpoint_root=Path(os.environ['OFM_CHECKPOINT_ROOT'])
    parameters=yaml.safe_load((ROOT/'configs/ofm_guidance.yaml').read_text())
    records=[]
    torch.set_num_threads(args.threads)
    for pde in spec['pdes']:
        if args.pdes and pde not in args.pdes:continue
        for prior in ['fm4pde','ofm']:
            task0='both' if pde=='burger' else 'forward'
            reference=load_config(ROOT/f'configs/main/{task0}/{pde}.yaml')
            weight=Path(reference.checkpoint_path) if prior=='fm4pde' else Path(os.environ.get('OFM_CHECKPOINT_'+pde.upper(),checkpoint_root/pde/'best.pt'))
            options=SimpleNamespace(fm4pde=ROOT,ofm=baseline/'official/OFM',source=compact_root/pde,
                                    checkpoint=weight,device=args.device,pde=pde,prior=prior)
            net,normalizer,payload,noise,_=load_prior(options,1 if pde=='burger' else 2)
            for task,values in parameters[pde].items():
                weights=dict(zip(['zeta_obs_a','zeta_obs_u','zeta_pde','clip_threshold'],values))
                for dist in spec['distributions']:
                    for i in range(min(spec['count'],args.limit or spec['count'])):
                        directory=args.output/prior/pde/task/dist/f'{i:04d}'
                        config=load_config(ROOT/f'configs/main/{task}/{pde}.yaml',dict(weights,
                            device=args.device,test_type=dist,offset=i,sample_seed=i,mask_seed=0,
                            checkpoint_path=str(weight),output_dir=str(directory),model_profile='auto',
                            num_steps=100,batch_size=1,save_plots=False))
                        truth=load_ground_truth(config)
                        masks,_=common_masks(truth,i,seed=0,task='inverse' if pde=='burger' else task,count=100)
                        with native_noise_provider(noise,enabled=prior=='ofm'):
                            result=run_single_ablation(config,checkpoint_bundle=(net,normalizer,payload),
                                                       ground_truth=truth,observation_masks=masks)
                        if result['status']!='ok':raise RuntimeError(result)
                        records.append(dict(prior=prior,pde=pde,task=task,distribution=dist,index=i,
                                            result=str(Path(result['run_dir'])/'result.pt'),checkpoint_sha256=digest(weight)))
                        write_json(args.output/'index.json',records)
            del net,normalizer,payload,noise
            if torch.cuda.is_available():torch.cuda.empty_cache()
