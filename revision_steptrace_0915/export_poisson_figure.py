"""Export one existing complete Poisson guidance control, without inference."""
import argparse, copy, dataclasses, hashlib, json, pathlib, sys
import numpy as np
import torch

def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',type=pathlib.Path,required=True)
    ap.add_argument('--output',type=pathlib.Path,required=True)
    ap.add_argument('--fm-code',type=pathlib.Path,required=True)
    a=ap.parse_args();sys.path.insert(0,str(a.fm_code))
    from sampling.config import AblationConfig
    from sampling.data import PDEGroundTruth
    from sampling.masks import PairMasks
    from sampling.state import SplitState
    from sampling.losses import compute_guidance_losses
    torch.set_num_threads(2)
    arrays={};controls={};common=None;noise=None;mask_digest=None
    for name in ['noguide','pde_only','obs_only','obs_pde']:
        folder=a.source/f'results/poisson/evaluation/guidance_{name}/seed0/batch425'
        receipt=json.loads((folder/'receipt.json').read_text())
        assert receipt['sample_ids']==[425,215,830,756]
        path=next(folder.rglob('result.pt'));x=torch.load(path,map_location='cpu',weights_only=False)
        ca=x['coef_ground_truth'];cu=x['sol_ground_truth'];ma=x['masks']['coef'];mu=x['masks']['sol']
        if common is None:
            common=(ca,cu,ma,mu);noise=receipt['initial_noise_sha256'];mask_digest=receipt['mask_tensor_sha256']
            arrays.update(truth_a=ca[0,0].numpy(),truth_u=cu[0,0].numpy(),mask_a=ma[0,0].numpy(),mask_u=mu[0,0].numpy())
        else:
            assert all(torch.equal(v,w) for v,w in zip(common,(ca,cu,ma,mu)))
            assert noise==receipt['initial_noise_sha256'] and mask_digest==receipt['mask_tensor_sha256']
        cfg=x['config'];cfg=cfg if isinstance(cfg,dict) else vars(cfg)
        assert cfg['clip_threshold']==50 and cfg['num_steps']==100 and cfg['guidance_components']==name
        metrics={}
        for field,truth,mask in [('a',ca,ma),('u',cu,mu)]:
            prediction=x['coef_final' if field=='a' else 'sol_final'][:1]
            arrays[name+'_'+field]=prediction[0,0].numpy()
            pd=prediction.double();td=truth[:1].double();md=mask[:1].double()
            metrics['full_relative_l2_'+field+'_percent']=100*float((pd-td).norm()/td.norm())
            metrics['observed_relative_l2_'+field+'_percent']=100*float(((pd-td)*md).norm()/(td*md).norm())
            metrics['observed_mse_'+field]=float((((pd-td)*md).square().sum())/md.sum())
        known={f.name for f in dataclasses.fields(AblationConfig)}
        metric_cfg=AblationConfig(**{k:v for k,v in cfg.items() if k in known})
        metric_cfg.guidance_components='obs_pde';metric_cfg.zeta_pde=1;metric_cfg.batch_size=1
        truth=PDEGroundTruth('poisson',ca[:1],cu[:1],torch.cat([ca[:1],cu[:1]],1),copy.deepcopy(x['pde_params']),['a'],['u'],{'sample_ids':['425']})
        masks=PairMasks(ma[:1],mu[:1],{'source':'saved actual masks'})
        loss=compute_guidance_losses(SplitState(x['coef_final'][:1],x['sol_final'][:1]),truth,masks,metric_cfg)
        metrics['L_pde']=float(loss.L_pde)
        assert all(np.isfinite(v) for v in metrics.values())
        controls[name]={'rel_l2_a_percent':metrics['full_relative_l2_a_percent'],'rel_l2_u_percent':metrics['full_relative_l2_u_percent'],'L_obs_a':metrics['observed_mse_a'],'L_obs_u':metrics['observed_mse_u'],'L_pde':metrics['L_pde'],'source_prediction':str(path),'source_prediction_sha256':sha(path),'source_receipt':str(folder/'receipt.json'),'source_receipt_sha256':sha(folder/'receipt.json'),'config':cfg,'metrics':metrics,'pde_residual_metadata':loss.metadata}
    a.output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output/'poisson_id425_guidance.npz',**arrays)
    report={'status':'pass','physical_input':425,'batch_row':0,'source_batch_ids':[425,215,830,756],'controls':controls,'truth_and_masks_exact_across_four_controls':True,'initial_noise_sha256_all_four':noise,'mask_tensor_sha256_all_four':mask_digest,'num_observations_per_field':500,'units':'Physical fields; relative L2 in percent; observed MSE and PDE residual MSE in physical units.','metric_evaluator':'sampling.losses.compute_guidance_losses; same original Poisson residual and BC settings; evaluated independently on the extracted one example.','metric_evaluator_sha256':sha(a.fm_code/'sampling/losses.py'),'tensor_file_sha256':sha(a.output/'poisson_id425_guidance.npz')}
    (a.output/'poisson_id425_guidance.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({n:c['metrics'] for n,c in controls.items()},indent=2))

if __name__=='__main__':main()
