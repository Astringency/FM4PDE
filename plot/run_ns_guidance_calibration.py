"""Separate, validation-only NS guidance calibration and paired evaluation."""
from __future__ import annotations
import argparse,copy,fcntl,json,os,random,socket,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from run_ns_loss_study import sha,write,numerical_failure


def candidates():
    pairs=[(1.,1.)]+[(o,p) for o in [.1,.3,1.,3.,10.] for p in [0.,1.,1000.] if (o,p)!=(1.,1.)]
    return [dict(name=f'obs{o:g}_pde{p:g}',observation_multiplier=o,pde_multiplier=p) for o,p in pairs]


def freeze(args):
    source=json.loads((args.inputs/'source.json').read_text())
    assert source['pilot_ids']==[251,878,197,364] and len(source['evaluation_ids'])==32
    assert not (args.output/'protocol.json').exists()
    write(args.output/'protocol.json',dict(source_sha256=sha(args.inputs/'source.json'),
        fields_sha256=sha(args.inputs/'fields_masks.npz'),checkpoint_sha256=sha(args.checkpoint),
        code_sha256={p:sha(ROOT/p) for p in ['plot/run_ns_guidance_calibration.py','plot/run_matched_timing.py','plot/ns_loss_exchange.py','sampling/guidance.py','sampling/losses.py','sampling/pde_residuals.py']},
        calibration_ids=source['pilot_ids'],evaluation_ids=source['evaluation_ids'],seeds=[0,1,2],
        tasks=['forward','inverse','both'],candidates=candidates(),steps=100,
        calibration_calls=540,evaluation_calls=576,
        selection='Mean primary error over four calibration inputs after seed averaging. Primary: u for forward, a for inverse, per-call max(a,u) for joint. Any failed call disqualifies candidate. Exact ties favor original then fixed candidate order.',
        scope='Exploratory four-input calibration; original endpoint-secant MSE, sampler, gate, clipping and checkpoint fixed. No evaluation target or metric enters weight selection. Separate from the original loss-exchange study.'))


def select(args,protocol,ph):
    lock=(args.output/'selection.lock').open('a+')
    with lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path=args.output/'selection.json'
        if path.exists():
            s=json.loads(path.read_text());assert s['protocol_sha256']==ph;return s
        rows=[];selected={}
        for task in protocol['tasks']:
            choices=[]
            for order,c in enumerate(protocol['candidates']):
                rr=[]
                for i in protocol['calibration_ids']:
                    for seed in protocol['seeds']:
                        p=args.output/'calibration'/task/c['name']/f'sample{i}_seed{seed}.json'
                        r=json.loads(p.read_text());assert r['protocol_sha256']==ph and r['sample_id']==i and r['seed']==seed
                        if r['status']=='complete':assert sha(p.with_suffix('.pt'))==r['prediction_sha256']
                        rr.append(r)
                finite=[r for r in rr if r['status']=='complete']
                per_input=[np.mean([r['primary_error'] for r in finite if r['sample_id']==i]) for i in protocol['calibration_ids']] if len(finite)==12 else None
                row=dict(task=task,**c,calls=12,finite_calls=len(finite),eligible=len(finite)==12,
                    mean_primary=float(np.mean(per_input)) if per_input is not None else None,
                    sd_input_primary=float(np.std(per_input,ddof=1)) if per_input is not None else None)
                rows.append(row)
                if row['eligible']:choices.append((row['mean_primary'],order,c))
            assert choices, f'No stable candidate for {task}; retained all outcomes.'
            selected[task]=min(choices,key=lambda x:(x[0],x[1]))[2]
        s=dict(protocol_sha256=ph,selected=selected,calibration_calls_verified=540,calibration_summary=rows)
        write(path,s);return s


def worker(args):
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    from sampling.config import AblationConfig
    from sampling.data import PDEGroundTruth
    from sampling.masks import PairMasks
    from sampling.model_io import load_fm4pde_checkpoint_bundle
    from run_matched_timing import fm_predict
    from ns_loss_exchange import fm_exchange
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    protocol=json.loads((args.output/'protocol.json').read_text());ph=sha(args.output/'protocol.json')
    assert sha(args.inputs/'source.json')==protocol['source_sha256'] and sha(args.inputs/'fields_masks.npz')==protocol['fields_sha256']
    assert sha(args.checkpoint)==protocol['checkpoint_sha256']
    for name,h in protocol['code_sha256'].items():assert sha(ROOT/name)==h,name
    lock=(args.output/f'worker{args.shard}.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    uuid=str(torch.cuda.get_device_properties(0).uuid)
    if not uuid.startswith('GPU-'):uuid='GPU-'+uuid
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    assert not [r for r in apps.splitlines() if r.startswith(uuid) and int(r.split(',')[1])!=os.getpid()]
    write(args.output/f'environment_{args.shard}.json',dict(protocol_sha256=ph,pid=os.getpid(),host=socket.gethostname(),
        torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),uuid=uuid,tf32=False,steps=100,batch_size=1))
    source=json.loads((args.inputs/'source.json').read_text());data=np.load(args.inputs/'fields_masks.npz')
    model=load_fm4pde_checkpoint_bundle(str(args.checkpoint),'nsnonbounded','cuda:0',model_profile='recommended')
    count={'n':0}
    def hook(*unused):count['n']+=1
    handle=model[0].model.register_forward_hook(hook)
    def predict(task,c,i,seed,hidden=False,scale=1.):
        a=torch.from_numpy(data[f'a_{i}']).to('cuda:0',torch.float32);u=torch.from_numpy(data[f'u_{i}']).to('cuda:0',torch.float32)
        ma=torch.from_numpy(data[f'mask_a_{i}']).to('cuda:0');mu=torch.from_numpy(data[f'mask_u_{i}']).to('cuda:0')
        if task=='both':mu=ma.clone()
        if task=='forward':mu=torch.zeros_like(mu)
        if task=='inverse':ma=torch.zeros_like(ma)
        aa=a*ma*scale;uu=u*mu*scale
        if hidden:aa=aa+2.7*(1-ma);uu=uu-4.1*(1-mu)
        masks=PairMasks(ma,mu,dict(source='frozen common observations'))
        conf=copy.deepcopy(source['fm_configs'][task])
        conf.update(device='cuda:0',sample_seed=seed,batch_size=1,offset=i,num_steps=100,
            save_plots=False,save_intermediate=False,save_per_sample_curves=False,checkpoint_path=str(args.checkpoint),
            output_dir=str(args.output/f'reference_{args.shard}'),initial_noise_source_indices=[],initial_noise_source_batch_size=None)
        for k in ['zeta_obs_a','zeta_obs_u']:conf[k]*=c['observation_multiplier']
        conf['zeta_pde']*=c['pde_multiplier'];cfg=AblationConfig(**conf)
        params={k:torch.tensor([v],device='cuda:0',dtype=torch.float32) for k,v in source['pde_params'].items()}
        gt=PDEGroundTruth('nsnonbounded',aa,uu,torch.cat([aa,uu],1),params,['w0'],['wT'],
            dict(sample_ids=[str(i)],sample_offsets=[i],offset=i,batch_size=1,synthetic=False,endpoint_pair=True,data_path=source['data_path']))
        trace=[];count['n']=0
        with fm_exchange(False,trace):pred=fm_predict(cfg,model,gt,masks)
        return pred,dict(truth=[a,u],masks=[ma,mu],trace=trace,config=cfg,gt=gt,pair_masks=masks)
    pilot=args.output/f'implementation_check_{args.shard}.json'
    if not pilot.exists():
        checks=[];i=protocol['calibration_ids'][args.shard]
        for c in [protocol['candidates'][0],next(x for x in protocol['candidates'] if x['observation_multiplier']==.1 and x['pde_multiplier']==1000)]:
            p,d=predict('both',c,i,0)
            repeat,_=predict('both',c,i,0);hidden,_=predict('both',c,i,0,hidden=True);changed,_=predict('both',c,i,0,scale=.83)
            delta=lambda q:max(float((x-y).abs().max()) for x,y in zip(p,q))
            row=dict(candidate=c['name'],sample_id=i,repeat=delta(repeat),hidden=delta(hidden),sensitivity=delta(changed))
            assert row['repeat']==row['hidden']==0 and row['sensitivity']>0
            if c['name']==protocol['candidates'][0]['name']:
                import sampling.runner as runner
                receipt=runner.run_single_ablation(d['config'],model,ground_truth=d['gt'],observation_masks=d['pair_masks'])
                ref=torch.load(Path(receipt['run_dir'])/'result.pt',map_location='cuda:0',weights_only=False)
                row['original_runner']=delta([ref['coef_final'],ref['sol_final']]);assert row['original_runner']==0
            checks.append(row)
        write(pilot,dict(status='pass',protocol_sha256=ph,checks=checks))
    def calls(stage,jobs,selection_sha=None):
        for task,label,c,i,seed in jobs:
            dest=args.output/stage/task/label/f'sample{i}_seed{seed}';rpath=dest.with_suffix('.json')
            if rpath.exists():
                old=json.loads(rpath.read_text());assert old['protocol_sha256']==ph
                if selection_sha:assert old['selection_sha256']==selection_sha
                continue
            torch.cuda.synchronize();start=time.perf_counter();count['n']=0
            r=dict(stage=stage,task=task,variant=label,candidate=c,sample_id=i,seed=seed,protocol_sha256=ph,
                   selection_sha256=selection_sha)
            try:
                pred,d=predict(task,c,i,seed);torch.cuda.synchronize()
                assert count['n']==100
                finite=all(bool(torch.isfinite(x).all()) for x in pred)
                errors=[float((x.double()-y.double()).norm()/y.double().norm()) for x,y in zip(pred,d['truth'])] if finite else [None,None]
                r.update(status='complete' if finite else 'nonfinite',seconds=time.perf_counter()-start,nfe=count['n'],
                    rel_l2_a=errors[0],rel_l2_u=errors[1],primary_error=(errors[1] if task=='forward' else errors[0] if task=='inverse' else max(errors)) if finite else None)
                dest.parent.mkdir(parents=True,exist_ok=True);tensor=dest.with_suffix('.pt')
                torch.save(dict(prediction=[x.detach().cpu() for x in pred],truth=[x.cpu() for x in d['truth']],masks=[x.cpu() for x in d['masks']],trace=d['trace'],receipt=r),tensor)
                r['prediction_sha256']=sha(tensor)
            except (RuntimeError,FloatingPointError) as exc:
                if not numerical_failure(exc):raise
                r.update(status='unstable',error=repr(exc),seconds=time.perf_counter()-start,nfe=count['n'])
            write(rpath,r);print('RESULT',stage,task,label,i,seed,r['status'],r.get('primary_error'),flush=True)
        write(args.output/f'{stage}_complete_{args.shard}.json',dict(protocol_sha256=ph,calls=len(jobs)))
    jobs=[(t,c['name'],c,i,seed) for i in protocol['calibration_ids'] for seed in protocol['seeds'] for t in protocol['tasks'] for c in protocol['candidates']]
    random.Random(20260909).shuffle(jobs);calls('calibration',jobs[args.shard::2])
    while not all((args.output/f'calibration_complete_{j}.json').exists() for j in [0,1]):time.sleep(10)
    chosen=select(args,protocol,ph);sh=sha(args.output/'selection.json')
    jobs=[(t,label,c,i,seed) for i in protocol['evaluation_ids'][args.shard::2] for seed in protocol['seeds'] for t in protocol['tasks']
        for label,c in [('reference',protocol['candidates'][0]),('selected',chosen['selected'][t])]]
    random.Random(20260910+args.shard).shuffle(jobs);calls('evaluation',jobs,sh)
    handle.remove()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['freeze','worker'])
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--shard',type=int,choices=[0,1],default=0)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    freeze(args) if args.mode=='freeze' else worker(args)


if __name__=='__main__':main()
