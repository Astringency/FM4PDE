"""Bounded development -> doubled-budget check -> held-out validation.

Never launches the formal 1000 when convergence is unestablished. Chooses only
from convergence diagnostics and measured cost, never reconstruction errors.
If development fails, it records that failure without treating chain outputs
as posterior evidence. This controller does not claim that its finite tests
prove stationarity or that they establish improved reconstruction accuracy.
"""
import argparse,datetime,io,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from experiments.strict_chain_diagnostics import diagnose
from scripts.train.resume_study import file_sha,write


def read_run(out,name):
    folder=out/name
    spec=json.loads((folder/'protocol.json').read_text())
    state=torch.load(io.BytesIO((folder/'state.pt').read_bytes()),map_location='cpu',weights_only=False)
    trace=np.asarray(state['trace'])
    return dict(name=name,spec=spec,diagnostics=diagnose(trace),seconds=state['seconds']),trace


def run_worker(args,name,spec,gpu,keep,warmup,sid=1103,extend=None):
    out=args.output
    command=[sys.executable,'-m','experiments.run_fm_tilt_geometry','--source',str(args.source),
        '--pilot-inputs',str(args.pilot_inputs),'--output',str(out),
        '--geometry',str(args.geometry_by_hash[spec['geometry_sha256']]),
        '--name',name,'--force',spec['force'],'--force-steps',str(spec['force_steps']),
        '--force-scale',str(spec['force_scale']),'--warmup',str(warmup),'--keep',str(keep),
        '--id',str(sid),'--beta',str(spec['beta']),'--seed',str(spec['seed'])]
    if extend:
        command+=['--extend-from',str(out/extend/'state.pt')]
    # Every stage uses the already-probed four-chain batch, one worker per GPU.
    free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
    assert int(free.splitlines()[gpu])>12000,'Insufficient free GPU memory for the probed batch'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu))
    execution=out/'execution'
    write(execution/(name+'_launch.json'),dict(command=command,gpu=gpu,
        as_of=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    log=(execution/(name+'.log')).open('w')
    process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
    return process,log,name


def wait_workers(args,workers):
    while any(p.poll() is None for p,_,_ in workers):
        time.sleep(20)
    for p,log,name in workers:
        log.close()
        (args.output/'execution'/(name+'.exit')).write_text(str(p.returncode)+'\n')
        if p.returncode:
            raise RuntimeError(f'{name} exited with {p.returncode}; inspect its saved log')


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ['source','pilot-inputs','output']:
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    out=args.output
    assert out.is_absolute() and '/outputs/' in str(out)
    report=out/'convergence_decision.json'
    assert not report.exists()
    args.geometry_by_hash={file_sha(file):file for file in
        [out/'geometry/geometry.pt',out/'geometry_enriched/geometry.pt']}
    candidates=['reference_pcnl','enriched_fullgrad']
    policy=dict(candidates=candidates,selection='Passing diagnostics first, then minimum bulk ESS/second',
        thresholds=dict(rhat=1.01,bulk_ess=100,tail_ess=100),
        extension='Exactly twice the selected original retained budget, frozen proposal, no further adaptation',
        held_out_ids=[1500,1501,1502,1503],formal_1000='Not launched by this controller; requires validated method and measured full-cohort deployment',
        if_none_pass='Only one bounded extension, provided max R-hat<=1.2 and bulk ESS>=20. Otherwise stop development without posterior claims.',
        reconstruction_errors_used_for_selection=False)
    write(out/'supervisor_protocol.json',policy)
    print('WAITING_FOR_DEVELOPMENT',policy,flush=True)
    while not all((out/name/'complete.json').exists() for name in candidates):
        for name in candidates:
            exitfile=out/'execution'/(name+'.exit')
            if exitfile.exists() and exitfile.read_text().strip()!='0':
                raise RuntimeError(f'{name} failed; development is not complete')
        time.sleep(20)
    rows=[read_run(out,name)[0] for name in candidates]
    ready=[r for r in rows if r['diagnostics']['passed']]
    if not ready:
        ready=[r for r in rows if r['diagnostics'].get('max_rhat',999)<=1.2
            and r['diagnostics'].get('min_bulk_ess',0)>=20]
    if not ready:
        write(report,dict(passed=False,stage='development_failed',development=rows,
            reason='Neither candidate passed or qualified for the one bounded extension',formal_1000_completed=0))
        print('DEVELOPMENT_FAILED',flush=True);return
    chosen=max(ready,key=lambda r:r['diagnostics']['min_bulk_ess']/r['seconds'])
    spec=chosen['spec'];name=chosen['name']+'_double_keep'
    print('EXTENDING',chosen['name'],flush=True)
    wait_workers(args,[run_worker(args,name,spec,0,2*spec['keep'],0,extend=chosen['name'])])
    extended,trace=read_run(out,name)
    prefix=diagnose(trace[:spec['keep']])
    first,second=diagnose(trace[:spec['keep']]),diagnose(trace[spec['keep']:])
    max_shift=None;stable=False
    if 'mcse' in first and 'mcse' in second:
        mcse=np.sqrt(np.asarray(first['mcse'])**2+np.asarray(second['mcse'])**2)
        shift=abs(np.asarray(first['mean'])-np.asarray(second['mean']))/np.maximum(mcse,1e-300)
        max_shift=float(max(shift)) if np.isfinite(shift).all() else None
        stable=bool(np.all(shift<3.5))
    development_pass=bool(extended['diagnostics']['passed'] and prefix['passed'] and stable)
    if not development_pass:
        write(report,dict(passed=False,stage='doubled_budget_failed',development=rows,selected=chosen['name'],
            extension=extended,prefix_diagnostics=prefix,max_mean_shift_in_combined_mcse=max_shift,
            formal_1000_completed=0))
        print('DOUBLED_BUDGET_FAILED',flush=True);return
    # If extra development was needed, charge those updates to fresh warm-up.
    warmup=spec['warmup']+(0 if chosen['diagnostics']['passed'] else spec['keep'])
    held=[]
    for pair in [[1500,1501],[1502,1503]]:
        workers=[run_worker(args,f'validation_{sid}',spec,gpu,2*spec['keep'],warmup,sid=sid)
            for gpu,sid in enumerate(pair)]
        wait_workers(args,workers)
        for sid in pair:
            row,_=read_run(out,f'validation_{sid}');held.append(row)
        write(out/'validation_progress.json',dict(rows=held,formal_1000_completed=0))
        if not all(row['diagnostics']['passed'] for row in held):
            break
    passed=len(held)==4 and all(row['diagnostics']['passed'] for row in held)
    write(report,dict(passed=passed,stage='independent_validation_complete' if passed else 'independent_validation_failed',
        selected=chosen['name'],development=rows,extension=extended,prefix_diagnostics=prefix,
        max_mean_shift_in_combined_mcse=max_shift,held_out=held,
        validated_warmup=warmup,validated_keep=2*spec['keep'],formal_1000_completed=0,
        scope='Full numerical learned-FM terminal target. No conclusion about 1000-input accuracy yet.'))
    print('VALIDATION_COMPLETE',passed,flush=True)


if __name__=='__main__':main()
