"""Recompute complete-cohort paired errors; refuse partial-cohort conclusions."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
import torch
from scripts.train.resume_study import file_sha,write
from scripts.train.evaluate_resume_study import field_scores


def interval(delta, rng):
    means=[]
    for _ in range(40):
        indices=rng.integers(0,len(delta),size=(500,len(delta)))
        means.extend(delta[indices].mean(1))
    return np.quantile(means,[.0125,.9875]).tolist()


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--name',required=True)
    args=p.parse_args()
    torch.set_num_threads(2)
    ip=json.loads((args.output/'inputs/protocol.json').read_text())
    assert file_sha(args.output/'inputs/cases.pt')==ip['cases_sha256']
    cases=torch.load(args.output/'inputs/cases.pt',map_location='cpu',weights_only=False)
    run=args.output/args.name
    protocol=json.loads((run/'protocol.json').read_text())
    assert not protocol['pilot'] and not protocol['probe_only']
    assert sorted(protocol['sample_ids'])==sorted(ip['sample_ids']) and len(ip['sample_ids'])==1000
    ph=file_sha(run/'protocol.json')
    rows={};diag={};sources=[]
    for path in sorted(run.glob('ids_*/complete.json')):
        receipt=json.loads(path.read_text())
        result=path.parent/'result.pt'
        assert receipt['protocol_sha256']==ph and file_sha(result)==receipt['result_sha256']
        raw=torch.load(result,map_location='cpu',weights_only=False)
        ids=raw['sample_ids']
        assert ids==receipt['sample_ids']
        for i,sid in enumerate(ids):
            assert sid not in rows and sid in ip['sample_ids']
            assert torch.equal(raw['truth'][i:i+1],cases[sid]['truth'])
            for f in ['coef','sol']:
                assert torch.equal(raw['masks'][f][i:i+1],cases[sid]['masks'][f])
            scores={}
            for name,pred in [('baseline',cases[sid]['baseline']),('draw',raw['draw'][i:i+1]),
                              ('finite_chain_mean',raw['finite_chain_mean'][i:i+1])]:
                channels=[('u',0)] if ip['pde']=='burger' else [('a',0),('u',1)]
                scores[name]={f:field_scores(pred[:,ch:ch+1],cases[sid]['truth'][:,ch:ch+1],ip['pde']) for f,ch in channels}
                if name!='baseline':
                    err=[scores[name][f]['relative_l2'][0] for f,ch in channels]
                    assert np.allclose(err,receipt['errors'][name][i],rtol=1e-12,atol=1e-12)
            rows[sid]=scores;diag[sid]=receipt['diagnostics'][i]
        sources.append(dict(path=str(result),sha256=receipt['result_sha256']))
    assert sorted(rows)==sorted(ip['sample_ids']), f'Incomplete: {len(rows)}/1000; no population conclusion allowed'
    ids=ip['sample_ids'];summary={};flat=[]
    for estimator in ['draw','finite_chain_mean']:
        summary[estimator]={}
        for field in rows[ids[0]]['baseline']:
            summary[estimator][field]={}
            for metric in ['relative_l2','low_relative_l2','low_power_ratio','low_alignment']:
                old=np.array([rows[i]['baseline'][field][metric][0] for i in ids])
                new=np.array([rows[i][estimator][field][metric][0] for i in ids])
                delta=new-old
                ci=interval(delta,np.random.default_rng(20260912))
                summary[estimator][field][metric]=dict(n=1000,baseline_mean=float(old.mean()),tilt_mean=float(new.mean()),
                    mean_change=float(delta.mean()),paired_97_5ci=ci,
                    relative_change=float(new.mean()/old.mean()-1),fraction_lower=float((new<old).mean()))
                for i,sid in enumerate(ids):
                    flat.append(dict(sample_id=sid,estimator=estimator,field=field,metric=metric,baseline=old[i],tilt=new[i],
                        difference=delta[i],mixing_diagnostics_passed=diag[sid]['passed']))
    with (run/'metrics_per_sample.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    passed=sum(d['passed'] for d in diag.values())
    write(run/'summary.json',dict(n=1000,summary=summary,mixing_passed_inputs=passed,mixing_total_inputs=1000,
        exact_posterior_performance_claim_allowed=False,
        interpretation=('All monitored inputs pass the predeclared necessary diagnostics; finite MCMC bias still cannot be excluded.' if passed==1000
            else 'Finite-chain reconstruction scores only. Mixing diagnostics fail; these results do not establish the performance of exact tilted sampling.'),
        baseline_source='Original main1000 saved predictions, independently recomputed and matched by input and actual observation masks.',
        estimator_caveat='Primary: fixed chain-0 terminal draw per input. Chain mean is a separate point estimator; averaging changes the estimator.',
        confidence_intervals='20,000 paired bootstrap resamples of the 1000 input IDs; 97.5% intervals for two primary field errors. Spectral intervals descriptive.',
        sources=sources,protocol_sha256=ph))
    print(json.dumps(dict(n=1000,mixing_passed_inputs=passed,summary=summary),indent=2))


if __name__=='__main__':main()
