"""Freeze field-wise choices using completed diagnostic inputs only."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import socket
import time
import os


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pdes',nargs='+',required=True)
    a=p.parse_args()
    for pde in a.pdes:
        target=a.output/pde
        assignment=target/'external_assignment.json'
        if assignment.exists():
            owner=json.loads(assignment.read_text())
            protocol_hash=hashlib.sha256((a.inputs/pde/'protocol.json').read_bytes()).hexdigest()
            assert owner['protocol_sha256']==protocol_hash
            is_owner = socket.gethostname()==owner['host'] and (
                'visible_devices' not in owner or os.environ.get('CUDA_VISIBLE_DEVICES')==owner['visible_devices'])
            if not is_owner:
                deadline=time.monotonic()+43200
                for stage in owner['stages']:
                    marker=target/f'{stage}_complete.json'
                    while not marker.exists():
                        if time.monotonic()>deadline:
                            raise TimeoutError(f'{pde} {stage} results have not arrived from {owner["host"]}')
                        print(f'Waiting for {pde} {stage} from {owner["host"]}',flush=True)
                        time.sleep(30)
                    assert json.loads(marker.read_text())['protocol_sha256']==protocol_hash
        selection=target/'selection.json'
        if selection.exists():
            print('EXISTS',selection);continue
        path=a.inputs/pde/'protocol.json'
        protocol=json.loads(path.read_text())
        ph=hashlib.sha256(path.read_bytes()).hexdigest()
        tasks=list(protocol['base_configs'])
        updates={task:{} for task in tasks}
        evidence={}
        if pde in ['helmholtz','darcy','burger']:
            reason='Unchanged archived anchor settings; no selection on ensemble inputs.'
        else:
            done=json.loads((target/'diagnose_complete.json').read_text())
            assert done['protocol_sha256']==ph
            rows=[json.loads(x.read_text()) for x in (target/'diagnostic').rglob('receipt.json')]
            assert len(rows)==done['runs']
            assert all(r['protocol_sha256']==ph for r in rows)
            for task in tasks:
                fields=['u'] if task=='forward' else ['a'] if task=='inverse' else ['a','u']
                named={}
                for r in rows:
                    t,name,_,_=r['label'].split('/')
                    if t==task:named.setdefault(name,[]).append(r)
                expected=len(protocol['tuning_ids'])*(len(protocol['tuning_observation_counts']) if task=='both' else 1)
                assert all(len(rs)==expected for rs in named.values())
                means={name:{f:statistics.mean(r['errors'][f][0] for r in rs)
                             if all(r['errors'][f][0] is not None for r in rs) else None
                             for f in fields} for name,rs in named.items()}
                if pde in ['poisson','nsnonbounded']:
                    # Select among finite candidates. A candidate is acceptable only
                    # if every diagnostic prediction has <100% error in each scored
                    # field. This criterion is applied before any ID-0 rerun.
                    acceptable=[name for name,rs in named.items() if name!='archived' and
                        all(r['errors'][f][0] is not None and r['errors'][f][0]<1.0 for r in rs for f in fields)]
                    if not acceptable:
                        acceptable=[name for name,rs in named.items() if name!='archived' and
                                    all(r['finite'] for r in rs)]
                    assert acceptable,(pde,task,'No finite clipping candidate')
                    # Lexicographic field-wise criterion: inverse/coefficient field
                    # first, then solution field. No max/average field score.
                    winner=min(acceptable,key=lambda n:tuple(means[n][f] for f in fields))
                    updates[task]={'clip_threshold':named[winner][0]['config']['clip_threshold']}
                else:
                    assert set(named)=={'archived','recommended'}
                    winner='recommended' if all(means['recommended'][f] is not None and
                        (means['archived'][f] is None or means['recommended'][f]<=means['archived'][f]*(1+1e-6))
                        for f in fields) else 'archived'
                    if winner=='recommended':
                        c=named[winner][0]['config']
                        updates[task]={k:c[k] for k in ['zeta_obs_a','zeta_obs_u','zeta_pde','clip_mode','clip_threshold']}
                evidence[task]=dict(chosen=winner,mean_errors_by_field=means,
                                    diagnostic_runs_per_candidate=expected,fields=fields)
            reason='Disjoint four-input diagnostics; field-wise selection; original input 0 and ensemble inputs unused.'
        target.mkdir(parents=True,exist_ok=True)
        selection.write_text(json.dumps(dict(protocol_sha256=ph,task_updates=updates,
            evidence=evidence,selection_reason=reason),indent=2,allow_nan=False)+'\n')
        print('FROZEN',pde,updates)


if __name__=='__main__':main()
