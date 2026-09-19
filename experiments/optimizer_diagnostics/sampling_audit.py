"""Independently recompute field metrics from retained physical predictions."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from experiments.optimizer_diagnostics.hard_sampling import location, report
from experiments.optimizer_diagnostics.study import sha, write


def audit(root,pde):
    out=location(root,pde)
    selection=json.loads((out/'selection.json').read_text())
    assert sha(out/'selected_reference.pt')==selection['selected_reference_sha256']
    reference=torch.load(out/'selected_reference.pt',map_location='cpu',weights_only=False)
    ids=selection['indices'];assert reference['indices']==ids and len(set(ids))==16
    gate=json.loads((out/'paired_execution_gate.json').read_text())
    assert gate['passed'] and gate['difference']['max_relative']<1e-6
    record=[]
    for complete_path in sorted((out/'runs').glob('*/complete.json')):
        folder=complete_path.parent
        complete=json.loads(complete_path.read_text())
        identity=json.loads((folder/'identity.json').read_text())
        assert sha(out/'selection.json')==identity['selection_sha256']
        assert complete['checkpoint_sha256']==identity['checkpoint_sha256']
        if folder.name=='original': checkpoint=root/'inputs'/pde/'source.pth'
        elif folder.name.endswith('_128'):
            checkpoint=root/'runs'/pde/('lr_control.pth' if folder.name=='lr_control_128' else 'selected_resume.pth')
        else:
            arm='lr_control' if folder.name=='lr_control_512' else 'selected_resume'
            checkpoint=root/'continuation/nsnonbounded'/arm/'step_0512.pth'
        assert sha(checkpoint)==identity['checkpoint_sha256']
        assert json.loads((out/f'{folder.name}.exit.json').read_text())['exit_code']==0
        for seed in selection['seeds']:
            seen=[];rows=[]
            for path in sorted(folder.glob(f'seed{seed}_batch*.pt')):
                receipt=json.loads(path.with_suffix('.json').read_text())
                assert sha(path)==receipt['sha256'] and receipt['seed']==seed
                pack=torch.load(path,map_location='cpu',weights_only=False)
                assert pack['indices']==receipt['indices']
                assert torch.isfinite(pack['prediction']).all()
                index=[ids.index(i) for i in pack['indices']]
                recomputed={}
                for field,channel,truth_key in [('u',0 if pde=='burger' else 1,'sol_truth')]+([] if pde=='burger' else [('a',0,'coef_truth')]):
                    actual=pack['prediction'][:,channel:channel+1].double()
                    truth=reference[truth_key][index].double()
                    error=(actual-truth).flatten(1).norm(dim=1)/truth.flatten(1).norm(dim=1).clamp_min(1e-12)
                    np.testing.assert_allclose(error.numpy(),[r[f'rel_l2_{field}'] for r in pack['metrics']],rtol=1e-12,atol=1e-12)
                    recomputed[field]=error.tolist()
                for j,i in enumerate(pack['indices']):
                    assert pack['metrics'][j]['index']==i
                    rows.append(dict(index=i,**{f'rel_l2_{f}':v[j] for f,v in recomputed.items()}))
                seen.extend(pack['indices'])
            assert seen==ids
            for row,saved in zip(rows,complete['results'][str(seed)]['rows']):
                assert row['index']==saved['index']
                for k,v in row.items(): np.testing.assert_allclose(v,saved[k],rtol=1e-12,atol=1e-12)
        record.append(dict(variant=folder.name,checkpoint_sha256=identity['checkpoint_sha256'],count=32,
            metrics_recomputed_cpu_float64=True,all_prediction_sha256_verified=True))
    assert any(x['variant']=='original' for x in record)
    report(root,pde) # Also recheck masks, configurations and all 101 noise hashes.
    write(out/'audit.json',dict(status='verified',pde=pde,selection_sha256=sha(out/'selection.json'),
        cases=16,seeds=selection['seeds'],variants=record,paired_execution_gate=gate['difference']))
    print('SAMPLING_AUDIT_VERIFIED',pde,[r['variant'] for r in record],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--pde',required=True)
    a=p.parse_args();torch.set_num_threads(4);audit(a.root,a.pde)
