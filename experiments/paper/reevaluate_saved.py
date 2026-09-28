"""Recompute current Darcy/NS residuals from immutable saved physical fields."""
import argparse
import csv
import json
from pathlib import Path

from experiments.paper.run import ROOT, digest, write_json
from experiments.paper.residual_metrics import evaluate_fields


def run(args):
    import torch
    torch.set_num_threads(args.threads)
    manifest=json.loads(args.manifest.read_text())
    rows=[]
    for entry in manifest['records']:
        pde=entry['pde']
        if pde not in {'darcy','nsnonbounded'}:
            raise ValueError(f'Unsupported saved-prediction comparison: {pde}')
        truth_path=Path(entry['truth_file'])
        if digest(truth_path)!=entry['truth_sha256']:
            raise ValueError(f'Truth checksum mismatch: {truth_path}')
        pack=torch.load(truth_path,map_location='cpu',weights_only=False)
        truth=pack['raw']['full_tensor'] if 'raw' in pack else pack['truth']
        seen=[]
        for source in entry['predictions']:
            path=Path(source['file'])
            if digest(path)!=source['sha256']:
                raise ValueError(f'Prediction checksum mismatch: {path}')
            saved=torch.load(path,map_location='cpu',weights_only=False)
            if 'prediction' in saved:
                indices=list(saved['indices'])
                pred=saved['prediction'].double()
            else:
                pred=torch.cat([saved['coef_final'],saved['sol_final']],1).double()
                indices=list(range(saved['config']['offset'],saved['config']['offset']+len(pred)))
            target=truth[indices].double()
            if pred.shape!=target.shape or pred.shape[1]!=2:
                raise ValueError('Saved pairs must have shape [batch, 2, height, width]')
            metrics=evaluate_fields(pde,pred[:,:1],pred[:,1:],target[:,:1],target[:,1:],entry.get('pde_params'))
            errors=(pred-target).flatten(2).norm(dim=2)/target.flatten(2).norm(dim=2).clamp_min(1e-12)
            for i,index in enumerate(indices):
                rows.append(dict(pde=pde,method=entry['method'],distribution=entry['distribution'],
                    index=index,rel_l2_a=float(errors[i,0]),rel_l2_u=float(errors[i,1]),
                    residual_mse=metrics['residual_mse'][i],truth_residual_mse=metrics['truth_residual_mse'][i],
                    residual_difference_mse=metrics['residual_difference_mse'][i],
                    secant_rhs_evaluation=metrics['secant_rhs_evaluation']))
            seen.extend(indices)
        if sorted(seen)!=list(range(entry['count'])):
            raise ValueError(f'Missing or duplicate cases: {entry["method"]} {pde} {entry["distribution"]}')
        print(pde,entry['method'],entry['distribution'],len(seen),flush=True)
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'per_sample.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(args.output/'receipt.json',dict(
        cases=len(rows),sampling_performed=False,secant_formula='(u-a)/T - (G_h(a)+G_h(u))/2',
        manifest=str(args.manifest.resolve()),manifest_sha256=digest(args.manifest),
        residual_source_sha256=digest(ROOT/'sampling/pde_residuals.py'),
        evaluator_sha256=digest(Path(__file__).with_name('residual_metrics.py')),
        metrics_sha256=digest(args.output/'per_sample.csv')))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--threads',type=int,default=2)
    run(p.parse_args())
