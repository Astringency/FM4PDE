"""Collect physical-unit relative errors, retaining every realization."""
import csv
import hashlib
import json
from pathlib import Path


def summarize(root):
    import numpy as np
    import torch
    from scipy.stats import t
    root = Path(root)
    records = json.loads((root/'index.json').read_text())
    rows = []
    settings = {}
    for record in records:
        data = torch.load(record['result'], map_location='cpu', weights_only=False)
        config = record['identity']['config']
        control = {k:v for k,v in config.items() if k not in {'offset','batch_size','mask_seed','sample_seed','noise_seed','output_dir','runtime_metadata','initial_noise_source_indices'}}
        group = hashlib.sha256(json.dumps(control,sort_keys=True).encode()).hexdigest()[:16]
        settings[group] = control
        errors = {}
        for name, key in [('a','coef'),('u','sol')]:
            if name == 'a' and config['pde'] == 'burger':
                continue
            pred, truth = data[key+'_final'].double(), data[key+'_ground_truth'].double()
            if not torch.isfinite(pred).all():
                raise ValueError(f"Non-finite prediction: {record['result']}")
            errors[name] = ((pred-truth).flatten(1).norm(dim=1)/truth.flatten(1).norm(dim=1).clamp_min(1e-12)).tolist()
        for i in range(len(next(iter(errors.values())))):
            rows.append(dict(job=record['id'], group=group, pde=config['pde'], task=config['task'],
                             distribution=config['test_type'], index=config['offset']+i,
                             relative_l2_a=errors.get('a', [None]*config['batch_size'])[i], relative_l2_u=errors['u'][i]))
    with (root/'errors.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    groups = {}
    for row in rows:
        # Identical controls across test offsets form one reported comparison.
        key = row['group']
        groups.setdefault(key, []).append(row)
    summary = []
    for key, values in groups.items():
        for field in ['a','u']:
            data = [v['relative_l2_'+field] for v in values if v['relative_l2_'+field] is not None]
            if not data: continue
            n=len(data); sd=float(np.std(data, ddof=1)) if n>1 else None
            summary.append(dict(group=key, settings=settings[key], field=field, count=n, mean=float(np.mean(data)),
                                std=sd, ci95_halfwidth=float(t.ppf(.975,n-1)*sd/np.sqrt(n)) if n>1 else None))
    (root/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
