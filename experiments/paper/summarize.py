"""Collect physical-unit relative errors, retaining every realization."""
import csv
import hashlib
import json
from pathlib import Path


def summarize(root, manifest=None):
    import numpy as np
    import torch
    from scipy.stats import t
    root = Path(root)
    records = json.loads((root/'index.json').read_text())
    if not records:
        raise ValueError('No completed predictions to summarize')
    # Older receipts lack cohort metadata. An explicit manifest can recover it
    # without sampling again or modifying the saved predictions and receipts.
    cohorts = None
    if manifest is not None:
        import yaml
        spec = yaml.safe_load(Path(manifest).read_text())
        cohorts = {job['id']: job.get('cohort') for job in spec['jobs']}
    rows = []
    settings = {}
    seen_jobs, seen_samples = set(), set()
    for record in records:
        from experiments.paper.run import digest
        if record['id'] in seen_jobs:
            raise ValueError(f"Duplicate job: {record['id']}")
        seen_jobs.add(record['id'])
        if digest(record['result']) != record['result_sha256']:
            raise ValueError(f"Prediction checksum mismatch: {record['result']}")
        data = torch.load(record['result'], map_location='cpu', weights_only=False)
        config = record['identity']['config']
        cohort = record['identity'].get('cohort')
        if cohorts is not None:
            if record['id'] not in cohorts:
                raise ValueError(f"Job absent from summary manifest: {record['id']}")
            declared = cohorts[record['id']]
            if cohort is not None and cohort != declared:
                raise ValueError(f"Cohort differs from summary manifest: {record['id']}")
            cohort = declared
        control = {k:v for k,v in config.items() if k not in {'offset','batch_size','mask_seed','sample_seed','noise_seed','output_dir','runtime_metadata','initial_noise_source_indices'}}
        control['observation_protocol'] = record['identity'].get('observation_protocol')
        if cohort is not None:
            control['cohort'] = cohort
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
        n = len(errors['u'])
        if n != config['batch_size']:
            raise ValueError(f"Incomplete batch: {record['id']}: {n} != {config['batch_size']}")
        from experiments.paper.residual_metrics import full_pde_loss
        # Re-evaluate the full objective, including the saved boundary treatment.
        # Never substitute the historical interior-only residual_mse diagnostic.
        pde_losses = full_pde_loss(data)
        consistency = record.get('consistency', {})
        solver = consistency.get('solver_relative_l2', [None] * n)
        if len(solver) != n:
            raise ValueError(f"Incomplete solver evaluation: {record['id']}")
        for i in range(len(next(iter(errors.values())))):
            sample_key = (group, config['offset'] + i)
            if sample_key in seen_samples:
                raise ValueError(f'Duplicate sample in reported setting: {sample_key}')
            seen_samples.add(sample_key)
            error_a = errors.get('a', [None]*n)[i]
            rows.append(dict(job=record['id'], group=group, cohort=cohort, pde=config['pde'], task=config['task'],
                             distribution=config['test_type'], index=config['offset']+i,
                             relative_l2_a=error_a, relative_l2_u=errors['u'][i],
                             relative_l2_joint=(error_a + errors['u'][i])/2 if error_a is not None else None,
                             pde_loss=pde_losses[i], solver_relative_l2=solver[i]))
    with (root/'errors.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    groups = {}
    for row in rows:
        # Pool batches only within the same cohort and scientific setting.
        key = row['group']
        groups.setdefault(key, []).append(row)
    summary = []
    for key, values in groups.items():
        for field, column in [('a','relative_l2_a'), ('u','relative_l2_u'),
                              ('joint','relative_l2_joint'), ('pde','pde_loss'),
                              ('solver','solver_relative_l2')]:
            data = [v[column] for v in values if v[column] is not None]
            if not data: continue
            if len(data) != len(values) or not np.isfinite(data).all():
                raise ValueError(f'Missing or non-finite {column} in group {key}')
            n=len(data); sd=float(np.std(data, ddof=1)) if n>1 else None
            summary.append(dict(group=key, settings=settings[key], field=field, metric=column, count=n, mean=float(np.mean(data)),
                                std=sd, ci95_halfwidth=float(t.ppf(.975,n-1)*sd/np.sqrt(n)) if n>1 else None))
    (root/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path, help='Experiment output containing index.json')
    parser.add_argument('--manifest', type=Path,
                        help='Recover cohort labels for receipts saved before cohort tracking')
    args = parser.parse_args()
    summarize(args.root, args.manifest)
