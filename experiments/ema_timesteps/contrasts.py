"""Paired EMA, training-recipe and continuation contrasts on fixed sampling cases."""
import argparse
import json
from pathlib import Path
import re

import numpy as np

from experiments.optimizer_diagnostics.study import write


def run(root, pde):
    folder = root/'evaluation'/pde
    selection = json.loads((root/'evaluation_inputs'/pde/'selection.json').read_text())
    variants = {}
    for path in (folder/'variants').glob('*/complete.json'):
        match = re.fullmatch(r'(.+)_e(\d+)_(raw|ema)', path.parent.name)
        if match:
            arm, epoch, weight = match.groups()
            variants[(arm, int(epoch), weight)] = json.loads(path.read_text())
    pairs = []
    for (arm, epoch, weight), candidate in sorted(variants.items()):
        if weight == 'ema' and (arm, epoch, 'raw') in variants:
            pairs.append(('ema_vs_raw', candidate, variants[(arm, epoch, 'raw')]))
        if arm != 'uniform' and ('uniform', epoch, weight) in variants:
            pairs.append(('recipe_vs_uniform', candidate, variants[('uniform', epoch, weight)]))
        if epoch > 2 and (arm, 2, weight) in variants:
            pairs.append(('longer_training_vs_epoch2', candidate, variants[(arm, 2, weight)]))
    results = []
    for kind, candidate, reference in pairs:
        result = dict(kind=kind, candidate=candidate['variant'], reference=reference['variant'], cohorts={})
        for cohort in ['hard', 'ordinary']:
            ids = selection[cohort]
            assert len(ids) == len(set(ids)) == 16
            result['cohorts'][cohort] = {}
            for field in ['u'] if pde == 'burger' else ['u', 'a']:
                def matrix(record):
                    values = []
                    for seed in selection['seeds']:
                        rows = record['results'][str(seed)]
                        assert [row['index'] for row in rows] == selection['indices']
                        by_id = {row['index']: row for row in rows}
                        values.append([by_id[i][f'rel_l2_{field}'] for i in ids])
                    return np.asarray(values, dtype=np.float64)
                ref, cand = matrix(reference), matrix(candidate)
                assert np.isfinite(ref).all() and np.isfinite(cand).all()
                difference = (cand-ref).mean(axis=0)
                draws = np.random.default_rng(20260924).integers(0, len(ids), (10000, len(ids)))
                result['cohorts'][cohort][field] = dict(
                    reference_mean=float(ref.mean()), candidate_mean=float(cand.mean()),
                    improvement_pct=float(100*(1-cand.mean()/ref.mean())),
                    paired_difference_ci95=np.quantile(difference[draws].mean(axis=1), [.025, .975]).tolist(),
                    improved_cases=int((difference < 0).sum()),
                    seed_improvement_pct=(100*(1-cand.mean(axis=1)/ref.mean(axis=1))).tolist())
        results.append(result)
    output = dict(pde=pde, contrasts=results, seeds=selection['seeds'],
                  sign='Positive improvement means lower physical relative L2; CI is candidate minus reference',
                  method='Average seeds within input, paired bootstrap over 16 inputs, 10000 draws',
                  scope='Exploratory ID cohorts used in selection; marginal intervals without multiplicity correction',
                  available_variants=sorted(v['variant'] for v in variants.values()))
    write(folder/'contrasts.json', output)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--pde', required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.pde)))
