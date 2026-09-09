"""Independently reconcile complete NS summaries with audited per-call scores."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parents[1]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))

manifests = {}
for folder, name in [('ns_complete_audit', 'ns_audit_manifest.json'),
                     ('ns_complete_report_v2', 'ns_report_manifest.json'),
                     ('ns_guidance_report_v2', 'ns_guidance_plot_manifest.json'),
                     ('ns_strategy_report', 'ns_strategy_manifest.json')]:
    path = STUDY / folder / name
    m = json.loads(path.read_text())
    assert m['status'] == 'complete' and m['calls_verified'] == 1728
    for filename, digest in m['outputs'].items():
        assert sha(path.parent / filename) == digest, filename
    manifests[folder] = sha(path)

source = json.loads((STUDY/'inputs_v2/source.json').read_text())
ids = source['evaluation_ids']
groups = defaultdict(list)
rows = read(STUDY/'ns_complete_audit/ns_loss_per_run.csv')
assert len(rows) == 1728
for r in rows:
    groups[r['task'], r['method'], int(r['steps']), r['exchange'], int(r['sample_id'])].append(r)
assert len(groups) == 576
assert all(len(v) == 3 and {int(r['seed']) for r in v} == {0, 1, 2} for v in groups.values())

def score(r, metric):
    ea, eu = float(r['rel_l2_a']), float(r['rel_l2_u'])
    if metric == 'primary':
        return eu if r['task'] == 'forward' else ea if r['task'] == 'inverse' else max(ea, eu)
    if metric == 'secant_rms':
        return float(r['fm_residual_mse_f64']) ** .5
    if metric == 'spatial_loss':
        return float(r['diffusion_spatial_loss_f64'])
    return float(r[metric]) if r[metric] else None

def check_stats(r, values, check_ci=True):
    values = np.asarray(values, dtype=np.float64)
    assert int(r['n']) == len(values)
    if not len(values):
        assert all(not r[k] for k in ['mean', 'sd', 'ci_low', 'ci_high'])
        return
    np.testing.assert_allclose([float(r['mean']), float(r['sd'])],
                               [values.mean(), values.std(ddof=1)], rtol=2e-12, atol=2e-14)
    if check_ci:
        indices = np.random.default_rng(20260908).integers(len(values), size=(4000, len(values)))
        ci = np.quantile(np.mean(values[indices], axis=1), [.025, .975])
        np.testing.assert_allclose([float(r['ci_low']), float(r['ci_high'])], ci, rtol=2e-12, atol=2e-14)

values = {}
summaries = read(STUDY/'ns_complete_report/ns_loss_summary.csv')
for r in summaries:
    key = r['task'], r['method'], int(r['steps']), r['exchange'], r['metric']
    per_input = {}
    for i in ids:
        scores = [score(x, r['metric']) for x in groups[key[:4] + (i,)]]
        if all(v is not None for v in scores):
            per_input[i] = float(np.mean(scores))
    values[key] = per_input
    check_stats(r, list(per_input.values()))

paired = read(STUDY/'ns_complete_report/ns_loss_paired_effects.csv')
for r in paired:
    prefix = r['task'], r['method'], int(r['steps'])
    a, b = (values[prefix + (e, r['metric'])] for e in ['False', 'True'])
    check_stats(r, [b[i]-a[i] for i in sorted(a.keys() & b.keys())])

strategies = read(STUDY/'ns_strategy_report/ns_strategy_per_input.csv')
strategy_summaries = read(STUDY/'ns_strategy_report/ns_strategy_summary.csv')
assert len(strategies) == 1824
for r in strategy_summaries:
    rr = [x for x in strategies if all(x[k] == r[k] for k in ['setting', 'estimator', 'task'])]
    assert len(rr) == 32 and {int(x['sample_id']) for x in rr} == set(ids)
    # Source row order preserves the predeclared input order used in the bootstrap.
    by_id = {int(x['sample_id']): x for x in rr}
    check_stats(r, [float(by_id[i][r['metric']]) for i in ids if by_id[i][r['metric']]])

identities = read(STUDY/'ns_strategy_report/ns_ensemble_identity.csv')
assert len(identities) == 1536
for r in identities:
    a, b, spread = [float(r[k]) for k in ['mean_individual_squared_error', 'ensemble_squared_error', 'normalized_spread']]
    assert spread >= 0 and b <= a + 1e-12
    np.testing.assert_allclose(a, b+spread, rtol=2e-12, atol=2e-14)

# Reconcile the spectral decomposition with mean individual squared field error.
bands = [r for r in read(STUDY/'ns_complete_report/ns_frequency_bands.csv')
         if r['task'] == 'inverse' and r['field'] == 'a' and r['method'] == 'DiffusionPDE'
         and r['steps'] == '1000' and r['cutoffs'] == '8/32']
changes = {}
for band in ['dc', 'low', 'mid', 'high']:
    means = []
    for exchange in ['False', 'True']:
        rr = [r for r in bands if r['band'] == band and r['exchange'] == exchange]
        assert len(rr) == 96
        means.append(float(np.mean([float(r['global_error_contribution']) for r in rr])))
    changes[band] = means[1] - means[0]
raw_means = [np.mean([float(r['rel_l2_a'])**2 for r in rows
                     if (r['task'], r['method'], r['steps'], r['exchange']) ==
                     ('inverse', 'DiffusionPDE', '1000', e)]) for e in ['False', 'True']]
np.testing.assert_allclose(sum(changes.values()), raw_means[1]-raw_means[0], rtol=2e-12, atol=2e-14)

# Plot readability revision must leave the gradient summaries unchanged.
for old in (STUDY/'ns_complete_report').iterdir():
    if old.suffix in ['.csv', '.tex']:
        assert sha(old) == sha(STUDY/'ns_complete_report_v2'/old.name)
for name in ['ns_guidance_per_run.csv', 'ns_guidance_summary.csv']:
    assert sha(STUDY/'ns_guidance_report'/name) == sha(STUDY/'ns_guidance_report_v2'/name)

result = dict(status='pass', manifests=manifests, calls=1728, input_seed_groups=len(groups),
              loss_summary_rows=len(summaries), paired_loss_rows=len(paired),
              strategy_summary_rows=len(strategy_summaries), variance_identity_checks=len(identities),
              inverse_diffusion_1000_squared_error_change=sum(changes.values()),
              inverse_diffusion_1000_band_error_changes=changes,
              inverse_diffusion_1000_band_reduction_shares={k: v/sum(changes.values()) for k, v in changes.items()},
              script_sha256=sha(Path(__file__)))
out = Path(__file__).with_name('complete_reporting_qa.json')
out.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
