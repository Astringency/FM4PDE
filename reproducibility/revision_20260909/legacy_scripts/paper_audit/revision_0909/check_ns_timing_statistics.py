"""Independently recompute completed NS timing summaries from the 80 receipts."""
from collections import defaultdict
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics

PAPER = Path(__file__).resolve().parents[2]
STUDY = Path('/home/tat512/C01Python/audit/ns_main_revision_0909')
RAW = STUDY / 'timing_results/nsnonbounded'
OUT = PAPER / 'audit/revision_0909/ns_timing_review'
OUT.mkdir(exist_ok=True)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

done = json.loads((RAW / 'complete.json').read_text())
assert done['status'] == 'complete' and done['calls'] == 80
summary_path = STUDY / 'tables/ns_timing_summary.csv'
summary = {(r['method'], int(r['steps'])): r
           for r in csv.DictReader(summary_path.open())}
groups = defaultdict(list)
receipt_hashes = {}
for path in sorted(RAW.glob('*.json')):
    if not path.name.startswith(('FM4PDE_', 'DiffusionPDE_')):
        continue
    r = json.loads(path.read_text())
    key = r['method'], r['steps']
    assert r['pde'] == 'nsnonbounded'
    assert r['protocol_sha256'] == done['protocol_sha256']
    assert r['uncontended'] and r['finite'] and r['telemetry_samples'] > 0
    assert r['nfe'] == (r['steps'] if key[0] == 'FM4PDE' else 2*r['steps']-1)
    assert math.isfinite(r['seconds']) and r['seconds'] > 0
    assert math.isclose(r['end_monotonic']-r['start_monotonic'], r['seconds'], abs_tol=1e-8)
    groups[key].append(r)
    receipt_hashes[path.name] = sha(path)
assert len(receipt_hashes) == 80
assert set(groups) == set(summary) == {(m,n) for m in ['FM4PDE','DiffusionPDE'] for n in [100,1000]}
ids = None
new = []
for key, rows in sorted(groups.items()):
    current_ids = sorted(r['sample_id'] for r in rows)
    assert len(rows) == len(set(current_ids)) == 20
    ids = current_ids if ids is None else ids
    assert current_ids == ids
    assert len({r['uuid'] for r in rows}) == 1
    times = [r['seconds'] for r in rows]
    values = dict(mean_seconds=statistics.mean(times), sd_seconds=statistics.stdev(times),
                  min_seconds=min(times), max_seconds=max(times))
    for name,value in values.items():
        assert math.isclose(value, float(summary[key][name]), rel_tol=1e-12, abs_tol=1e-12)
    new.append(dict(pde='nsnonbounded', method=key[0], steps=key[1], n=20, **values))
assert len({r['uuid'] for rows in groups.values() for r in rows}) == 1

old_path = PAPER / 'source_data/diffusion_fm_timing_summary.csv'
old = list(csv.DictReader(old_path.open()))
combined = [r for r in old if r['pde'] != 'nsnonbounded'] + new
assert len(combined) == 20
index = {(r['pde'],r['method'],int(r['steps'])):float(r['mean_seconds']) for r in combined}
ratios = []
for pde in sorted({r['pde'] for r in combined}):
    for n in [100,1000]:
        ratios.append(dict(pde=pde,steps=n,
                           diffusion_to_fm=index[pde,'DiffusionPDE',n]/index[pde,'FM4PDE',n]))
ranges = {str(n):dict(min=min(r['diffusion_to_fm'] for r in ratios if r['steps']==n),
                     max=max(r['diffusion_to_fm'] for r in ratios if r['steps']==n))
          for n in [100,1000]}
result = dict(status='pass',scope='Receipt-level timing statistics and metadata. Full prediction and telemetry audit is performed by the study exporter.',
              original_other_pde_summary_sha256=sha(old_path), ns_summary_sha256=sha(summary_path),
              completion_sha256=sha(RAW/'complete.json'),protocol_sha256=done['protocol_sha256'],
              complete_calls=80, common_input_ids=ids, receipt_hashes=receipt_hashes,
              ns_recomputed=new, ratios=ratios, ratio_ranges=ranges,
              fm_lower_mean_all_ten_comparisons=all(r['diffusion_to_fm']>1 for r in ratios),
              canonical_manuscript_updated=False)
(OUT/'independent_statistics.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':'pass','calls':80,'ratio_ranges':ranges,'ns':new},indent=2))
