"""Paired, input-level frequency and sample-averaging evidence."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path
import numpy as np
from spectral_diagnostics import paired_bootstrap
from run_ns_loss_study import sha, write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True)
    args=p.parse_args();root=args.report
    with gzip.open(root/'frequency_records.json.gz','rt') as f: records=json.load(f)
    groups=defaultdict(list)
    for r in records:groups[r['pde'],r['method'],r['steps'],r['ensemble'],r['sample_id']].append(r)
    ids=sorted({r['sample_id'] for r in records});assert len(ids)==32
    def values(pde,method='FM4PDE',steps=100,ensemble=False):
        return np.array([np.mean([r['rel_l2'] for r in groups[pde,method,steps,ensemble,i]]) for i in ids])
    effects=[]
    def effect(meta, first, second):
        delta=first-second;ci=paired_bootstrap(delta)
        effects.append(dict(**meta,examples=len(delta),first_mean=float(first.mean()),second_mean=float(second.mean()),
            difference=float(delta.mean()),ci_low=float(ci[0]),ci_high=float(ci[1])))
    for pde in ['poisson','darcy']:
        for n in [25,50,100,200]:
            effect(dict(pde=pde,comparison='three-sample mean minus single sample',steps=n,metric='relative_l2',band='full'),values(pde,steps=n,ensemble=True),values(pde,steps=n))
        effect(dict(pde=pde,comparison='200 minus 100 steps, single sample',steps=200,metric='relative_l2',band='full'),values(pde,steps=200),values(pde))
        for method in ['RecFNO','Senseiver','VoronoiCNN']:
            effect(dict(pde=pde,comparison='FM4PDE minus '+method,steps=100,metric='relative_l2',band='full'),values(pde),values(pde,method,1))

    # Read every predeclared cutoff, rather than choosing the most favorable
    # band boundary. Average stochastic outcomes within each physical input.
    band_groups=defaultdict(list)
    with (root/'frequency_band_per_sample.csv').open() as f:
        for row in csv.DictReader(f):
            if row['ensemble']=='True' or (row['method']=='FM4PDE' and row['steps']!='100'):continue
            key=(row['pde'],row['method'],float(row['low_cutoff']),float(row['high_cutoff']),row['band'],int(row['sample_id']))
            tr,pr,er=(float(row[k]) for k in ['reference_energy','prediction_energy','error_energy'])
            ratio=float(row['predicted_reference_energy_ratio']) if row['predicted_reference_energy_ratio'] else None
            # This is coefficient alignment, not a temporal coherence estimate.
            alignment=(pr+tr-er)/(2*np.sqrt(pr*tr)) if pr>0 and ratio is not None else None
            band_groups[key].append(dict(relative_l2=float(row['relative_error']) if row['relative_error'] else None,
                reference_fraction=float(row['reference_fraction']),global_squared_error=float(row['global_error_contribution']),
                energy_ratio=ratio,energy_mismatch=abs(ratio-1) if ratio is not None else None,alignment=alignment))
    summaries=[]
    for pde in ['poisson','darcy']:
        for low,high in [(4.,16.),(8.,32.),(16.,48.)]:
            for band in ['dc','low','mid','high']:
                arrays={}
                for method in ['FM4PDE','RecFNO','Senseiver','VoronoiCNN']:
                    for metric in ['relative_l2','reference_fraction','global_squared_error','energy_ratio','energy_mismatch','alignment']:
                        rows=[band_groups[pde,method,low,high,band,i] for i in ids]
                        if any(any(r[metric] is None for r in rr) for rr in rows):continue
                        x=np.array([np.mean([r[metric] for r in rr]) for rr in rows]);arrays[method,metric]=x
                        ci=paired_bootstrap(x)
                        summaries.append(dict(pde=pde,method=method,low_cutoff=low,high_cutoff=high,band=band,metric=metric,
                            examples=len(x),mean=float(x.mean()),sd=float(x.std(ddof=1)),ci_low=float(ci[0]),ci_high=float(ci[1])))
                for metric in ['relative_l2','global_squared_error','energy_mismatch','alignment']:
                    for method in ['RecFNO','Senseiver','VoronoiCNN']:
                        if ('FM4PDE',metric) in arrays and (method,metric) in arrays:
                            effect(dict(pde=pde,comparison='FM4PDE minus '+method,steps=100,metric=metric,
                                band=f'{band}: {low:g}/{high:g}'),arrays['FM4PDE',metric],arrays[method,metric])
    for name,rows in [('frequency_paired_effects.csv',effects),('frequency_band_summary.csv',summaries)]:
        with (root/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write(root/'paired_evidence_manifest.json',dict(status='complete',examples_per_pde=32,
        uncertainty='95% pointwise percentile bootstrap confidence interval, 4,000 paired resamples of physical inputs; no multiple-comparison correction.',
        seed_aggregation='Average seed-level errors, powers, energy mismatch and alignment within each input before resampling. Ensemble means average fields before scoring.',
        alignment='Real normalized coefficient inner product, recovered from prediction, reference and error energies. Does not certify posterior calibration.',
        source_hashes={name:sha(root/name) for name in ['frequency_records.json.gz','frequency_band_per_sample.csv']},
        outputs={name:sha(root/name) for name in ['frequency_paired_effects.csv','frequency_band_summary.csv']},script_sha256=sha(Path(__file__))))
    for r in effects:
        if r['band']=='full' and r['steps']==100:print(json.dumps(r))
    for r in summaries:
        if r['low_cutoff']==8 and r['high_cutoff']==32 and r['band']=='high':print(json.dumps(r))


if __name__=='__main__':main()
