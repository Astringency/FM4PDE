"""Replace combined NS reporting with separate physical-field outcomes."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from report_ns_checkpoint_comparison import font_setup,save

FIELDS=[('forward','rel_l2_u','Forward $e_u$'),('inverse','rel_l2_a','Inverse $e_a$'),
        ('both','rel_l2_a','Joint $e_a$'),('both','rel_l2_u','Joint $e_u$')]


def read(path):return list(csv.DictReader(path.open()))


def fmt(row,digits=2):
    return f'${100*float(row["mean"]):.{digits}f}\\pm{100*float(row["sd"]):.{digits}f}$'


def table(caption,label,header,rows,spec):
    return '\n'.join([r'\begin{table}[!htbp]\centering\footnotesize\setlength{\tabcolsep}{4pt}',
        '\\caption{'+caption+'}', '\\label{'+label+'}',
        r'\begin{tabular}{@{}'+spec+r'@{}}\toprule',header+r' \\\midrule',
        *[r' & '.join(row)+r' \\' for row in rows],r'\bottomrule\end{tabular}\end{table}',''])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--calibration-calls',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    used=[]
    def data(name):
        path=a.source/name;used.append(path);return read(path)
    loss=data('ns_loss_summary.csv');raw=data('ns_loss_per_run.csv')
    groups=defaultdict(list)
    for r in raw:
        groups[r['task'],r['method'],r['steps'],r['exchange'],r['sample_id']].append(r)
    assert len(raw)==1728 and len(groups)==576
    assert all({r['seed'] for r in rows}=={'0','1','2'} and len(rows)==3 for rows in groups.values())
    for r in loss:
        if r['metric'] not in ['rel_l2_a','rel_l2_u']:continue
        values=np.array([np.mean([float(x[r['metric']]) for x in rows]) for key,rows in groups.items()
                         if key[:4]==(r['task'],r['method'],r['steps'],r['exchange'])])
        assert len(values)==32
        assert np.isclose(values.mean(),float(r['mean']),rtol=1e-12,atol=1e-12)
        assert np.isclose(values.std(ddof=1),float(r['sd']),rtol=1e-12,atol=1e-12)
    effects=data('ns_loss_paired_effects.csv')
    rows=[]
    for method,n in [('FM4PDE','100'),('DiffusionPDE','100'),('DiffusionPDE','1000')]:
        for task,metric,label in FIELDS:
            r=next(x for x in effects if (x['method'],x['steps'],x['task'],x['metric'])==(method,n,task,metric))
            digits=5 if n=='100' else 3
            rows.append([method,n,label,fmt(r,digits),
                f'$[{100*float(r["ci_low"]):.{digits}f},{100*float(r["ci_high"]):.{digits}f}]$'])
    (a.output/'ns_loss_effects_fields.tex').write_text(table(
        'Paired changes after NS loss exchange (exchanged minus original, percentage points). '
        'Both joint fields are retained. Means and SDs use 32 input-level seed averages; '
        'intervals are pointwise 95\\% bootstrap intervals over physical inputs.',
        'tab:ns-loss-effects',r'Recipient & Steps & Task / field & Mean $\pm$ SD (pp) & 95\% interval',rows,'lrllr'))
    strategies=data('ns_strategy_summary.csv')
    settings=[('FM4PDE_100_original',r'FM100 / $L_F$'),('FM4PDE_100_exchanged',r'FM100 / $L_D$'),
              ('DiffusionPDE_100_original',r'DM100 / $L_D$'),('DiffusionPDE_100_exchanged',r'DM100 / $L_F$'),
              ('DiffusionPDE_1000_original',r'DM1000 / $L_D$'),('DiffusionPDE_1000_exchanged',r'DM1000 / $L_F$'),
              ('FM_cal_reference','FM100 / calibration reference'),('FM_cal_selected','FM100 / calibrated'),
              ('RecFNO','RecFNO'),('Senseiver','Senseiver'),('VoronoiCNN','VoronoiCNN')]
    rows=[]
    for setting,label in settings:
        estimators=['deterministic'] if setting in ['RecFNO','Senseiver','VoronoiCNN'] else ['single','mean3']
        for estimator in estimators:
            rr=[next(x for x in strategies if (x['setting'],x['estimator'],x['task'],x['metric'])==
                     (setting,estimator,task,metric)) for task,metric,_ in FIELDS]
            assert all(x['n']=='32' for x in rr)
            rows.append([label,'Mean of 3' if estimator=='mean3' else 'Single' if estimator=='single' else '---',*[fmt(x) for x in rr]])
    (a.output/'ns_strategy_fields.tex').write_text(table(
        'NS sampling strategies on 32 common inputs. Errors are percentages, mean $\\pm$ SD across inputs. '
        'Single averages three individual-seed errors within each input; Mean of 3 scores the average '
        'of three physical predictions and triples sampling calls. FM and DM denote FM4PDE and DiffusionPDE. '
        'The original calibration control is retained separately because it was evaluated in the calibration experiment.',
        'tab:ns-sampling-strategies','Method / configuration & Estimate & '+ ' & '.join(x[2] for x in FIELDS),rows,'llrrrr'))
    guidance=data('guidance_evaluation_summary.csv')
    rows=[]
    for variant,estimator,label in [('reference','single','FM original / single'),('selected','single','FM selected / single'),
        ('reference','mean3','FM original / mean of 3'),('selected','mean3','FM selected / mean of 3'),
        ('RecFNO','deterministic','RecFNO'),('Senseiver','deterministic','Senseiver'),('VoronoiCNN','deterministic','VoronoiCNN')]:
        rr=[next(x for x in guidance if (x['variant'],x['estimator'],x['task'],x['metric'])==
                 (variant,estimator,task,metric)) for task,metric,_ in FIELDS]
        assert all(x['n']=='32' for x in rr)
        rows.append([label,*[fmt(x) for x in rr]])
    tex=table('NS guidance calibration at 100 steps: field errors in percent, mean $\\pm$ SD over 32 inputs. '
        'Single averages three seed errors within an input; the mean-of-three estimate averages the physical predictions before scoring. '
        'Selection used four separate development inputs.','tab:ns-calibration',
        'Method / estimate & '+' & '.join(x[2] for x in FIELDS),rows,'lrrrr')
    tex=tex.replace(r'\label{tab:ns-calibration}',r'\label{tab:ns-calibration}\label{tab:ns-calibration-fields}')
    (a.output/'ns_calibration_fields.tex').write_text(tex)
    development=read(a.calibration_calls);used.append(a.calibration_calls)
    development=[x for x in development if x['stage']=='calibration']
    assert len(development)==540
    selection_path=a.source/'ns_guidance_calibration_frozen_selection.json';used.append(selection_path)
    selected=json.loads(selection_path.read_text())['selected']
    rows=[]
    for task,metric,label in FIELDS:
        vals=[]
        for name in ['obs1_pde1',selected[task]['name']]:
            rr=[r for r in development if r['task']==task and r['candidate']==name]
            ids=sorted({r['sample_id'] for r in rr});assert len(ids)==4 and len(rr)==12
            v=np.array([np.mean([float(r[metric]) for r in rr if r['sample_id']==i]) for i in ids])
            vals.append(dict(mean=v.mean(),sd=v.std(ddof=1)))
        rows.append([label,f'{selected[task]["observation_multiplier"]:g}',
            f'{selected[task]["pde_multiplier"]:g}',*[fmt(v) for v in vals]])
    (a.output/'ns_calibration_selection_fields.tex').write_text(table(
        'Retained NS guidance multipliers and development errors. Values are mean $\\pm$ SD (\\%) '
        'across four inputs after averaging three seed errors per input. Joint coefficient and solution errors are shown separately.',
        'tab:ns-calibration-selection',r'Task / field & Obs. multiplier & PDE multiplier & Original & Selected',rows,'lrrrr'))
    pairs=data('guidance_paired_effects.csv')
    font_setup()
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(9.2,5.5),layout='constrained')
    comparisons=[('reference','single','Original FM'),('RecFNO','deterministic','RecFNO'),
                 ('VoronoiCNN','deterministic','VoronoiCNN'),('Senseiver','deterministic','Senseiver')]
    for ax,(task,metric,label) in zip(axes.flat,FIELDS):
        for j,(method,estimator,_) in enumerate(comparisons):
            r=next(x for x in pairs if (x['task'],x['variant'],x['estimator'],x['comparator'],x['comparator_estimator'],x['metric'])==
                   (task,'selected','single',method,estimator,metric))
            m,lo,hi=[100*float(r[k]) for k in ['mean','ci_low','ci_high']]
            ax.errorbar(m,j,xerr=np.array([[m-lo],[hi-m]]),fmt='o',color='#24658c',capsize=3,markersize=4)
        ax.axvline(0,color='.4',lw=.8)
        ax.set(yticks=range(4),yticklabels=[c[2] for c in comparisons],xlabel='Selected FM − comparator (percentage points)',
               title=label.replace('$e_u$','u').replace('$e_a$','a'))
        ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    save(fig,a.output/'ns_guidance_paired_fields')
    manifest=dict(loss_calls_checked=len(raw),loss_groups_checked=len(groups),calibration_calls=len(development),
        fields=FIELDS,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in used},
        figure='ns_guidance_paired_fields.pdf',font='Times New Roman',
        reporting='Each joint field is scored separately; source primary-score columns are never used.')
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('EXPORTED field-wise NS tables and paired plot')


if __name__=='__main__':main()
