"""Render tables and loss-form evidence from independently checked extracts."""
from pathlib import Path
import collections
import csv
import gzip
import hashlib
import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT.parents[1]/'C04Papers/fm4pde_jmlr'
SOURCE=PAPER/'source_data'
LABELS={'poisson':'Poisson','helmholtz':'Helmholtz','darcy':'Darcy','nsnonbounded':'Navier--Stokes','burger':'Burgers'}


def read(name):return list(csv.DictReader((SOURCE/name).open()))


def number(value):
    x=float(value)
    if x==0:return '$0$'
    if abs(x)>=10000:
        power=int(math.floor(math.log10(abs(x)))); mantissa=x/10**power
        return f'${mantissa:g}\\!\\times\\!10^{{{power}}}$' if mantissa!=1 else f'$10^{{{power}}}$'
    return f'${x:g}$'


def main_table():
    rows=read('main_hyperparameters_verified.csv')
    cache=SOURCE/'main_gate_trace_audit.json.gz'
    if not cache.exists():
        archive=json.load(gzip.open(SOURCE/'main_configs_archive.json.gz','rt'))
        traces=[]
        for r in archive['records']:
            cfg=r['config']
            if cfg.get('pde_guidance_start_ratio') is not None:continue
            p=Path(r['source']).parent/'curves.csv'; data=p.read_bytes()
            trace=list(csv.DictReader(data.decode().splitlines()))
            assert len(trace)==100
            assert [int(t['step']) for t in trace]==list(range(100))
            assert all(math.isclose(float(t['zeta_pde_t']),float(cfg['zeta_pde']),rel_tol=1e-6) for t in trace)
            traces.append(dict(sheet=r['sheet'],workbook_row=r['workbook_row'],source=str(p),
                               sha256=hashlib.sha256(data).hexdigest(),
                               evidence='Logged physical weight equals configured weight at all 100 steps.',
                               zeta_pde_t=[float(t['zeta_pde_t']) for t in trace]))
        with gzip.open(cache,'wt') as f:json.dump(traces,f)
    traces=json.load(gzip.open(cache,'rt'))
    counts=collections.Counter((r['sheet'],r['workbook_row']) for r in traces)
    for r in rows:
        if r['pde_guidance_start_ratio']=='':
            assert counts[(r['sheet'],r['workbook_row'])]==int(r['archived_batches'])
            r['gate']='All*'
        else:
            assert float(r['pde_guidance_start_ratio'])==.8 and float(r['pde_guidance_ramp_ratio'])==0
            r['gate']='Late'
    keys=['pde','task','observations','zeta_obs_a','zeta_obs_u','zeta_pde','clip_threshold','gate']
    groups=collections.defaultdict(list)
    for r in rows:
        for k in ['zeta_obs_a','zeta_obs_u','zeta_pde','clip_threshold']:r[k]=f'{float(r[k]):g}'
        groups[tuple(r[k] for k in keys)].append(r['distribution'])
    lines=[r'\begin{table}[!htbp]',r'\centering\scriptsize\setlength{\tabcolsep}{3pt}',
           r'\caption{Archived hyperparameters for all 66 FM4PDE main-table cells, checked against 2,892 batch configurations. I/S/R denote ID/Smooth/Rough; F/I/J denote forward/inverse/joint tasks. Random means 500 locations per observed field, columns means five Burgers spatial columns, and full means all input locations. The stochastic coefficient is $c_\zeta=0.1$ throughout. Late activates physics at $t=0.8$ without a ramp. All* denotes an older Smooth run whose constant physical weight was verified in every step log; its config predates the gate fields.}',
           r'\label{tab:verified-main-hyperparameters}',r'\begin{tabular}{@{}llllrrrrl@{}}\toprule',
           r'PDE & Task & Observations & Dist. & $\zeta_a$ & $\zeta_u$ & $\zeta_{\rm pde}$ & Clip & Gate \\ \midrule']
    order={'poisson':0,'helmholtz':1,'darcy':2,'nsnonbounded':3,'burger':4}
    for key,dist in sorted(groups.items(),key=lambda item:(order[item[0][0]],item[0][2],item[0][1],item[0][-1])):
        r=dict(zip(keys,key)); d='/'.join(x[0] for x in ['ID','Smooth','Rough'] if x in dist)
        lines.append(' & '.join([LABELS[r['pde']],{'both':'J','forward':'F','inverse':'I'}[r['task']],
                                {'random':'Random','sensor_column':'Columns','full':'Full'}[r['observations']],d]+
                               [number(r[k]) for k in ['zeta_obs_a','zeta_obs_u','zeta_pde','clip_threshold']]+[r['gate']])+r' \\')
    lines += [r'\bottomrule\end{tabular}',r'\end{table}']
    (SOURCE/'main_hyperparameters_table.tex').write_text('\n'.join(lines)+'\n')
    print('MAIN TABLE',len(groups),'rows;',len(traces),'legacy gate traces checked')


def loss_figure():
    rows=read('loss_comparison_verified.csv'); samples=read('loss_holdout_verified.csv')
    plt.rcParams.update({'font.size':9,'axes.titlesize':10,'axes.labelsize':9,'pdf.fonttype':42,
                         'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(7.9,3.0),layout='constrained',sharey=True)
    intervals=[];rng=np.random.default_rng(20260907)
    for y,row in enumerate(rows):
        pde=row['pde'];byform={f:sorted([r for r in samples if r['pde']==pde and r['form']==f],key=lambda r:int(r['sample_id'])) for f in ['mse','rms']}
        for ax,field,label in zip(axes,['primary_error','pde_residual_norm'],['Error','Physical diagnostic']):
            a=np.array([float(r[field]) for r in byform['mse']]);b=np.array([float(r[field]) for r in byform['rms']])
            indices=rng.integers(0,8,size=(20000,8));boot=100*(b[indices].mean(1)/a[indices].mean(1)-1)
            low,high=np.quantile(boot,[.025,.975]);value=100*(b.mean()/a.mean()-1)
            ax.errorbar(value,y,xerr=[[value-low],[high-value]],fmt='o',color='#24649a',capsize=3,markersize=4)
            intervals.append(dict(pde=pde,metric=field,change_pct=value,ci95_low=low,ci95_high=high,n=8))
    for ax,title in zip(axes,['Reconstruction error','Physical residual diagnostic']):
        ax.axvline(0,color='.55',lw=.8,ls='--');ax.set_title(title,loc='left')
        ax.set_xlabel('RMS / MSE change (%)');ax.grid(axis='x',color='.9',lw=.5);ax.set_axisbelow(True)
    axes[0].set_yticks(range(len(rows)),[LABELS[r['pde']].replace('--','–') for r in rows]);axes[0].invert_yaxis()
    for ext in ['pdf','png']:fig.savefig(PAPER/f'figures/loss_form_holdout.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    with (SOURCE/'loss_holdout_intervals.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(intervals[0]));writer.writeheader();writer.writerows(intervals)
    lines=[r'\begin{table}[!htbp]',r'\centering\small\setlength{\tabcolsep}{4pt}',
           r'\caption{MSE/RMS guidance after separate four-example selection, evaluated on the same eight held-out examples. Entries in the setting columns give gate/$\zeta_{\rm pde}$. L, R and A mean late hard activation, late ramp, and always active. Errors are percentages of the mean-of-fields score, with full-trajectory error for Burgers. These settings and this score differ from the original max-of-fields sweep.}',
           r'\label{tab:loss-form-holdout}',r'\begin{tabular}{@{}lllrrrr@{}}\toprule',
           r'PDE & MSE setting & RMS setting & MSE err. & RMS err. & MSE residual & RMS residual \\ \midrule']
    for r in rows:
        setting=lambda f: {'late_hard':'L','late_ramp':'R','always_on':'A'}[r[f+'_schedule']]+'/'+f"{float(r[f+'_zeta']):g}"
        lines.append(' & '.join([LABELS[r['pde']],setting('mse'),setting('rms'),f"{100*float(r['mse_error']):.4f}",
                                f"{100*float(r['rms_error']):.4f}",f"{float(r['mse_residual']):.5f}",f"{float(r['rms_residual']):.5f}"])+r' \\')
    lines += [r'\bottomrule\end{tabular}',r'\end{table}']
    (SOURCE/'loss_form_table.tex').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main_table();loss_figure()
