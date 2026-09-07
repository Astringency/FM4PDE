"""Reproduce JMLR figures and DiffusionPDE tables from frozen, traceable data.

Usage: python plot/plot_jmlr_revision.py --paper /path/to/fm4pde_jmlr
       Add --snapshot-training once to copy original checkpoint training logs.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
import numpy as np
import openpyxl
from publication_style import use_times_new_roman

FMROOT = Path(__file__).resolve().parents[1]
PDES = ['poisson', 'helmholtz', 'darcy', 'nsnonbounded', 'burger',
        'reaction_diffusion', 'shallow_water', 'heat', 'wave',
        'advection_diffusion', 'steady_heat_conduction']
LABELS = dict(zip(PDES, ['Poisson', 'Helmholtz', 'Darcy', 'Navier–Stokes', 'Burgers',
                       'Reaction–diff.', 'Shallow water', 'Heat', 'Wave', 'Adv.–diff.', 'Steady heat']))
DIST = ['ID', 'Smooth', 'Rough']
BLUE, GOLD, ORANGE, OLIVE = '#28628F', '#A77B19', '#BF6634', '#697540'
COLORS = [BLUE, GOLD, ORANGE, OLIVE]
CMAP = LinearSegmentedColormap.from_list('paper_blue', ['#F5F8FA', '#A9C8DF', '#28628F', '#102D44'])
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 10,
                     'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
                     'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.linewidth': .6,
                     'grid.color': '#DDDDDD', 'grid.linewidth': .5})
use_times_new_roman()


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f'Empty data for {path}')
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def canonical(name):
    return {'ns': 'nsnonbounded', 'burgers': 'burger'}.get(name.lower(), name.lower())


def save(fig, paper, stem):
    target = paper / 'figures'
    target.mkdir(exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(target / f'{stem}.{ext}', dpi=180, bbox_inches='tight',
                    metadata={'Creator': 'plot_jmlr_revision.py'})
    plt.close(fig)


def verify_workbook_rows(paper, rows):
    """Verify plotted audit values against the actual supplied workbook cells."""
    books = {}
    for row in rows:
        filename = row['file']
        if filename not in books:
            books[filename] = openpyxl.load_workbook(paper / 'source_data' / filename,
                                                   read_only=True, data_only=True)
        sheet = books[filename][row['sheet']]
        headers = next(sheet.values)
        metric = row.get('source_metric') or row['metric']
        value = sheet.cell(int(row['row']), headers.index(metric) + 1).value
        assert np.isclose(float(value), float(row['mean']), rtol=1e-10, atol=1e-12), row
    for book in books.values():
        book.close()


def main_comparisons(paper):
    all_rows = read_csv(paper / 'audit/table_value_provenance.csv')
    specs = [
        ('full-forward-results', 'Full forward: solution', ['FNO', 'DeepONet', 'IFNO', 'FM4PDE'], 'rel L2(u)'),
        ('full-inverse-results', 'Full inverse: input', ['IFNO', 'FM4PDE'], 'rel L2(a)'),
        ('sparse-forward-results', 'Sparse forward: solution', ['RecFNO', 'Senseiver', 'VoronoiCNN', 'FM4PDE'], 'rel L2(u)'),
        ('sparse-inverse-results', 'Sparse inverse: input', ['RecFNO', 'Senseiver', 'VoronoiCNN', 'FM4PDE'], 'rel L2(a)'),
        ('sparse-joint-results', 'Sparse joint: input', ['RecFNO', 'Senseiver', 'VoronoiCNN', 'FM4PDE'], 'rel L2(a)'),
        ('sparse-joint-results', 'Sparse joint: solution', ['RecFNO', 'Senseiver', 'VoronoiCNN', 'FM4PDE'], 'rel L2(u)'),
    ]
    plotted = []
    # Designed for approximately six-inch journal text width after inclusion.
    fig, axes = plt.subplots(3, 2, figsize=(8.0, 9.2), layout='constrained')
    for ax, (table, title, methods, metric) in zip(axes.flat, specs):
        rows = [r for r in all_rows if r['table'] == 'tab:' + table and r['metric'] == metric]
        plotted.extend(rows)
        lookup = {(canonical(r['PDE']), r['distribution'], r['method']): float(r['mean']) * 100 for r in rows}
        expected = [(p, d) for p in PDES[:4] for d in DIST]
        values = np.array([[lookup[(p, d, m)] for m in methods] for p, d in expected])
        im = ax.imshow(values, norm=LogNorm(.1, 100), cmap=CMAP, aspect='auto')
        ax.set_xticks(range(len(methods)), ['iFNO' if m == 'IFNO' else m for m in methods], rotation=28, ha='right')
        ax.set_yticks(range(12), [f'{LABELS[p]} / {d}' for p, d in expected])
        ax.tick_params(labelsize=9.5)
        ax.set_title(title, loc='left', pad=9, fontsize=11)
        for i in range(12):
            for j in range(len(methods)):
                val = values[i, j]
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=9.5,
                        color='white' if val > 9 else '#222222')
    bar=fig.colorbar(im, ax=axes, shrink=.7, extend='both')
    bar.set_label('Mean relative L2 error (%) · logarithmic color scale',fontsize=10)
    bar.ax.tick_params(labelsize=9.5)
    fig.suptitle('Archived paired-field benchmarks\nReported means; sparse sensor protocols differ across methods', fontsize=11.5)
    verify_workbook_rows(paper, plotted)
    write_csv(paper / 'source_data/main_figure_values.csv', plotted)
    save(fig, paper, 'main_comparisons')

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.5), layout='constrained')
    rows = [r for r in all_rows if r['table'] == 'tab:physics-smooth-results']
    methods = ['PINN-Sparse', 'PDE-Opt', 'PC-BNN', 'FM4PDE']
    for j, method in enumerate(methods):
        sub = [r for r in rows if r['method'] == method]
        lookup = {(canonical(r['PDE']), r['task']): float(r['mean']) * 100 for r in sub}
        keys = [(p, t) for p in PDES[:3] for t in ['forward', 'inverse']]
        axes[0].plot([lookup[k] for k in keys], np.arange(6) + (j-1.5)*.16,
                     ['o', 's', '^', 'D'][j], color=COLORS[j], ms=4, label=method)
    axes[0].set_yticks(range(6), [f'{LABELS[p]} / {t}' for p, t in keys])
    axes[0].invert_yaxis()
    axes[0].set_xscale('log')
    axes[0].set_title('Physics-based reconstruction\nSmooth distribution', loc='left',fontsize=10.5)
    axes[0].set_xlabel('Mean target-field error (%)',fontsize=10)
    axes[0].legend(fontsize=9, loc='upper center', bbox_to_anchor=(.5,-.18),
                   ncols=2, frameon=False)
    br = [r for r in all_rows if r['table'] == 'tab:burgers-results']
    bm = ['RecFNO', 'VoronoiCNN', 'FM4PDE', 'Senseiver', 'Var4D', 'VIVID']
    for j, dist in enumerate(DIST):
        lookup = {r['method']: float(r['mean']) * 100 for r in br if r['distribution'] == dist}
        axes[1].plot([lookup[m] for m in bm], np.arange(6) + (j-1)*.18,
                     ['o','s','^'][j], color=COLORS[j], ms=4, label=dist)
    axes[1].set_yticks(range(6), ['4D-Var' if m == 'Var4D' else m for m in bm])
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Mean full-trajectory error (%)',fontsize=10)
    axes[1].set_xlim(left=0)
    axes[1].set_title('Burgers trajectory\n500 random space–time observations', loc='left',fontsize=10.5)
    axes[1].legend(fontsize=9, loc='upper center', bbox_to_anchor=(.5,-.18),
                   ncols=3, frameon=False)
    for ax in axes:
        ax.tick_params(labelsize=9.5)
        ax.grid(axis='x', alpha=.7)
    fig.suptitle('Archived reconstruction errors\nReported means; descriptive comparison under the recorded protocols', fontsize=11)
    verify_workbook_rows(paper, rows + br)
    save(fig, paper, 'physics_burgers_comparisons')


def ablation_figures(paper):
    rows = read_csv(paper / 'audit/ablation_metrics_tidy.csv')
    grouped = defaultdict(dict)
    for row in rows:
        grouped[(row['sheet'], int(row['row']))][row['metric']] = row
    points = []
    for (sheet, idx), metrics in grouped.items():
        if 'RelL2(u_traj)' in metrics:
            a = u = metrics['RelL2(u_traj)']
        elif 'rel L2(a)' in metrics and 'rel L2(u)' in metrics:
            a, u = metrics['rel L2(a)'], metrics['rel L2(u)']
        else:
            continue
        cfg = json.loads(a['configuration'])
        points.append(dict(sheet=sheet, row=idx, pde=a['PDE'], task=a['task'], config=cfg,
                           error=100*(float(u['mean']) if a['PDE']=='burger' else max(float(a['mean']),float(u['mean'])))))
    phases = [('stochastic', .5), ('deterministic', .5)] + [(s, r) for s in ['hybrid_d2s','hybrid_s2d'] for r in [.2,.5,.8]]
    fig, ax = plt.subplots(figsize=(10, 5.1), layout='constrained')
    sub = [p for p in points if p['sheet']=='sampler_phase']
    lookup = {(p['pde'], p['config']['SAMPLER'], float(p['config']['SWITCH RATIO'])):p['error'] for p in sub}
    matrix = np.array([[lookup[(p,s,r)] for s,r in phases] for p in PDES])
    im = ax.imshow(matrix, norm=LogNorm(.05,2500), cmap=CMAP, aspect='auto')
    ax.set_yticks(range(len(PDES)), [LABELS[p] for p in PDES])
    ax.set_xticks(range(8), ['S','D','D→S .2','D→S .5','D→S .8','S→D .2','S→D .5','S→D .8'])
    for i in range(11):
        for j in range(8):
            value=matrix[i,j]
            ax.text(j,i,f'{value:.2f}',ha='center',va='center',fontsize=8,color='white' if value>25 else '#222222')
    ax.set_title('Sampler phase and switch fraction\nOne ID instance per PDE; maximum field error, or full trajectory for Burgers',loc='left')
    fig.colorbar(im, ax=ax, label='Primary relative error (%) · log scale', shrink=.85)
    save(fig,paper,'sampler_phase')

    fig, axes=plt.subplots(3,4,figsize=(12.6,8),layout='constrained')
    samplers=['stochastic','deterministic','hybrid_d2s','hybrid_s2d']
    for ax,pde in zip(axes.flat,PDES):
        for j,s in enumerate(samplers):
            sub=sorted([p for p in points if p['sheet']=='num_steps_by_sampler' and p['pde']==pde and p['config']['SAMPLER']==s],key=lambda p:p['config']['NUM STEPS'])
            ax.plot([p['config']['NUM STEPS'] for p in sub],[p['error'] for p in sub],
                    marker=['o','s','^','D'][j],ms=3,lw=1,color=COLORS[j],label=s.replace('hybrid_','').replace('2','→'))
        ax.set_title(LABELS[pde],loc='left')
        ax.set_xscale('log'); ax.set_yscale('log'); ax.grid(alpha=.6)
        ax.set_xlabel('Sampling steps'); ax.set_ylabel('Primary error (%)')
    handles,labels=axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].axis('off'); axes.flat[-1].legend(handles,labels,loc='center',frameon=False)
    fig.suptitle('Sampling-budget sensitivity\nOne ID instance per PDE; fixed configured coefficients, varying guidance and reinjection budgets',fontsize=12)
    save(fig,paper,'sampling_steps')

    fig,axes=plt.subplots(1,2,figsize=(11.2,5),layout='constrained')
    for ax,sampler in zip(axes,['stochastic','deterministic']):
        for j,state in enumerate(['xt','x_next','endpoint']):
            sub=[p for p in points if p['sheet']=='loss_state_by_sampler' and p['config']['SAMPLER']==sampler and p['config']['LOSS STATE']==state]
            lookup={p['pde']:p['error'] for p in sub}
            ax.plot([lookup[p] for p in PDES],np.arange(11)+(j-1)*.17,['o','s','^'][j],color=COLORS[j],ms=4,label={'xt':'Current state','x_next':'Proposal','endpoint':'Endpoint'}[state])
        ax.set_yticks(range(11),[LABELS[p] for p in PDES]); ax.invert_yaxis(); ax.set_xscale('log')
        ax.set_xlabel('Primary relative error (%)'); ax.set_title(sampler.capitalize(),loc='left'); ax.grid(axis='x',alpha=.6)
    axes[1].legend(fontsize=8)
    fig.suptitle('State used to evaluate guidance\nOne ID instance per PDE; maximum field error, or full trajectory for Burgers',fontsize=11)
    save(fig,paper,'loss_state')


def training_curves(paper,snapshot=False):
    folder=paper/'source_data/training_logs'; folder.mkdir(exist_ok=True)
    if snapshot:
        summary=(FMROOT/'outputs/pretrained/training_summary.md').read_text()
        manifest=[]
        for rel in re.findall(r'`(formal/[^`]+/)`',summary):
            run=FMROOT/'outputs/pretrained'/rel
            pde=Path(rel).parts[1]
            source=run/f'{pde}_log.txt'
            raw=source.read_bytes(); target=folder/source.name
            if target.exists() and target.read_bytes()!=raw:
                raise ValueError(f'Changed frozen log: {target}')
            target.write_bytes(raw)
            manifest.append(dict(pde=pde,source=str(source),sha256=hashlib.sha256(raw).hexdigest()))
        (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if not (folder/'manifest.json').exists():
        return
    fig,axes=plt.subplots(4,3,figsize=(8.4,8.7),layout='constrained')
    export=[]
    for ax,pde in zip(axes.flat,PDES):
        rows=[json.loads(line) for line in (folder/f'{pde}_log.txt').read_text().splitlines() if line.startswith('{')]
        epochs=[r['epoch']+1 for r in rows]
        assert epochs==sorted(set(epochs)),pde
        for key,color,style in [('train_loss',BLUE,'-'),('val_loss',GOLD,'--')]:
            ax.plot(epochs,[r[key] for r in rows],color=color,lw=.85,ls=style,label=key.replace('_loss',''))
        for r in rows:
            export.append(dict(pde=pde,epoch=r['epoch']+1,train_loss=r['train_loss'],val_loss=r['val_loss'],lr=r['lr']))
        ax.set_title(LABELS[pde],loc='left'); ax.set_xlabel('Epoch (one-based)'); ax.set_ylabel('Flow-matching MSE')
        ax.set_xlim(0,300); ax.set_yscale('log'); ax.grid(alpha=.5)
    axes.flat[-1].axis('off')
    axes.flat[-1].legend(*axes.flat[0].get_legend_handles_labels(),loc='upper left',bbox_to_anchor=(0,1),frameon=False)
    axes.flat[-1].text(0,.62,'Missing pre-resume epochs\nare left blank.\n\nOne training run per PDE;\nno seed uncertainty.',va='top',transform=axes.flat[-1].transAxes,fontsize=9)
    fig.suptitle('Archived training and validation losses\nOriginal checkpoint directories; each panel has its own logarithmic loss scale',fontsize=12)
    write_csv(paper/'source_data/training_curves.csv',export)
    save(fig,paper,'training_curves')


def diffusion_summary(paper):
    with gzip.open(paper/'source_data/diffusion_snapshot_20260906.json.gz','rt') as stream:
        snapshot=json.load(stream)
    assert not snapshot['read_errors'],snapshot['read_errors']
    groups=defaultdict(list)
    for r in snapshot['records']:
        groups[(r['pde'],r['problem'],r['steps'])].append(r)
    summary=[]
    for inventory in snapshot['inventory']:
        pde,task,steps=inventory['pde'],inventory['task'],inventory['steps']
        records=groups[(pde,task,steps)]
        ids=[r['offset'] for r in records]
        assert len(ids)==len(set(ids)),(pde,task,steps,'duplicate IDs')
        for metric in ['rel_l2_a','rel_l2_u','wall_clock_time']:
            values=np.array([r.get(metric,np.nan) for r in records],dtype=float)
            finite=np.isfinite(values)
            complete=len(records)==1000 and set(ids)==set(range(1000)) and finite.all()
            row=dict(pde=pde,task=task,distribution='Smooth',steps=steps,metric=metric,
                     n_records=len(records),n_finite=int(finite.sum()),n_nonfinite=int((~finite).sum()),
                     sample_files=inventory['sample_files'],status='complete' if complete else 'not_evaluated' if not records else 'incomplete',
                     mean=float(values.mean()) if complete else '',sd=float(values.std(ddof=1)) if complete else '',
                     source_snapshot='diffusion_snapshot_20260906.json.gz')
            summary.append(row)
    write_csv(paper/'source_data/diffusion_summary.csv',summary)
    lookup={(r['pde'],r['task'],r['steps'],r['metric']):r for r in summary}
    def cell(pde,task,step,metric):
        row=lookup[(pde,task,step,metric)]
        return f"${100*row['mean']:.2f} \\pm {100*row['sd']:.2f}$" if row['status']=='complete' else '--'
    lines=[r'\begin{table}[!htbp]',r'\centering\footnotesize',
           r'\caption{DiffusionPDE on the legacy Smooth test distribution: mean $\pm$ SD relative error (\%). Each numerical cell uses 1,000 distinct examples (offsets 0--999). A dash means that the corresponding complete evaluation was unavailable in the 6 September snapshot; it is not a zero error.}',
           r'\label{tab:diffusion-smooth}',r'\setlength{\tabcolsep}{4pt}',
           r'\begin{tabular}{@{}lllrrr@{}}\toprule',r'PDE & Task & Target & 100 steps & 1000 steps & 2000 steps \\ \midrule']
    for pde in PDES[:5]:
        for task in (['both'] if pde=='burger' else ['forward','inverse','both']):
            metrics=['rel_l2_u'] if task=='forward' or pde=='burger' else ['rel_l2_a'] if task=='inverse' else ['rel_l2_a','rel_l2_u']
            for metric in metrics:
                target=r'$\mathbf u_{\rm traj}$' if pde=='burger' else '$a$' if metric.endswith('_a') else '$u$'
                name=LABELS[pde].replace('–','--')
                lines.append(' & '.join([name,{'both':'Joint','forward':'Forward','inverse':'Inverse'}[task],target]+[cell(pde,task,s,metric) for s in [100,1000,2000]])+r' \\')
    lines += [r'\bottomrule\end{tabular}',r'\end{table}']
    (paper/'source_data/diffusion_table.tex').write_text('\n'.join(lines)+'\n')
    return {'complete_cells':len([r for r in summary if r['metric']=='rel_l2_u' and r['status']=='complete']),
            'records':len(snapshot['records']),'snapshot_finished':snapshot['finished_utc']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper',type=Path,default=FMROOT.parents[1]/'C04Papers/fm4pde_jmlr')
    parser.add_argument('--snapshot-training',action='store_true')
    args=parser.parse_args(); paper=args.paper.resolve()
    main_comparisons(paper)
    ablation_figures(paper)
    training_curves(paper,args.snapshot_training)
    diffusion=diffusion_summary(paper)
    sources=[p for p in (paper/'source_data').rglob('*') if p.is_file()]
    manifest={'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'diffusion':diffusion,'matplotlib_version':matplotlib.__version__,
              'sources':{str(p.relative_to(paper)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)}}
    (paper/'audit/figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'figures':sorted(p.name for p in (paper/'figures').glob('*.pdf')),'diffusion':diffusion},indent=2))


if __name__=='__main__':
    main()
