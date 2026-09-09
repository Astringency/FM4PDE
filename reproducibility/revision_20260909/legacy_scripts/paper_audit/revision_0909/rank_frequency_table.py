from pathlib import Path
import csv
base=Path('/home/tat512/C04Papers/fm4pde_jmlr')
rows=list(csv.DictReader((base/'source_data/frequency_band_summary.csv').open()))
methods=['FM4PDE','RecFNO','Senseiver','VoronoiCNN']
body=[]
for pde in ['poisson','darcy']:
    cells={m:[] for m in methods}
    for band,metric,kind in [('low','relative_l2','error'),('high','relative_l2','error'),('high','energy_ratio','ratio'),('high','alignment','alignment')]:
        selected={m:next(r for r in rows if r['pde']==pde and r['method']==m and r['band']==band and r['metric']==metric and r['low_cutoff']=='8.0' and r['high_cutoff']=='32.0') for m in methods}
        scores={m:(abs(float(r['mean'])-1) if kind=='ratio' else -float(r['mean']) if kind=='alignment' else float(r['mean'])) for m,r in selected.items()}
        ranks=sorted(set(scores.values()))
        for m,r in selected.items():
            mean=float(r['mean']);sd=float(r['sd'])
            text=f'{100*mean:.2f}' if kind=='error' else f'{mean:.3f}'
            if scores[m]==ranks[0]:text=r'\mathbf{'+text+'}'
            elif scores[m]==ranks[1]:text+=r'^{\dagger}'
            if kind=='error':text+=rf'\pm{100*sd:.2f}'
            cells[m].append('$'+text+'$')
    if body:body.append(r'\midrule')
    for i,m in enumerate(methods):body.append(' & '.join([pde.title() if i==0 else '',m,*cells[m]])+r' \\')
text=r'''\begin{table}[!htbp]
\centering\footnotesize\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}llrrrr@{}}\toprule
PDE & Method & Low error (\%) & High error (\%) & $q_H$ & $A_H$ \\\midrule
'''+ '\n'.join(body)+r'''
\bottomrule\end{tabular}
\caption{Frequency diagnostics at cutoffs 8 and 32. FM4PDE uses 100 steps. Band errors are mean $\pm$ sample SD over 32 input-level seed averages, in percent. The low band excludes DC; the high band has radius above 32. Boldface and $\dagger$ mark the best and second-best method within each PDE and metric, using unrounded values: smaller band errors, an energy ratio $q_H$ closer to one, and larger coefficient alignment $A_H$ are preferred.}
\label{tab:frequency-bands}
\end{table}
'''
(base/'audit/revision_0909/frequency_table_ranked.tex').write_text(text)
