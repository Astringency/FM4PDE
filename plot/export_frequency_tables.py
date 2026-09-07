"""Export the complete matched frequency summaries to manuscript tables."""
import argparse,csv,json
from pathlib import Path
from run_ns_loss_study import sha,write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True);p.add_argument('--paper',type=Path,required=True)
    a=p.parse_args();source=a.paper/'source_data'
    m=json.loads((a.report/'paired_evidence_manifest.json').read_text())
    for n,h in m['outputs'].items():assert sha(a.report/n)==h
    rows=list(csv.DictReader((a.report/'frequency_band_summary.csv').open()))
    def get(pde,method,band,metric,hi=32):
        r=next((r for r in rows if r['pde']==pde and r['method']==method and r['band']==band and r['metric']==metric and float(r['high_cutoff'])==hi),None)
        if r is None:return None
        assert int(r['examples'])==32
        return r
    def mean(r,factor=1,decimals=3):return '---' if r is None else f"{factor*float(r['mean']):.{decimals}f}"
    def sd(r,factor=1,decimals=2):return '---' if r is None else f"${factor*float(r['mean']):.{decimals}f}\\pm{factor*float(r['sd']):.{decimals}f}$"
    methods=['FM4PDE','RecFNO','Senseiver','VoronoiCNN']
    lines=[r'\begin{table}[!htbp]',r'\centering\footnotesize\setlength{\tabcolsep}{3pt}',
        r'\caption{Frequency diagnostics at the predeclared cutoffs 8 and 32. FM4PDE uses 100 steps. Band errors are percentages, reported as mean $\pm$ sample SD over 32 input-level seed averages. $q_H$ is the mean high-band energy ratio and $A_H$ is mean coefficient alignment. Low excludes DC; high has radius above 32.}',
        r'\label{tab:frequency-bands}',r'\begin{tabular}{@{}llrrrr@{}}\toprule',
        r'PDE & Method & Low error (\%) & High error (\%) & $q_H$ & $A_H$ \\\midrule']
    for pde in ['poisson','darcy']:
        for j,method in enumerate(methods):
            lines.append(' & '.join([pde.title() if j==0 else '',method,sd(get(pde,method,'low','relative_l2'),100),sd(get(pde,method,'high','relative_l2'),100),mean(get(pde,method,'high','energy_ratio')),mean(get(pde,method,'high','alignment'))])+r' \\')
        if pde=='poisson':lines.append(r'\midrule')
    lines += [r'\bottomrule\end{tabular}',r'\end{table}']
    (source/'frequency_band_table.tex').write_text('\n'.join(lines)+'\n')
    lines=[]
    for pde in ['poisson','darcy']:
        lines += [r'\begin{table}[!htbp]',r'\centering\scriptsize\setlength{\tabcolsep}{3pt}',
            r'\caption{Complete '+pde.title()+r' frequency decomposition at cutoffs 8 and 32. Reference share and band errors are percentages; $D_B/E$ is the contribution to full relative squared error. $q_B$ and $A_B$ denote energy ratio and coefficient alignment. Values are input-level means; errors include sample SD. A dash marks an undefined ratio for negligible reference-band energy.}',
            r'\label{tab:frequency-complete-'+pde+'}',r'\begin{tabular}{@{}llrrrrr@{}}\toprule',
            r'Method & Band & Ref. share (\%) & Band error (\%) & $D_B/E$ & $q_B$ & $A_B$ \\\midrule']
        for method in methods:
            for j,band in enumerate(['dc','low','mid','high']):
                vals=[method if j==0 else '',band.upper() if band=='dc' else band,mean(get(pde,method,band,'reference_fraction'),100,3),sd(get(pde,method,band,'relative_l2'),100),f"{float(get(pde,method,band,'global_squared_error')['mean']):.3g}",mean(get(pde,method,band,'energy_ratio')),mean(get(pde,method,band,'alignment'))]
                lines.append(' & '.join(vals)+r' \\')
            if method!=methods[-1]:lines.append(r'\addlinespace')
        lines += [r'\bottomrule\end{tabular}',r'\end{table}']
    effects=list(csv.DictReader((a.report/'frequency_paired_effects.csv').open()))
    lines += [r'\begin{table}[!htbp]',r'\centering\scriptsize\setlength{\tabcolsep}{3pt}',
        r'\caption{High-band cutoff sensitivity for FM4PDE (100 steps) versus RecFNO. Differences are FM minus RecFNO. Brackets give paired 95\% pointwise bootstrap intervals over 32 inputs. Negative energy-mismatch differences favor FM4PDE; positive band-error differences favor RecFNO. Full comparisons with every baseline are retained in the source CSV.}',
        r'\label{tab:frequency-cutoff-sensitivity}',r'\begin{tabular}{@{}llrr@{}}\toprule',
        r'PDE & High radius & Energy-mismatch difference & Band-error difference (pp) \\\midrule']
    for pde in ['poisson','darcy']:
        for low,high in [(4,16),(8,32),(16,48)]:
            vals=[]
            for metric,factor,dec in [('energy_mismatch',1,3),('relative_l2',100,2)]:
                r=next(r for r in effects if r['pde']==pde and r['comparison']=='FM4PDE minus RecFNO' and r['band']==f'high: {low}/{high}' and r['metric']==metric)
                vals.append(f"${factor*float(r['difference']):.{dec}f}$ [$ {factor*float(r['ci_low']):.{dec}f}, {factor*float(r['ci_high']):.{dec}f} $]")
            lines.append(' & '.join([pde.title(),f'$>{high}$']+vals)+r' \\')
    lines += [r'\bottomrule\end{tabular}',r'\end{table}']
    (source/'frequency_complete_tables.tex').write_text('\n'.join(lines)+'\n')
    write(source/'frequency_table_export.json',dict(examples=32,source_hashes=m['outputs'],script_sha256=sha(Path(__file__)),
        outputs={n:sha(source/n) for n in ['frequency_band_table.tex','frequency_complete_tables.tex']}))


if __name__=='__main__':main()
