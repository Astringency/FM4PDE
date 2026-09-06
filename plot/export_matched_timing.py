"""Export a fully audited matched-timing report and its complete result tables."""
import argparse
import csv
import json
from pathlib import Path
import shutil
from run_revision_sampling import digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--paper',type=Path,required=True)
    args=p.parse_args()
    manifest=json.loads((args.report/'matched_timing_manifest.json').read_text())
    assert manifest['calls_verified']==1920
    assert manifest['protocol_sha256']==digest(args.inputs/'protocol.json')
    for name,sha in manifest['outputs'].items():assert digest(args.report/name)==sha,name
    rows=list(csv.DictReader((args.report/'matched_timing_summary.csv').open()))
    assert len(rows)==20 and {r['pde'] for r in rows}=={'poisson','darcy'}
    names={'fm':'FM4PDE','recfno':'RecFNO','senseiver':'Senseiver','voronoicnn':'VoronoiCNN','pde_opt':'PDE-Opt'}
    tex=['% Generated from the audited 1,920-call matched timing report.']
    for pde in ['poisson','darcy']:
        tex += [r'\begin{table}[!htbp]',r'\centering\footnotesize',
                r'\caption{Matched '+pde.capitalize()+r' inverse reconstruction: 32 physical examples, three FM inference seeds or three deterministic timing repetitions. Error intervals are 95\% physical-example bootstrap intervals; latency brackets give IQRs in seconds. $N$ is the FM step count; $K$ is the PDE-Opt iteration cap.}',
                r'\label{tab:matched-timing-'+pde+r'}',r'\begin{tabular}{@{}llrrr@{}}\toprule',
                r'Method & Budget & Error (\%) [CI] & Latency (s) [IQR] & Opt. steps \\\midrule']
        for r in [r for r in rows if r['pde']==pde]:
            method=r['method'];budget=int(r['budget'])
            b=f'$N={budget}$' if method=='fm' else f'$K={budget}$' if method=='pde_opt' else '---'
            e=[100*float(r[k]) for k in ['mean_error','ci95_low','ci95_high']]
            t=[float(r[k]) for k in ['median_seconds','q25_seconds','q75_seconds']]
            lo,hi=int(r['optimizer_steps_min']),int(r['optimizer_steps_max'])
            steps=(str(lo) if lo==hi else f'{lo}--{hi}') if method=='pde_opt' else '---'
            tex.append(f'{names[method]} & {b} & {e[0]:.2f} [{e[1]:.2f}, {e[2]:.2f}] & {t[0]:.3g} [{t[1]:.3g}, {t[2]:.3g}] & {steps} '+r'\\')
        zero=[r for r in rows if r['pde']==pde and r['method']=='pde_opt']
        tex += [r'\bottomrule\end{tabular}',r'\par\smallskip\parbox{\linewidth}{\footnotesize '
                +'PDE-Opt returns an exactly zero inverse field in '+', '.join(f"{r['zero_prediction_calls']}/96" for r in zero)
                +' calls at caps 50, 100, and 500, respectively. These calls remain in all summaries; they do not characterize a fully converged optimizer.}',r'\end{table}']
    source=args.paper/'source_data';figures=args.paper/'figures'
    source.mkdir(parents=True,exist_ok=True);figures.mkdir(parents=True,exist_ok=True)
    for path in args.report.iterdir():
        if path.suffix in {'.pdf','.png'}:shutil.copy2(path,figures/path.name)
        elif path.is_file():shutil.copy2(path,source/path.name)
    shutil.copy2(args.inputs/'protocol.json',source/'matched_timing_protocol.json')
    (source/'matched_timing_tables.tex').write_text('\n'.join(tex)+'\n')
    print('EXPORTED complete matched timing report and 20 result rows')


if __name__=='__main__':main()
