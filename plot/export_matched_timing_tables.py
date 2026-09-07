"""Present the existing audited inverse timing study as mean/SD tables."""
from pathlib import Path
import argparse,csv,json,hashlib
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--paper',type=Path,required=True);a=p.parse_args();d=a.paper/'source_data'
 read=lambda n:list(csv.DictReader((d/n).open()))
 examples=read('matched_timing_per_example.csv');calls=read('matched_timing_calls.csv');summary=read('matched_timing_summary.csv')
 assert len(calls)==1920
 labels={'fm':'FM4PDE','recfno':'RecFNO','senseiver':'Senseiver','voronoicnn':'VoronoiCNN','pde_opt':'PDE-Opt'}
 lines=[];rows=[]
 for pde in ['poisson','darcy']:
  lines += [r'\begin{table}[!htbp]',r'\centering\small',r'\setlength{\tabcolsep}{5pt}',
   r'\caption{Matched '+pde.capitalize()+r' inverse reconstruction on 32 ID examples. Error is mean $\pm$ sample SD across examples, after averaging three FM inference seeds; latency is mean $\pm$ sample SD across 96 single-example calls. Deterministic methods use three timing repetitions. $N$ is the FM step count and $K$ the optimizer cap.}',rf'\label{{tab:matched-main-{pde}}}',
   r'\begin{tabular}{@{}llrrr@{}}\toprule',r'Method & Budget & Error (\%) & Latency (s) & Opt. steps \\\midrule']
  for s in [r for r in summary if r['pde']==pde]:
   match=lambda r:r['pde']==pde and r['method']==s['method'] and r['budget']==s['budget']
   ee=[float(r['mean_seed_error']) for r in examples if match(r)];tt=[float(r['seconds']) for r in calls if match(r)]
   assert len(ee)==32 and len(tt)==96
   r=dict(pde=pde,method=s['method'],budget=int(s['budget']),n_examples=32,n_calls=96,mean_error=float(np.mean(ee)),sd_error=float(np.std(ee,ddof=1)),mean_seconds=float(np.mean(tt)),sd_seconds=float(np.std(tt,ddof=1)))
   assert np.isclose(r['mean_error'],float(s['mean_error']))
   rows.append(r);budget='$N='+s['budget']+'$' if s['method']=='fm' else '$K='+s['budget']+'$' if s['method']=='pde_opt' else '---'
   latency=f"${r['mean_seconds']:.5f} \\pm {r['sd_seconds']:.5f}$" if r['mean_seconds']<.01 else f"${r['mean_seconds']:.3f} \\pm {r['sd_seconds']:.3f}$"
   lines.append(f"{labels[s['method']]} & {budget} & ${r['mean_error']*100:.2f} \\pm {r['sd_error']*100:.2f}$ & {latency} & "+('10' if s['method']=='pde_opt' else '---')+r' \\')
  lines += [r'\bottomrule\end{tabular}',r'\par\smallskip\parbox{\linewidth}{\footnotesize PDE-Opt returns the zero inverse field in every call. These outcomes are retained at all three caps.}',r'\end{table}',r'\FloatBarrier','']
 f=d/'matched_timing_main_tables.tex';f.write_text('\n'.join(lines))
 manifest=dict(rows=rows,source_sha256={n:hashlib.sha256((d/n).read_bytes()).hexdigest() for n in ['matched_timing_calls.csv','matched_timing_per_example.csv']},table_sha256=hashlib.sha256(f.read_bytes()).hexdigest())
 (d/'matched_timing_main_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':main()
