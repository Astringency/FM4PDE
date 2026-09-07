"""Integrate only a complete, audited controlled timing export into the paper."""
import argparse,hashlib,json
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--paper',type=Path,required=True);a=p.parse_args();paper=a.paper
 manifest=json.loads((paper/'source_data/diffusion_fm_timing_manifest.json').read_text())
 assert manifest['calls_verified']==400 and manifest['settings_verified']==20
 for name,sha in manifest['outputs'].items():assert hashlib.sha256((paper/name).read_bytes()).hexdigest()==sha
 rows=manifest['summary'];labels={'poisson':'Poisson','helmholtz':'Helmholtz','darcy':'Darcy','nsnonbounded':'Navier--Stokes','burger':'Burgers'}
 lookup={(r['pde'],r['method'],r['steps']):r for r in rows}
 sentence=[]
 for n in [100,1000]:
  ratios=[lookup[(p,'DiffusionPDE',n)]['mean_seconds']/lookup[(p,'FM4PDE',n)]['mean_seconds'] for p in labels]
  if min(ratios)>1:
   sentence.append(f'At {n:,} steps, FM4PDE has lower mean latency on all five PDEs; the\nDiffusionPDE-to-FM4PDE mean-time ratio ranges from ${min(ratios):.2f}$ to ${max(ratios):.2f}$.')
  else:
   faster=[labels[p] for p,r in zip(labels,ratios) if r>1]
   sentence.append(f'At {n:,} steps, FM4PDE has lower mean latency on {len(faster)} of the five\nPDEs. The DiffusionPDE-to-FM4PDE mean-time ratio ranges from\n${min(ratios):.2f}$ to ${max(ratios):.2f}$.')
 text=r'''\subsection{FM4PDE and DiffusionPDE Sampling Time}
\label{sec:diffusion-fm-timing}

Figure~\ref{fig:diffusion-fm-timing} compares single-sample sampling time at
100 and 1,000 steps. Each of the 20 settings uses the same 20 preselected
Smooth inputs, yielding 400 timed calls. For each PDE, both methods run on
the same physical RTX 4090 with common observations and masks, batch size
one, float32 arithmetic, and a full warm-up per method and budget. Calls
are interleaved in a predeclared random order. GPU/process telemetry checks
for competing compute work throughout each call. Error bars show sample
standard deviations over the 20 inputs, and the labels give mean seconds.

'''+ '\n'.join(sentence)+r'''
The comparison includes each method's initialization, guidance gradients,
and physical decoding. It excludes model loading, observation preparation,
input transfer, diagnostics, and file writes. CUDA synchronization brackets
the sampling call. DiffusionPDE retains Heun integration and therefore uses
$2N-1$ forward network calls at $N$ steps, compared with $N$ for FM4PDE;
network architectures and guidance computations also differ. Equal nominal
step counts consequently do not equate computational work.

These are controlled-precision sampling times. DiffusionPDE's native
float64 sampler-state and residual operations are converted to float32,
while the archived accuracy tables retain their original arithmetic and
observation draws. The timing results characterize the specified sampling
implementations and do not establish a speed advantage at equal accuracy.
Appendix~\ref{app:diffusion-fm-timing} gives the complete protocol and numeric
means and SDs. The preceding matched inverse study separately shows that
specialized sparse predictors can be much faster than either iterative
sampling procedure; training and tuning costs are excluded throughout.

\input{figures/diffusion_fm_timing.tex}
\FloatBarrier
'''
 # The earlier study times FM against specialized predictors, not Diffusion.
 text=text.replace('much faster than either iterative\nsampling procedure','much faster than FM4PDE at its tested\nsampling budgets')
 f=paper/'source_data/diffusion_fm_timing_discussion.tex';f.write_text(text)
 for name in ['fm4pde_jmlr_revision_0906.tex','fm4pde_jmlr_revision_scoped_reviewer_map_0906.tex']:
  f=paper/name;s=f.read_text();needle='\\input{source_data/matched_timing_discussion.tex}'
  if '\\input{source_data/diffusion_fm_timing_discussion.tex}' not in s:
   assert s.count(needle)==1;s=s.replace(needle,needle+'\n\\input{source_data/diffusion_fm_timing_discussion.tex}')
  needle='\\input{source_data/matched_timing_tables.tex}'
  if '\\input{source_data/diffusion_fm_timing_protocol.tex}' not in s:
   assert s.count(needle)==1;s=s.replace(needle,needle+'\n\\FloatBarrier\n\\input{source_data/diffusion_fm_timing_protocol.tex}')
  f.write_text(s)
 f=paper/'response_to_reviewers_scoped_0906.tex';s=f.read_text();needle='\nThe endpoint inventory is also resolved:'
 addition=r'''
The added FM4PDE--DiffusionPDE timing comparison contains 400 single-sample
calls: five PDEs, two methods, two budgets (100/1,000 steps), and 20 inputs
per setting. Each PDE uses the same physical RTX 4090 for both methods,
common inputs and masks, float32, one full warm-up per setting, interleaved
calls, synchronized timing, and continuous GPU/process telemetry. Grouped
bars show mean time with sample-SD error bars and numeric mean labels.
The protocol explicitly records DiffusionPDE's native-float64-to-float32
conversion and the $N$ versus $2N-1$ network-call counts. All predictions,
masks, truths, call counts and source hashes pass the independent export
audit. This is a controlled-precision latency comparison; it does not align
the archived accuracy draws or establish equal-accuracy speedups.
'''
 if 'The added FM4PDE--DiffusionPDE timing comparison contains 400' not in s:
  assert needle in s;s=s.replace(needle,'\n'+addition+needle)
 f.write_text(s)
 print('INTEGRATED audited 400-call timing results; final build and visual review still required')

if __name__=='__main__':main()
