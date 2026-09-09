#!/usr/bin/env python3
"""Insert the completed Poisson sample-count study into the revision once."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(text, old, new):
    assert text.count(old) == 1, old[:100]
    return text.replace(old, new, 1)


def main(args):
    paper = args.paper.resolve()
    source = paper / 'source_data/conditional_scaling_0909'
    review = json.loads(args.review.read_text())
    assert review['status'] == 'pass' and review['complete']
    assert sha(source / 'conditional_scaling_final_manifest.json') == review['final_manifest_sha256']
    assert sha(source / 'conditional_scaling_per_input.csv') == review['per_input_csv_sha256']
    audit = paper / 'audit/revision_0909/conditional_scaling_final_review'
    numerical = json.loads((audit / 'numerical_review.json').read_text())
    assert numerical['status'] == 'pass' and numerical['numerical_review_complete']
    assert all(sha(Path(path)) == value for path, value in numerical['input_sha256'].items())
    backup = audit / 'before_manuscript_integration'
    assert not backup.exists(), 'This manuscript integration is a one-time operation'
    names = ['fm4pde_jmlr_revision_0906.tex', 'supplement.tex', 'response_to_reviewers_scoped_0906.tex']
    old = {name: (paper / name).read_text() for name in names}
    main_text = old[names[0]]
    assert main_text.count('CONDITIONAL_SAMPLE_COUNT_EXPERIMENT_INSERTION_POINT') == 1
    settings = (source / 'settings.tex').read_text()
    settings = '\n'.join(line for line in settings.splitlines() if not line.startswith('%'))
    discussion = r'''
Table~\ref{tab:conditional-scaling} and
Figure~\ref{fig:conditional-scaling-accuracy} show decreasing mean errors
at each tested sample count. From $K=1$ to $K=1000$, the forward
$\operatorname{RelL2}_u$ decreases from $7.44\%$ to $3.86\%$, and the inverse
$\operatorname{RelL2}_a$ from $22.64\%$ to $17.82\%$. Joint reconstruction
errors decrease from $17.24\%$ to $12.26\%$ for $\mathbf a$ and from
$5.12\%$ to $2.83\%$ for $\mathbf u$. All 16 Bonferroni-adjusted paired
intervals for changes relative to $K=1$ lie below zero. Individual inputs
need not improve monotonically: in Figure~\ref{fig:conditional-scaling-reconstructions},
the displayed joint $\mathbf u$ estimate has its lowest error at $K=3$.

The largest reductions occur at small sample counts. Averaging ten
predictions lowers mean error by $18.2$--$39.3\%$ relative to a single
prediction across the four task--field combinations, accounting for
$81.4$--$86.8\%$ of the total decrease observed at $K=1000$.
Increasing $K$ from 100 to 1000 reduces the four mean errors by only
$0.060$--$0.093$ additional percentage points. These are descriptive
differences; the adjusted intervals compare each larger sample count
with $K=1$, rather than testing adjacent sample counts.

Batching substantially reduces the elapsed-time cost of small averages
(Figure~\ref{fig:conditional-scaling-time}). For $K=3$, median latencies
are $5.38$--$5.55$ s, compared with $5.20$--$5.32$ s for $K=1$.
For $K=10$, they are $9.40$--$9.89$ s; the median within-input ratios to
$K=1$ are $1.86$--$1.89$. Larger averages remain costly: median latency
increases from $79$--$86$ s at $K=100$ to $788$--$859$ s at $K=1000$.
Thus, ten predictions recover much of the observed accuracy gain with
less than twice the median latency of a single prediction in this study,
whereas increasing the sample count beyond 100 gives a small further
decrease in mean error at nearly ten times the latency.
The dashed serial reference is $K$ times the measured single-prediction
latency, not a separate serial timing experiment. These timings include
construction of the averaged physical fields and have a different
boundary from the FM4PDE--DiffusionPDE sampling-time comparison.
The conclusions condition on the tested Poisson model, observations,
guidance settings, and devices. Averaging can reduce sampling fluctuations
without eliminating systematic reconstruction error.
'''
    block = '\n\n'.join([settings, (source / 'conditional_scaling_table.tex').read_text(),
                           discussion, (source / 'conditional_scaling_figures.tex').read_text(),
                           r'\FloatBarrier'])
    main_text = replace_once(main_text, '% CONDITIONAL_SAMPLE_COUNT_EXPERIMENT_INSERTION_POINT', block)
    appendix = (source / 'appendix_settings.tex').read_text()
    main_text = replace_once(main_text, '\\section{Computational Environments}',
                             appendix + '\n\\section{Computational Environments}')
    hardware = r'''The Poisson sample-count study uses one A100 and three A800 GPUs, each
with 80 GB of memory, PyTorch 2.8.0, and CUDA 12.8. Sampling uses float32
with TF32 enabled and batches of at most 64 predictions. All sample counts
for a given input and task use the same device; latency summaries pool
the 32 inputs. Appendix~\ref{app:conditional-averaging} specifies the
timing boundary and the independent executions used to measure latency.

'''
    main_text = replace_once(main_text, 'The original training configurations specify two distributed processes and',
                             hardware + 'The original training configurations specify two distributed processes and')
    prior = ('Averaging three conditional predictions improves reconstruction on nine of eleven PDEs in a paired '
             '32-input study, with three times as many network evaluations.')
    revised = ('Averaging three conditional predictions improves reconstruction on nine of eleven PDEs in a paired '
               '32-input study. Additional Poisson experiments show diminishing gains with sample count; batching '
               'ten predictions lowers mean errors at less than twice the single-prediction median latency.')
    main_text = replace_once(main_text, prior, revised)
    conclusion = r'''The additional Poisson experiment extends the comparison to 1,000
conditional predictions per input. Mean errors decrease at every tested
sample count, with adjusted paired intervals supporting improvement over
one prediction in all four reported task--field combinations. Ten
predictions account for $81.4$--$86.8\%$ of the decrease observed at
1,000 predictions, at a median within-input latency ratio of
$1.86$--$1.89$. Increasing the count from 100 to 1,000 gives only a small
further decrease in mean error at nearly ten times the latency. These
results support small batched averages for the tested Poisson setting;
they do not establish monotonic improvement for every input or comparable
costs on other devices.

'''
    main_text = replace_once(main_text, 'The Poisson multi-condition study further shows that task reuse is not exclusive',
                             conclusion + 'The Poisson multi-condition study further shows that task reuse is not exclusive')
    response = r'''An additional Poisson study uses the same 32 ID inputs for forward,
inverse, and joint reconstruction, with $K=1,3,10,100,1000$ conditional
predictions and 100 stochastic Euler steps per prediction. Errors are
computed from nested averages of a common 1,000-prediction collection for
each input and task. The study contains 96,000 such trajectories;
separate executions at the smaller sample counts provide measured
latencies, for 106,944 trajectories in total. All four mean errors
decrease at each tested sample count, and all 16 adjusted paired intervals
relative to $K=1$ lie below zero. The text also states that individual
inputs need not improve monotonically. Ten predictions account for
$81.4$--$86.8\%$ of the decrease observed at $K=1000$, with median
within-input latency ratios of $1.86$--$1.89$. The small additional
decreases from $K=100$ to $K=1000$ are reported descriptively, without a
claim of statistical significance for that comparison. Three figures
show mean errors, measured latency, and the first prespecified input's
reconstructions. Detailed paired intervals and timing summaries are
provided in the Supplementary Material. The appendix specifies the
independent development inputs, fixed guidance, batching, numerical
precision, and timing boundary.
'''
    response_text = replace_once(old[names[2]], '% NEW_CONDITIONAL_SAMPLE_COUNT_RESPONSE_INSERTION_POINT', response)
    with (source / 'conditional_scaling_paired_effects.csv').open() as stream:
        effects = list(csv.DictReader(stream))
    with (source / 'conditional_scaling_timing.csv').open() as stream:
        timing = list(csv.DictReader(stream))
    task_names = dict(forward='Forward', inverse='Inverse', both='Joint')
    lines = [r'\section{Poisson Conditional-Sample Counts}', r'\label{supp:conditional-scaling}',
             r'The study uses the inputs and nested averages specified in Appendix~\ref{main-app:conditional-averaging}. The main manuscript reports the mean field errors and their sample standard deviations. The following tables give all paired comparisons and measured latency summaries.',
             r'\begin{table}[!htbp]\centering\small\setlength{\tabcolsep}{5pt}',
             r'\begin{tabular}{@{}llrrrr@{}}\toprule',
             r'Task & Field & $K$ & Change (pp) & Adjusted interval (pp) & Improved \\\midrule']
    for row in effects:
        value = f"{float(row['mean_delta_pp']):.3f}"
        if row['K'] == '1000': value = r'\mathbf{' + value + '}'
        if row['K'] == '100': value += r'^{\dagger}'
        lines.append(f"{task_names[row['task']]} & ${row['field']}$ & {row['K']} & ${value}$ & $[{float(row['simultaneous_ci_low']):.3f}, {float(row['simultaneous_ci_high']):.3f}]$ & {row['improved_inputs']}/32 " + r'\\')
    lines += [r'\bottomrule\end{tabular}',
              r'\caption{Paired changes in Poisson relative error from $K=1$, in percentage points. Intervals use 100,000 input-level bootstrap resamples with a Bonferroni adjustment for 16 comparisons. Negative values favor averaging. Improved counts inputs with a strictly smaller error than at $K=1$. Boldface and $\dagger$ mark the lowest and second-lowest distinct mean changes within each task--field combination, ranked before rounding. These comparisons do not test differences between two larger sample counts.}',
              r'\label{tab:conditional-scaling-paired}', r'\end{table}',
              r'\begin{table}[!htbp]\centering\small\setlength{\tabcolsep}{6pt}',
              r'\begin{tabular}{@{}lrrrr@{}}\toprule',
              r'Task & $K$ & Median (s) & $[Q_{0.25},Q_{0.75}]$ (s) & Paired ratio \\\midrule']
    for row in timing:
        value = f"{float(row['median_seconds']):.2f}"
        if row['K'] == '1': value = r'\mathbf{' + value + '}'
        if row['K'] == '3': value += r'^{\dagger}'
        lines.append(f"{task_names[row['task']]} & {row['K']} & ${value}$ & $[{float(row['q25_seconds']):.2f}, {float(row['q75_seconds']):.2f}]$ & ${float(row['median_ratio_to_K1']):.3f}$ " + r'\\')
    lines += [r'\bottomrule\end{tabular}',
              r'\caption{Latency of averaged Poisson estimates over 32 inputs per task. Quartiles describe the measured latencies; the paired ratio is the median of the latency for each input divided by its own $K=1$ latency. The lowest and second-lowest distinct median latencies within each task are bold and marked $\dagger$, respectively. All sample counts for an input and task use the same device. Timing includes batch preparation, generation, physical-field conversion, transfer, and averaging, with batches of at most 64. It excludes model loading and result storage.}',
              r'\label{tab:conditional-scaling-latency}', r'\end{table}', r'\FloatBarrier', '']
    supplement_block = '\n'.join(lines)
    supplement_text = replace_once(old[names[1]], r'\section{Numerical Sampling Times}',
                                   supplement_block + '\n' + r'\section{Numerical Sampling Times}')
    proof = r'\section{A Mechanism-Level Example'
    assert main_text[main_text.index(proof):] == old[names[0]][old[names[0]].index(proof):]
    abstract = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', main_text, re.S).group(1)
    abstract = re.sub(r'(?m)%.*$', '', abstract)
    assert len(abstract.split()) <= 200, len(abstract.split())
    backup.mkdir()
    for name in names:
        (backup / name).write_bytes((paper / name).read_bytes())
    for name, text in zip(names, (main_text, supplement_text, response_text)):
        (paper / name).write_text(text)
    (source / 'conditional_scaling_supplement.tex').write_text(supplement_block)
    report = dict(status='pass', scientific_review_sha256=sha(args.review),
                  script_sha256=sha(Path(__file__)), abstract_words=len(abstract.split()),
                  proof_appendices_unchanged=True,
                  files={name: dict(before_sha256=sha(backup / name), after_sha256=sha(paper / name)) for name in names})
    (audit / 'manuscript_integration.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper', type=Path, required=True)
    parser.add_argument('--review', type=Path, required=True)
    main(parser.parse_args())
