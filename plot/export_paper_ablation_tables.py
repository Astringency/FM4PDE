"""Fieldwise LaTeX tables for the main ablation inventory."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from run_paper_ablation_revision import PDES, digest
from export_paper_seed_ensemble import NAMES

PHASES = [('stochastic','S'),('deterministic','D'),('hybrid_d2s',r'D$\to$S'),('hybrid_s2d',r'S$\to$D')]


def tex_number(value):
    if value is None or not np.isfinite(value):
        return r'\text{nonfinite}'
    if abs(value) >= 1e4:
        mantissa, exponent = f'{value:.3e}'.split('e')
        return f'{mantissa}\\times10^{{{int(exponent)}}}'
    if 0 < abs(value) < .005:
        if abs(value) >= .0001:
            return f'{value:.2g}'
        mantissa, exponent = f'{value:.1e}'.split('e')
        return f'{mantissa}\\times10^{{{int(exponent)}}}'
    return f'{value:.2f}'


def table(caption, label, header, rows):
    return '\n'.join([
        r'\begin{table}[!htbp]',rf'\FMTableMark{{start}}{{{label}}}',
        r'\centering\scriptsize\setlength{\tabcolsep}{3pt}',
        r'\renewcommand{\arraystretch}{1.10}',r'\caption{'+caption+'}',
        r'\label{'+label+'}',r'\begin{tabular}{@{}'+'ll'+'r'*(len(header)-2)+r'@{}}\toprule',
        ' & '.join(header)+r' \\\midrule',
        *[' & '.join(row)+r' \\' for row in rows],r'\bottomrule\end{tabular}',
        rf'\FMTableMark{{end}}{{{label}}}',r'\end{table}',''])


def export(args):
    args.output.mkdir(parents=True,exist_ok=True)
    records=json.loads((args.source/'records.json').read_text())
    manifest=json.loads((args.source/'manifest.json').read_text())
    assert manifest['final_ready'] or args.development, 'Main ablation reruns are not complete'
    def get(pde,group,**filters):
        candidates=[r for r in records if r['pde']==pde and r['config']['ablation_group']==group and
                    all(r['config'].get(k)==v for k,v in filters.items())]
        assert len(candidates)==1,(pde,group,filters,len(candidates))
        return candidates[0]
    fields=[(pde,field) for pde in PDES for field in (['u'] if pde=='burger' else ['a','u'])]
    outputs={}
    def save(name,caption,label,columns,group,pdes=PDES,base=None):
        rows=[]
        for pde,field in fields:
            if pde not in pdes:continue
            values=[]
            for _,query in columns:
                r=get(pde,group,**dict({'task':'both'},**(base or {}),**query))
                values.append('$'+tex_number(100*r['rel_l2_'+field])+'$')
            rows.append([NAMES[pde],f'${field}$',*values])
        text=table(caption,label,['PDE','Field',*[c[0] for c in columns]],rows)
        (args.output/name).write_text(text)
        outputs[name]=dict(label=label,rows=len(rows),group=group)
    units='Relative field errors in percent on the main ID sample; coefficient and solution are reported separately. '

    # Directional targets and both joint fields, including Burgers trajectory u.
    rows=[]
    for task,title in [('forward','Forward'),('inverse','Inverse'),('both','Joint')]:
        for pde in PDES:
            if pde=='burger' and task!='both':continue
            scored=['u'] if task=='forward' or pde=='burger' else ['a'] if task=='inverse' else ['a','u']
            for field in scored:
                values=['$'+tex_number(100*get(pde,'guidance_components',task=task,guidance_components=g)['rel_l2_'+field])+'$'
                        for g in ['noguide','pde_only','obs_only','obs_pde']]
                rows.append([NAMES[pde],title+f' ${field}$',*values])
    name='ablation_guidance_fields.tex'
    (args.output/name).write_text(table(units+'Guidance components at 100 stochastic steps.',
        'tab:ablation-guidance-complete',['PDE','Task / field','No guide','PDE only','Obs. only','Obs.+PDE'],rows))
    outputs[name]=dict(label='tab:ablation-guidance-complete',rows=len(rows),group='guidance_components')

    save('ablation_loss_state_fields.tex',units+'Loss evaluation at the current, next, or endpoint state. S and D denote stochastic and deterministic sampling.',
         'tab:ablation-loss-state',[(phase_label+' / '+state_label,dict(sampler_phase=phase,loss_state=state))
             for phase,phase_label in PHASES[:2] for state,state_label in [('xt',r'$x_t$'),('x_next',r'$x_{t+\Delta t}$'),('endpoint','endpoint')]],'loss_state_by_sampler')
    columns=[(label,dict(sampler_phase=phase)) for phase,label in PHASES[:2]]
    columns += [(label+f' {ratio}',dict(sampler_phase=phase,switch_ratio=ratio)) for phase,label in PHASES[2:] for ratio in [.2,.5,.8]]
    save('ablation_phase_fields.tex',units+'Sampler phases at 100 steps. Hybrid labels give the switch ratio.',
         'tab:ablation-sampler-complete',columns,'sampler_phase')
    for phase,phase_label in PHASES:
        label={'stochastic':'tab:ablation-numsteps-complete','deterministic':'tab:ablation-numsteps-deterministic',
               'hybrid_d2s':'tab:ablation-numsteps-hybrid-d2s','hybrid_s2d':'tab:ablation-numsteps-hybrid-s2d'}[phase]
        save(f'ablation_steps_{phase}_fields.tex',units+f'Step-count sweep for {phase_label}; guidance coefficients remain fixed.',
             label,[(str(n),dict(num_steps=n)) for n in [10,50,100,200,500,1000,2000]],
             'num_steps_by_sampler',base=dict(sampler_phase=phase))
    save('ablation_coverage_fields.tex',units+'Number of observed locations per active field; a shared location measures every component of a vector field.',
         'tab:ablation-sensor-count',[(str(n),dict(num_obs=n)) for n in [50,100,250,500,1000]],'sensor_sparsity')
    for phase,phase_label in PHASES:
        save(f'ablation_grid_{phase}_fields.tex',units+f'Time-grid comparison for {phase_label} at 100 steps.',
             'tab:ablation-grid-'+phase,[(label,dict(time_grid=key)) for key,label in [('uniform','Uniform'),('cosine','Cosine'),('geometric','Geometric')]],
             'time_grid_by_sampler',base=dict(sampler_phase=phase))
    save('ablation_integrator_fields.tex',units+'Euler and midpoint updates at 100 steps, for each sampler phase.',
         'tab:ablation-integrator',[(r'\shortstack{'+label+r'\\'+method.title()+'}',dict(sampler_phase=phase,step_method=method)) for phase,label in PHASES for method in ['euler','midpoint']],
         'step_method_by_sampler')
    save('ablation_layout_fields.tex',units+'Observation layouts at the configured sensor budget.',
         'tab:ablation-sensor-layout',[(label,dict(sensor_mode=key)) for key,label in [('random','Random'),('per_sample_random','Per-input random'),('fixed','Fixed'),('grid','Grid'),('sensor_column','Columns')]],'sensor_mode')
    save('ablation_noise_fields.tex',units+'Observation-noise amplitudes; errors use the clean reference field.',
         'tab:ablation-noise',[(f'{x:g}',dict(noise_level=x)) for x in [0.,.01,.05,.1]],'noise_robustness')
    save('ablation_temporal_fields.tex',units+'Temporal residuals. The near-endpoint variant also uses sparse auxiliary time observations.',
         'tab:ablation-temporal',[(label,dict(residual_mode=key)) for key,label in [('endpoint_secant','Secant'),('hermite_bridge','Hermite'),('near_endpoint_temporal','Near-endpoint')]],
         'temporal_residual_mode',pdes=['nsnonbounded','reaction_diffusion','shallow_water','heat','wave','advection_diffusion'])
    rows=[]
    for pde,field in fields:
        values=np.array([100*get(pde,'statistics_stability',task='both',sample_seed=seed)['rel_l2_'+field] for seed in range(5)])
        rows.append([NAMES[pde],f'${field}$',*[f'${tex_number(x)}$' for x in values],f'${tex_number(values.mean())}\\pm{tex_number(values.std(ddof=1))}$'])
    name='ablation_stability_fields.tex'
    (args.output/name).write_text(table('Inference and observation-mask seeds on the main ID sample. Each field error is a percentage; the final column gives the mean and sample SD across five seeds.',
        'tab:ablation-stability',['PDE','Field',*[str(x) for x in range(5)],r'Mean $\pm$ SD'],rows))
    outputs[name]=dict(label='tab:ablation-stability',rows=len(rows),group='statistics_stability')
    (args.output/'table_manifest.json').write_text(json.dumps(dict(final_ready=manifest['final_ready'],
        source_sha256=digest(args.source/'records.json'),tables=outputs,
        exporter_sha256=digest(Path(__file__))),indent=2)+'\n')
    print('EXPORTED',len(outputs),'fieldwise tables',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--development',action='store_true')
    export(parser.parse_args())
