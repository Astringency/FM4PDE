"""Resolve every archived endpoint-inventory row to its full run configuration."""
from pathlib import Path
import csv
import hashlib
import json
import math
import openpyxl
import yaml

ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT.parents[1]/'C04Papers/fm4pde_jmlr'


def main():
    raw=[r for r in csv.DictReader((ROOT/'outputs/ablations/summary_all_raw.csv').open())
         if r['ablation_group']=='deterministic_endpoint_bt' and r['pde']=='poisson']
    book=openpyxl.load_workbook(PAPER/'source_data/FM4PDE0905_ablations_summary.xlsx',read_only=True,data_only=True)
    values=list(book['deterministic_endpoint_bt'].values)
    rows=[]; configs=[]
    for index,values_row in enumerate(values[1:],2):
        record=dict(zip(values[0],values_row))
        match=[r for r in raw if all(math.isclose(float(r[k]),record[m],rel_tol=1e-10,abs_tol=1e-12)
               for k,m in [('rel_l2_a','rel L2(a)'),('rel_l2_u','rel L2(u)'),('L_pde','pde L')])]
        assert len(match)==1,(index,len(match))
        run=ROOT/match[0]['run_dir']; path=run/'resolved_config.yaml'
        contents=path.read_bytes(); cfg=yaml.safe_load(contents)
        metrics=json.loads((run/'metrics_final.json').read_text())
        for key,metric in [('rel_l2_a','rel L2(a)'),('rel_l2_u','rel L2(u)'),('L_pde','pde L')]:
            assert math.isclose(float(metrics[key]),record[metric],rel_tol=1e-10,abs_tol=1e-12),(index,key)
        row=dict(workbook_row=index,run_dir=str(run),config_sha256=hashlib.sha256(contents).hexdigest())
        for key in ['pde','task','sampler_phase','deterministic_endpoint_mode','deterministic_bt_mode',
                    'deterministic_guidance_coeff','deterministic_bt_max_scale','zeta_pde',
                    'zeta_obs_a','zeta_obs_u','stochastic_guidance_coeff','clip_threshold',
                    'num_steps','sensor_mode','offset','sample_seed','mask_seed','device','checkpoint_path']:
            row[key]=cfg.get(key)
        row.update(rel_l2_a=record['rel L2(a)'],rel_l2_u=record['rel L2(u)'],L_pde=record['pde L'])
        rows.append(row)
        configs.append(dict(workbook_row=index,source_path=str(path),sha256=row['config_sha256'],config=cfg))
    out=PAPER/'source_data/endpoint_inventory_resolved.csv'
    with out.open('w',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    (PAPER/'source_data/endpoint_inventory_configs.json').write_text(json.dumps(configs,indent=2)+'\n')
    lines=[r'\begin{table}[!htbp]',r'\centering\scriptsize',
           r'\caption{Resolved Poisson endpoint-correction inventory (one ID instance per row). Row numbers identify the source workbook. Phase and $\zeta_{\rm pde}$ distinguish the previously duplicated display keys. D/S denote deterministic/stochastic sampling. The coefficient and cap refer to deterministic updates and are inactive in the S row, whose stochastic coefficient is $0.1$. Field errors are percentages; $L_{\rm pde}$ is the archived diagnostic loss.}',
           r'\label{tab:ablation-det-endpoint-complete}',r'\setlength{\tabcolsep}{3pt}',
           r'\begin{tabular}{@{}rllllrrrrrr@{}}\toprule',
           r'Row & Phase & Endpoint & Multiplier & Coeff. & Cap & $\zeta_{\rm pde}$ & $a$ err. & $u$ err. & $L_{\rm pde}$ \\ \midrule']
    for row in rows:
        fields=[str(row['workbook_row']),{'deterministic':'D','stochastic':'S'}[row['sampler_phase']],
                {'single_step':'Single','rollout':'Rollout'}[row['deterministic_endpoint_mode']],
                {'capped_stochastic_like':'Capped S-like','legacy':'Legacy','clipped_zero_at_t0':'Clip-zero',
                 'clipped':'Clipped','stochastic_like':'S-like','t_next':'Next-time','zero_at_t0':'Zero-first'}[row['deterministic_bt_mode']],
                f"{row['deterministic_guidance_coeff']:g}" if row['sampler_phase']=='deterministic' else '--',
                f"{row['deterministic_bt_max_scale']:g}" if row['sampler_phase']=='deterministic' else '--',f"{row['zeta_pde']:g}",
                f"{100*row['rel_l2_a']:.2f}",f"{100*row['rel_l2_u']:.2f}",f"{row['L_pde']:.5g}"]
        lines.append(' & '.join(fields)+r' \\')
    lines.extend([r'\bottomrule\end{tabular}',r'\end{table}'])
    # Ten columns: source ID, three categorical fields, six numeric fields.
    lines=[line.replace('{@{}rllllrrrrrr@{}}','{@{}rlllrrrrrr@{}}') for line in lines]
    (PAPER/'source_data/endpoint_inventory_table.tex').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'resolved_rows':len(rows),'distinct_runs':len({r['run_dir'] for r in rows})}))


if __name__=='__main__': main()
