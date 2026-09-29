"""Compare the supplied revision's guidance tables with effective experiment configs.

This checks configuration agreement, not reproduction of measured table values.
"""
import argparse
from collections import defaultdict
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re

from sampling.config import load_config, load_yaml_file

ROOT = Path(__file__).resolve().parents[1]
PDES = {'Poisson': 'poisson', 'Helmholtz': 'helmholtz', 'Darcy': 'darcy',
        'Navier--Stokes': 'nsnonbounded', 'Burgers': 'burger',
        'Burgers (random)': 'burger', 'Burgers (structured)': 'burger',
        'Reaction--diffusion': 'reaction_diffusion', 'Shallow-water': 'shallow_water',
        'Heat': 'heat', 'Wave': 'wave', 'Advection--diffusion': 'advection_diffusion',
        'Steady heat conduction': 'steady_heat_conduction'}
TASKS = {'Forward': 'forward', 'Inverse': 'inverse', 'Joint': 'both'}
WEIGHTS = ('zeta_obs_a', 'zeta_obs_u', 'zeta_pde', 'clip_threshold')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(cell):
    value = cell.strip().strip('$').replace(r'\!', '').replace(r'\times', '*')
    match = re.fullmatch(r'(?:(\d+(?:\.\d+)?)\*)?10\^\{?(-?\d+)\}?', value)
    return float(Decimal(match[1] or 1)*Decimal(10)**int(match[2])) if match else float(value)


def guidance_rows(tex, label):
    # Strip comments without treating escaped percent signs as comments.
    tex = '\n'.join(re.split(r'(?<!\\)%', line, maxsplit=1)[0] for line in tex.splitlines())
    marker = '\\label{' + label + '}'
    require(tex.count(marker) == 1, f'Expected one active table label: {label}')
    table = tex.split(marker, 1)[1].split(r'\end{table}', 1)[0]
    current = None
    rows = []
    for line in table.splitlines():
        cells = [x.strip() for x in line.split('&')]
        if len(cells) != 7 or cells[1] not in TASKS:
            continue
        name = re.sub(r'\\multirow\{[^}]+\}\{[^}]+\}\{([^}]+)\}', r'\1', cells[0])
        if name:
            require(name in PDES, f'Unrecognized PDE in {label}: {name}')
            current = name
        require(current is not None, f'Missing PDE name in {label}')
        rows.append((current, PDES[current], TASKS[cells[1]], [number(x) for x in cells[2:6]]))
    return rows


def validate(manuscript):
    tex = manuscript.read_text()
    main_rows = guidance_rows(tex, 'tab:verified-main-hyperparameters')
    ofm_rows = guidance_rows(tex, 'tab:ofm-guidance-parameters')
    require(len(main_rows) == 32, f'Expected 32 FM4PDE rows, found {len(main_rows)}')
    require(len(ofm_rows) == 9, f'Expected 9 OFM rows, found {len(ofm_rows)}')
    expected = {(pde, task): values for _, pde, task, values in main_rows}
    for name, pde, task, values in main_rows:
        require(expected[pde, task] == values, f'Conflicting guidance rows: {name}/{task}')
    require(len(expected) == 31, 'Missing or duplicate PDE/task rows')

    main = load_yaml_file(ROOT/'configs/experiments/comparison/sparse_forward_inverse.yaml')
    burgers = load_yaml_file(ROOT/'configs/experiments/comparison/burgers_trajectory.yaml')
    consistency = load_yaml_file(ROOT/'configs/experiments/comparison/physical_consistency.yaml')
    checked = []
    for name, pde, task, values in main_rows:
        candidates = [j for j in main['jobs'] if (j['pde'], j['task']) == (pde, task)]
        if pde == 'burger':
            structured = name.endswith('(structured)')
            candidates = [j for j in burgers['jobs']
                          if (j['overrides'].get('sensor_mode') == 'time_slices') == structured]
        if not candidates:
            candidates = [dict(overrides={})]
        for job in candidates:
            cfg = load_config(ROOT/f'configs/main/{task}/{pde}.yaml', job['overrides'])
            actual = [getattr(cfg, key) for key in WEIGHTS]
            require(actual == values, f'{name}/{task}: manuscript {values}, code {actual}')
            require((cfg.num_steps, cfg.stochastic_guidance_coeff, cfg.pde_guidance_start_ratio,
                     cfg.loss_state, cfg.time_grid) == (100, .1, .8, 'endpoint', 'uniform'),
                    f'{name}/{task}: common sampling protocol changed')
            phase = 'hybrid_s2d' if pde == 'nsnonbounded' and task != 'both' else 'stochastic'
            require(cfg.sampler_phase == phase, f'{name}/{task}: wrong sampler phase')
            if phase == 'hybrid_s2d':
                require(cfg.switch_ratio == .5, 'NS main comparison must switch at 0.5')
        checked.append(dict(pde=pde, task=task, manuscript_name=name, values=values, cases=len(candidates)))
    ofm = load_yaml_file(ROOT/'configs/ofm_guidance.yaml')
    for name, pde, task, values in ofm_rows:
        require(ofm[pde][task] == values, f'OFM {name}/{task}: guidance mismatch')

    cohorts = {}
    for label, spec in [('main', main), ('burgers', burgers), ('consistency', consistency)]:
        groups = defaultdict(list)
        job_ids = [job['id'] for job in spec['jobs']]
        require(len(job_ids) == len(set(job_ids)), f'{label}: duplicate job IDs')
        for job in spec['jobs']:
            cfg = load_config(ROOT/f"configs/main/{job['task']}/{job['pde']}.yaml", job['overrides'])
            require([getattr(cfg, key) for key in WEIGHTS] == expected[cfg.pde, cfg.task],
                    f'{label}/{job["id"]}: guidance differs from the manuscript table')
            key = (cfg.pde, cfg.task, cfg.test_type, cfg.sensor_mode)
            groups[key].extend(range(cfg.offset, cfg.offset + cfg.batch_size))
        expected_groups = {'main': 24, 'burgers': 6, 'consistency': 22}[label]
        require(len(groups) == expected_groups, f'{label}: expected {expected_groups} cohorts, found {len(groups)}')
        for key, indices in groups.items():
            require(sorted(indices) == list(range(100)), f'{label}/{key}: missing or duplicate evaluation IDs')
        cohorts[label] = dict(settings=len(groups), realizations=sum(map(len, groups.values())))
    recipe = load_yaml_file(ROOT/'configs/training.yaml')
    require(len(recipe['pdes']) == 11, 'Expected eleven training recipes')
    for pde, cfg in recipe['pdes'].items():
        require(cfg['epochs'] == 300 and cfg['batch_size']*cfg['accum_iter']*recipe['world_size'] == 64,
                f'{pde}: training epochs or effective batch changed')
    return dict(manuscript_sha256=hashlib.sha256(manuscript.read_bytes()).hexdigest(),
                guidance=checked, ofm_rows=len(ofm_rows), cohorts=cohorts, training_recipes=11,
                scope='Configuration agreement only; not numerical table reproduction')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manuscript', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = validate(args.manuscript)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(f"Manuscript guidance: 32 FM4PDE rows, 9 OFM rows; cohorts: {report['cohorts']}; training: 11 recipes.")
    print(report['scope'])
