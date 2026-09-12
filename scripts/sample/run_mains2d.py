"""Run the existing main sweeps with S -> D at 0.2 and audit all 42 cells."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[2]
PDES = ['poisson', 'helmholtz', 'darcy', 'nsnonbounded']
TASKS = ['forward', 'inverse', 'both']
DISTRIBUTIONS = ['id', 'smooth', 'rough']


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def cells():
    return [dict(test_type=d, pde=p, task=t, sensor_mode=s)
            for d in DISTRIBUTIONS
            for p, t, s in ([(p, t, 'random') for p in PDES for t in TASKS] +
                            [('burger', 'both', s) for s in ['random', 'sensor_column']])]


def prepare(output, source):
    configs, weights, data = {}, {}, {}
    for pde, task in [(p, t) for p in PDES for t in TASKS] + [('burger', 'both')]:
        relative = Path(task) / (pde + '.yaml')
        original_path = source / 'configs/main' / relative
        assert original_path.read_bytes() == (ROOT / 'configs/main' / relative).read_bytes()
        config = yaml.safe_load(original_path.read_text())
        checkpoint = source / config['checkpoint_path']
        assert checkpoint.is_file(), checkpoint
        if str(checkpoint) not in weights:
            weights[str(checkpoint)] = dict(size=checkpoint.stat().st_size, sha256=sha(checkpoint))
        for distribution in DISTRIBUTIONS:
            path = Path(config['data_paths'][distribution])
            assert path.is_file(), path
            data[str(path)] = dict(size=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
        config.update(sampler_phase='hybrid_s2d', switch_ratio=0.2, checkpoint_path=str(checkpoint))
        destination = output / 'configs' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        serialized = yaml.safe_dump(config, sort_keys=False)
        if destination.exists():
            assert destination.read_text() == serialized, destination
        else:
            destination.write_text(serialized)
        configs[str(relative)] = dict(source_sha256=sha(original_path), experiment_sha256=sha(destination))
    manifest = dict(code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    source_root=str(source), output_root=str(output), sampler_phase='hybrid_s2d',
                    switch_ratio=0.2, num_steps=100, stochastic_steps=20, deterministic_steps=80,
                    samples_per_cell=1000, offsets=[0, 999], sample_seed=0, num_obs=500,
                    num_sensor_columns_burger=16, cells=cells(), total_sample_results=42000,
                    configs=configs, weights=weights, data=data)
    if (output / 'experiment.json').exists():
        assert json.loads((output / 'experiment.json').read_text()) == manifest
    write(output / 'experiment.json', manifest)
    print('Prepared 42 cells, 42000 samples; S20 -> D80.', flush=True)


def sweep(output, label, *, distribution='id', pdes=None, tasks=None, burger=False,
          count=1000, batch=10, devices='cuda:0 cuda:1', parallel=True, aggregate=False, pilot=False,
          workers=2):
    env = os.environ.copy()
    env.update(NUM_SAMPLES=str(count), MAX_BATCH_SIZE=str(batch), NUM_STEPS='100', NUM_OBS='500',
               SAMPLE_SEED='0', SAMPLER_LIST='hybrid_s2d', SENSOR_MODE_LIST='random',
               BURGER_SENSOR_MODE_LIST='random sensor_column',
               TEST_TYPE=distribution, CONFIG_DIR=str(output / 'configs'),
               OUTPUT_DIR=str(output / ('_pilot/' + label if pilot else distribution)),
               DEVICE_LIST=devices, PARALLEL=str(parallel).lower(), MAX_PARALLEL_TASKS=str(workers) if parallel else '1',
               RESUME='true', VIS='false', AGGREGATE=str(aggregate).lower(), PLAN_ONLY='false',
               DRY_RUN='false', PROGRESS_INTERVAL='30', PDE_LIST=' '.join(pdes or PDES),
               TASK_LIST=' '.join(tasks or TASKS))
    script = 'scripts/sample/run_sample_sweep_burger.sh' if burger else 'scripts/sample/run_sample_sweep.sh'
    execution = output / 'execution'
    execution.mkdir(parents=True, exist_ok=True)
    receipt = execution / (label + '.json')
    previous = json.loads(receipt.read_text()) if receipt.exists() else None
    if previous and previous.get('exit_code') == 0:
        print('Already finished:', label, flush=True)
        return
    write(execution / (label + '.command.json'), dict(command=['bash', script], cwd=str(ROOT),
          environment={k: env[k] for k in ['NUM_SAMPLES', 'MAX_BATCH_SIZE', 'NUM_STEPS', 'NUM_OBS', 'SAMPLE_SEED',
          'SAMPLER_LIST', 'SENSOR_MODE_LIST', 'BURGER_SENSOR_MODE_LIST', 'TEST_TYPE', 'CONFIG_DIR', 'OUTPUT_DIR',
          'DEVICE_LIST', 'PARALLEL', 'MAX_PARALLEL_TASKS', 'RESUME', 'VIS', 'AGGREGATE', 'PDE_LIST', 'TASK_LIST']}))
    start = time.time()
    print('Starting:', label, 'batch', batch, flush=True)
    with (execution / (label + '.log')).open('a') as stream:
        code = subprocess.run(['bash', script], cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT).returncode
    write(receipt, dict(label=label, exit_code=code, started_at=start, finished_at=time.time()))
    assert code == 0, f'{label} exited {code}; inspect {execution / (label + ".log")}'
    print('Finished:', label, flush=True)


def pilot(output):
    choice = output / 'execution/batch_size.json'
    if choice.exists():
        return json.loads(choice.read_text())['batch_size']
    memory = output / 'execution/pilot_gpu_memory.csv'
    memory.parent.mkdir(parents=True, exist_ok=True)
    with memory.open('w') as stream:
        monitor = subprocess.Popen(['nvidia-smi', '--id=0', '--query-gpu=memory.used,memory.total',
                                    '--format=csv,noheader,nounits', '--loop-ms=250'], stdout=stream)
        try:
            sweep(output, 'pilot_b1', pdes=PDES + ['burger'], tasks=['both'], count=1, batch=1,
                  devices='cuda:0', parallel=False, pilot=True)
        finally:
            monitor.terminate(); monitor.wait()
    observations = list(csv.reader(memory.read_text().splitlines()))
    peak = max(float(row[0]) for row in observations)
    total = min(float(row[1]) for row in observations)
    candidate = min(10, int(0.6 * total / peak))
    assert candidate >= 1 and peak < 0.5 * total, (peak, total)
    memory = output / 'execution/pilot_batch_gpu_memory.csv'
    with memory.open('w') as stream:
        monitor = subprocess.Popen(['nvidia-smi', '--id=0', '--query-gpu=memory.used,memory.total',
                                    '--format=csv,noheader,nounits', '--loop-ms=250'], stdout=stream)
        try:
            sweep(output, 'pilot_batch', pdes=PDES + ['burger'], tasks=['both'], count=candidate, batch=candidate,
                  devices='cuda:0', parallel=False, pilot=True)
        finally:
            monitor.terminate(); monitor.wait()
    peak_batch = max(float(row[0]) for row in csv.reader(memory.read_text().splitlines()))
    assert peak_batch < 0.7 * total, (peak_batch, total)
    validate_phases(output / '_pilot')
    write(choice, dict(batch_size=candidate, pilot_b1_peak_mib=peak,
                       pilot_batch_peak_mib=peak_batch, gpu_total_mib=total,
                       workers=2, workers_per_gpu=1))
    print('Pilot passed:', candidate, 'samples/batch; peak MiB', peak_batch, flush=True)
    return candidate


def validate_phases(root):
    count = 0
    for path in root.rglob('metrics_step.jsonl'):
        steps = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(steps) == 100, path
        assert [r['phase'] for r in steps] == ['stochastic'] * 20 + ['deterministic'] * 80, path
        count += 1
    assert count, root
    return count


def audit(output):
    summaries = []
    excluded = []
    total_runs = 0
    for case in cells():
        root = output / case['test_type'] / case['pde'] / case['task']
        samples = {}
        runs = 0
        for config_path in root.rglob('resolved_config.yaml'):
            config = yaml.safe_load(config_path.read_text())
            if config['sensor_mode'] != case['sensor_mode']:
                continue
            assert all(config[k] == v for k, v in case.items()), config_path
            assert config['sampler_phase'] == 'hybrid_s2d' and config['switch_ratio'] == 0.2
            assert config['num_steps'] == 100 and config['sample_seed'] == 0
            assert config['num_obs'] == 500 and config['allow_synthetic_data'] is False
            if case['sensor_mode'] == 'sensor_column':
                assert config['num_sensor_columns'] == 16
            folder = config_path.parent
            if not (folder / 'metrics_final.json').exists():
                excluded.append(dict(run_dir=str(folder), reason='unfinished attempt; no final metrics'))
                continue
            metrics = json.loads((folder / 'metrics_final.json').read_text())
            if metrics['status'] != 'ok':
                excluded.append(dict(run_dir=str(folder), reason='unsuccessful attempt: ' + metrics['status']))
                continue
            assert not metrics['synthetic_data'], folder
            assert (folder / 'result.pt').is_file(), folder
            assert validate_phases(folder) == 1
            with (folder / 'metrics_per_sample.csv').open() as stream:
                batch = list(csv.DictReader(stream))
            assert len(batch) == config['batch_size']
            assert {int(r['sample_id']) for r in batch} == set(range(config['offset'], config['offset'] + config['batch_size']))
            for row in batch:
                sample_id = int(row['sample_id'])
                assert sample_id not in samples, (case, sample_id)
                errors = [float(row['rel_l2_a']), float(row['rel_l2_u'])]
                assert all(math.isfinite(x) for x in errors), (case, sample_id)
                samples[sample_id] = errors
            runs += 1
        assert set(samples) == set(range(1000)), (case, len(samples))
        summaries.append(dict(**case, samples=len(samples), runs=runs,
                              a_relative_error_pct=100*sum(v[0] for v in samples.values())/1000,
                              u_relative_error_pct=100*sum(v[1] for v in samples.values())/1000))
        total_runs += runs
    write(output / 'completion.json', dict(status='complete', verified_cells=42, verified_samples=42000,
          verified_runs=total_runs, sampler_phase='hybrid_s2d', switch_ratio=0.2, rows=summaries,
          excluded_attempts=excluded))
    with (output / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader(); writer.writerows(summaries)
    print('Verified all 42 cells and 42000 sample results.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--phase', choices=['prepare', 'pilot', 'full', 'audit', 'all'], default='all')
    parser.add_argument('--workers', type=int, choices=[2, 4], default=2)
    args = parser.parse_args()
    assert args.output.is_absolute() and args.source_root.is_absolute()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.phase in ['prepare', 'all']:
        prepare(args.output, args.source_root)
    if args.phase in ['pilot', 'all']:
        pilot(args.output)
    if args.phase in ['full', 'all']:
        memory = json.loads((args.output / 'execution/batch_size.json').read_text())
        batch = memory['batch_size']
        assert (args.workers // 2) * memory['pilot_batch_peak_mib'] < .7 * memory['gpu_total_mib']
        write(args.output / 'execution/concurrency.json', dict(workers=args.workers,
              workers_per_gpu=args.workers // 2, batch_size=batch,
              conservative_gpu_peak_mib=(args.workers // 2)*memory['pilot_batch_peak_mib']))
        for distribution in DISTRIBUTIONS:
            sweep(args.output, distribution + '_main', distribution=distribution, batch=batch, workers=args.workers)
            sweep(args.output, distribution + '_burger', distribution=distribution, batch=batch,
                  burger=True, aggregate=True, workers=args.workers)
        audit(args.output)
    elif args.phase == 'audit':
        audit(args.output)


if __name__ == '__main__':
    main()
