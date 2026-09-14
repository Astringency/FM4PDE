"""Measure S80+D20 throughput with existing sampling scripts and batch size 10."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import subprocess
import time

from run_maind2s import ROOT, PDES, TASKS, write, validate_phases
from verify_maind2s import verify_tensors


def benchmark(output, label, cases, workers):
    execution = output / 'execution'
    target = execution / (label + '.json')
    if target.exists():
        previous = json.loads(target.read_text())
        assert previous['status'] == 'complete' and previous['workers'] == workers
        return previous
    slots = queue.Queue()
    for index in range(workers):
        slots.put(f'cuda:{index % 2}')
    root = output / '_pilot' / label
    root.mkdir(parents=True, exist_ok=True)
    memory_path = execution / (label + '_gpu.csv')
    with memory_path.open('w') as memory:
        monitor = subprocess.Popen([
            'nvidia-smi', '--query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu',
            '--format=csv,noheader,nounits', '--loop-ms=500'], stdout=memory)
        started = time.time()

        def run(index, case):
            device = slots.get()
            pde, task, sensor = case
            job = root / f'job_{index:02d}'
            job.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env.update(PDE=pde, TASK=task, SENSOR_MODE=sensor, SAMPLER_PHASE='hybrid_s2d',
                BATCH_SIZE='10', OFFSET=str(10 * index), NUM_STEPS='100', NUM_OBS='500',
                SAMPLE_SEED='0', TEST_TYPE='id', DEVICE=device, VIS='false', DRY_RUN='false',
                CONFIG_DIR=str(output / 'configs'), OUTPUT_DIR=str(job))
            command = ['bash', 'scripts/sample/run_sample.sh']
            write(job / 'command.json', dict(command=command, cwd=str(ROOT),
                environment={k: env[k] for k in ['PDE', 'TASK', 'SENSOR_MODE', 'SAMPLER_PHASE',
                'BATCH_SIZE', 'OFFSET', 'NUM_STEPS', 'NUM_OBS', 'SAMPLE_SEED', 'TEST_TYPE',
                'DEVICE', 'VIS', 'DRY_RUN', 'CONFIG_DIR', 'OUTPUT_DIR']}))
            start = time.time()
            try:
                with (job / 'runner.log').open('w') as stream:
                    code = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                          stderr=subprocess.STDOUT).returncode
                finish = time.time()
                assert code == 0, (job, code)
                paths = list(job.rglob('metrics_final.json'))
                assert len(paths) == 1, (job, paths)
                metrics = json.loads(paths[0].read_text())
                assert metrics['status'] == 'ok' and not metrics['synthetic_data'], job
                assert metrics['pde_eval_error_count'] == 0, job
                assert validate_phases(job) == 1
                import yaml
                config = yaml.safe_load((paths[0].parent / 'resolved_config.yaml').read_text())
                assert config['switch_ratio'] == .8 and config['sampler_phase'] == 'hybrid_s2d'
                assert config['batch_size'] == 10 and config['offset'] == index * 10
                with (paths[0].parent / 'metrics_per_sample.csv').open() as stream:
                    rows = list(csv.DictReader(stream))
                assert len(rows) == 10
                assert {int(row['sample_id']) for row in rows} == set(range(index * 10, index * 10 + 10))
                result = dict(index=index, pde=pde, task=task, sensor_mode=sensor,
                    device=device, exit_code=code, wall_seconds=finish-start,
                    sampling_seconds=metrics['wall_clock_time'], samples=10, run_dir=str(paths[0].parent))
                print(label, index, pde, task, f'{finish-start:.1f}s', flush=True)
                return result
            finally:
                slots.put(device)

        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(run, index, case) for index, case in enumerate(cases)]
                results = [future.result() for future in futures]
            finished = time.time()
        finally:
            monitor.terminate()
            monitor.wait()
    readings = list(csv.reader(memory_path.read_text().splitlines()))
    peaks = {str(gpu): max(float(row[2]) for row in readings if int(row[1]) == gpu) for gpu in (0, 1)}
    total = min(float(row[3]) for row in readings)
    report = dict(status='complete', label=label, workers=workers, batch_size=10,
        started_at=started, finished_at=finished, wall_seconds=finished-started,
        samples=sum(row['samples'] for row in results), peak_memory_mib=peaks,
        gpu_total_mib=total, samples_per_second=sum(row['samples'] for row in results)/(finished-started),
        runs=results)
    write(target, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    # All scientific task/layout configurations, followed by the same 12 heavy
    # jobs at two concurrency levels. No benchmark samples enter formal results.
    cases = [(p, t, 'random') for p in PDES for t in TASKS]
    cases += [('burger', 'both', mode) for mode in ('random', 'sensor_column')]
    coverage = benchmark(output, 'pilot_all_tasks_w4', cases, 4)
    heavy = [('nsnonbounded', 'both', 'random')] * 12
    four = benchmark(output, 'pilot_ns_w4', heavy, 4)
    six = benchmark(output, 'pilot_ns_w6', heavy, 6)
    speedup = six['samples_per_second'] / four['samples_per_second']
    six_safe = max(six['peak_memory_mib'].values()) < .9 * six['gpu_total_mib']
    selected = 6 if six_safe and speedup >= 1.05 else 4
    chosen = six if selected == 6 else four
    peak_per_slot = max(chosen['peak_memory_mib'].values()) / (selected // 2)
    assert max(chosen['peak_memory_mib'].values()) < .9 * chosen['gpu_total_mib']
    # Validate saved predictions outside the timed region.
    import yaml
    maximum = 0.0
    for report in (coverage, four, six):
        for row in report['runs']:
            folder = Path(row['run_dir'])
            config = yaml.safe_load((folder / 'resolved_config.yaml').read_text())
            with (folder / 'metrics_per_sample.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            maximum = max(maximum, verify_tensors(folder, config, rows))
    write(output / 'execution/batch_size.json', dict(batch_size=10, workers=selected,
        workers_per_gpu=selected//2, gpu_total_mib=chosen['gpu_total_mib'],
        pilot_batch_peak_mib=peak_per_slot, memory_basis='Measured full-card peak divided by concurrent slots.',
        full_card_peak_mib=chosen['peak_memory_mib']))
    projected_four_hours = sum(row['wall_seconds'] for row in coverage['runs']) * 100 * 3 / 4 / 3600
    write(output / 'execution/pilot_validation.json', dict(
        checked_utc=datetime.now(timezone.utc).isoformat(), status='passed', verified_runs=38,
        verified_samples=380, sampler_phase='hybrid_s2d', switch_ratio=.8,
        stochastic_steps=80, deterministic_steps=20, selected_workers=selected,
        selected_batch_size=10, ns_throughput_speedup_6_over_4=speedup,
        six_worker_memory_safe=six_safe, tensor_metric_relative_difference_max=maximum,
        rough_four_worker_projection_hours=projected_four_hours,
        projection_limitations='One batch per task, ID only; projection is not a deadline guarantee and excludes scheduling tails.',
        five_hour_target_supported=projected_four_hours / max(1.0, speedup) <= 5))
    print('PILOT PASSED; selected workers:', selected, 'NS speedup:', speedup,
          'rough 4-worker hours:', projected_four_hours, flush=True)


if __name__ == '__main__':
    main()
