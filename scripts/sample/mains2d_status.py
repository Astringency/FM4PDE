"""Read live main-sweep processes and successful chunk artifacts."""
import argparse
import datetime
import json
from pathlib import Path
import sys
import time


def snapshot(root):
    expected = json.loads((root / 'experiment.json').read_text())
    counts = {}
    active = []
    for distribution in ['id', 'smooth', 'rough']:
        for marker in (root / distribution).glob('.sample_sweeps/*/completed/**/*.json'):
            record = json.loads(marker.read_text())
            final = Path(record['run_dir']) / 'metrics_final.json'
            if not final.is_file() or json.loads(final.read_text()).get('status') != 'ok':
                continue
            key = distribution + '/' + record['experiment']
            counts.setdefault(key, set()).update(range(record['offset'], record['offset'] + record['batch_size']))
        for path in (root / distribution).glob('.sample_sweeps/*/progress/*.json'):
            record = json.loads(path.read_text())
            if record.get('status') in ['running', 'initializing']:
                active.append(dict(distribution=distribution, experiment=path.stem,
                    **{k: record.get(k) for k in ['status', 'offset', 'batch_size', 'step', 'num_steps']}))
    drivers = []
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        try:
            args = (process / 'cmdline').read_text().split('\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(arg.endswith('/scripts/sample/run_mains2d.py') for arg in args) and str(root) in args:
            drivers.append(int(process.name))
    exit_file = root / 'execution/driver.exit'
    exit_code = int(exit_file.read_text().strip()) if exit_file.exists() else None
    complete = root / 'completion.json'
    verified = (complete.is_file() and json.loads(complete.read_text()).get('verified_samples') == 42000
                and sum(map(len, counts.values())) == 42000)
    state = 'complete' if verified else 'running' if drivers else 'stopped'
    result = dict(status=state, checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  driver_pids=drivers, driver_exit_code=exit_code, tmux='fm4pde_mains2d_0913',
                  total_sample_results=expected['total_sample_results'],
                  completed_sample_results=sum(map(len, counts.values())),
                  completed_by_cell={k: len(v) for k, v in sorted(counts.items())}, active=active,
                  output_root=str(root))
    temporary = root / 'status.json.tmp'
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(root / 'status.json')
    return result


def table(result):
    print(result['checked_utc'], result['status'], 'drivers', result['driver_pids'], flush=True)
    print('Completed:', result['completed_sample_results'], '/', result['total_sample_results'])
    print(f"{'PDE / task / sensor':42s} {'id':>7s} {'smooth':>7s} {'rough':>7s}")
    for pde, task, sensor in ([(p, t, 'random') for p in ['poisson', 'helmholtz', 'darcy', 'nsnonbounded']
                             for t in ['forward', 'inverse', 'both']] +
                             [('burger', 'both', s) for s in ['random', 'sensor_column']]):
        values = [result['completed_by_cell'].get(f'{d}/{pde}/{task}/hybrid_s2d/{sensor}', 0)
                  for d in ['id', 'smooth', 'rough']]
        print(f'{pde + " / " + task + " / " + sensor:42s}' + ''.join(f'{v:8d}' for v in values))
    for row in result['active']:
        print(row['distribution'], row['experiment'], 'offset', row['offset'], 'step', row['step'], '/', row['num_steps'])
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    while True:
        result = snapshot(root)
        if args.watch and sys.stdout.isatty():
            print('\033[2J\033[H', end='')
        if args.json:
            print(json.dumps(result, indent=2), flush=True)
        else:
            table(result)
        if not args.watch or result['status'] == 'complete':
            break
        time.sleep(30)


if __name__ == '__main__':
    main()
