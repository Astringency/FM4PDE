"""Let an active sampler finish before replacing only its waiting queue manager.

Linux SIGSTOP applies to the explicitly identified manager PID, never to its
sampler child or process group. The child runs to a normal exit. Complete saved
outputs are checked before the stopped manager is terminated and a smaller
static continuation starts. Existing predictions are not modified.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def process(pid):
    folder = Path('/proc') / str(pid)
    fields = (folder/'stat').read_text().rsplit(')', 1)[1].split()
    return {'pid': pid, 'state': fields[0], 'ppid': int(fields[1]),
            'start_ticks': int(fields[19]), 'exit_code': int(fields[49]),
            'argv': (folder/'cmdline').read_bytes().decode().split('\0')[:-1]}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temp = path.with_suffix('.partial.json')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def suspend_until_child_exits(manager_pid, child_pid, manager_start, child_start):
    parent = process(manager_pid)
    child = process(child_pid)
    assert parent['start_ticks'] == manager_start
    assert child['start_ticks'] == child_start and child['ppid'] == manager_pid
    os.kill(manager_pid, signal.SIGSTOP)
    for _ in range(100):
        stopped = process(manager_pid)
        assert stopped['start_ticks'] == manager_start
        if stopped['state'] == 'T':
            break
        time.sleep(.01)
    else:
        raise RuntimeError('Queue manager did not stop')
    while True:
        parent = process(manager_pid)
        child = process(child_pid)
        assert parent['start_ticks'] == manager_start and parent['state'] == 'T'
        assert child['start_ticks'] == child_start and child['ppid'] == manager_pid
        if child['state'] == 'Z':
            return child['exit_code']
        time.sleep(1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manager-pid', type=int, required=True)
    p.add_argument('--child-pid', type=int, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--code', type=Path, required=True)
    p.add_argument('--diffusion-root', type=Path, required=True)
    p.add_argument('--current-pde', required=True)
    p.add_argument('--next-pdes', nargs='+', required=True)
    a = p.parse_args()
    root, code = a.root.resolve(), a.code.resolve()
    manager, child = process(a.manager_pid), process(a.child_pid)
    assert str(code/'run_all.py') in manager['argv']
    assert str(code/'collect_trace.py') in child['argv']
    for item in [manager, child]:
        assert item['argv'][item['argv'].index('--root')+1] == str(root)
    assert child['argv'][child['argv'].index('--pde')+1] == a.current_pde
    assert child['argv'][child['argv'].index('--mode')+1] == 'run'
    assert child['argv'][child['argv'].index('--steps')+1] == '1000'
    assert child['ppid'] == a.manager_pid
    out = root/'queue_control'
    out.mkdir(exist_ok=True)
    record = out/f'after_{a.current_pde}.json'
    assert not record.exists(), record
    data = {'status': 'waiting_for_active_sampler', 'manager': manager, 'child': child,
            'current_pde': a.current_pde, 'next_pdes': a.next_pdes,
            'script_sha256': sha(__file__), 'started_unix': time.time()}
    write(record, data)
    try:
        exit_code = suspend_until_child_exits(a.manager_pid, a.child_pid,
                                              manager['start_ticks'], child['start_ticks'])
        assert exit_code == 0, ('Sampler did not exit normally', exit_code)
        manifest = json.loads((root/'manifest.json').read_text())
        ids = manifest['cells'][a.current_pde]['evaluation_ids']
        receipts = []
        for method in ['FM4PDE', 'DiffusionPDE']:
            for sid in ids:
                stem = root/'traces'/a.current_pde/f'{method}_1000_{sid}'
                receipt = stem.with_suffix('.json')
                r = json.loads(receipt.read_text())
                assert r['status'] == 'complete' and r['sample_id'] == sid
                assert sha(stem.with_suffix('.csv')) == r['trace_sha256']
                assert sha(stem.with_suffix('.pt')) == r['prediction_sha256']
                receipts.append({'path': str(receipt), 'sha256': sha(receipt)})
        assert len(receipts) == 40
        data.update(status='active_pde_complete', sampler_exit_code=exit_code,
                    completed_receipts=receipts, sampler_exit_observed_unix=time.time())
        write(record, data)
        # The sampler is already a successfully exited zombie. Only its stopped
        # Python manager receives these signals; no GPU process is terminated.
        os.kill(a.manager_pid, signal.SIGTERM)
        os.kill(a.manager_pid, signal.SIGCONT)
        for _ in range(100):
            try:
                state = process(a.manager_pid)
            except FileNotFoundError:
                break
            if state['state'] == 'Z':
                break
            time.sleep(.1)
        else:
            raise RuntimeError('Stopped queue manager did not terminate')
        command = [sys.executable, '-u', str(code/'run_all.py'), '--root', str(root),
                   '--diffusion-root', str(a.diffusion_root.resolve()), '--pdes', *a.next_pdes]
        data.update(status='running_replacement_queue', replacement_argv=command,
                    replacement_started_unix=time.time())
        write(record, data)
        result = subprocess.run(command)
        data.update(status='complete' if result.returncode == 0 else 'error',
                    replacement_exit_code=result.returncode, finished_unix=time.time())
        write(record, data)
        if result.returncode:
            raise SystemExit(result.returncode)
    except BaseException as exc:
        data.update(status='error', error=repr(exc), failed_unix=time.time())
        write(record, data)
        raise


if __name__ == '__main__':
    main()
