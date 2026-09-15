#!/usr/bin/env python3
"""Relay completed fixed-observation pools through the local host, then audit and plot.

No sampler is restarted or interrupted. Immutable predictions use ignore-existing;
conflicting bytes fail the final strict audit. Network failures are retried.
"""
from __future__ import annotations
import argparse
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'plot'))
from run_paper_ablation_revision import digest, write

REMOTE_SNAPSHOT = r'''
import json,os
from pathlib import Path
root=Path(ROOT_VALUE)
expected={(t,i) for t in ['forward','inverse','both'] for i in range(1500,1532)}
completed=[];files=[];mutable=[];workers=[]
for path in sorted(root.glob('cases/*/offset*/complete.json')):
 x=json.loads(path.read_text());b=x['binding'];key=(b['task'],b['offset'])
 assert key in expected and x['status']=='complete' and x['predictions']==1000
 completed.append(key)
 for f in [path,path.parent/'pool.pt',*[path.parent/f'prefix_{k}.json' for k in [1,3,10,100,1000]]]:
  assert f.is_file();files.append(str(f.relative_to(root)))
 for f in path.parent.glob('batches/*/receipt.json'):files.append(str(f.relative_to(root)))
assert len(completed)==len(set(completed))
for dirname in ['inputs','pilot','actual_stock_check','invocations','provenance']:
 for f in (root/dirname).rglob('*'):
  if f.is_file() and not any(p.startswith('.partial_') for p in f.parts):files.append(str(f.relative_to(root)))
for dirname in ['logs','workers']:
 for f in (root/dirname).rglob('*'):
  if f.is_file() and not f.name.endswith('.tmp'):mutable.append(str(f.relative_to(root)))
for worker in WORKER_VALUES:
 p=root/'workers'/f'{worker}.json'
 if not p.exists():workers.append({'worker':worker,'status':'missing','live':False});continue
 x=json.loads(p.read_text());pid=x.get('pid');proc=Path('/proc')/str(pid)/'cmdline'
 try:command=proc.read_bytes().decode().replace(chr(0),' ')
 except FileNotFoundError:command=''
 x['live']='run_conditional_sample_scaling.py run' in command and str(root) in command
 workers.append(x)
print(json.dumps({'completed':completed,'immutable_files':sorted(set(files)),
 'mutable_files':sorted(set(mutable)),'workers':workers}))
'''


class Collector:
    def __init__(self, args):
        self.args = args
        self.control = args.local_root / 'collector'
        self.control.mkdir(parents=True, exist_ok=True)
        self.log = self.control / 'commands.log'

    def command(self, argv, *, timeout=1800, capture=False):
        with self.log.open('a') as log:
            log.write(json.dumps(dict(time=time.time(), argv=argv)) + '\n');log.flush()
            result = subprocess.run(argv, text=True, stdout=subprocess.PIPE if capture else log,
                                    stderr=log, timeout=timeout)
            log.write(f'EXIT {result.returncode} {time.time()}\n')
        if result.returncode:
            raise subprocess.CalledProcessError(result.returncode, argv)
        return result.stdout if capture else None

    def ssh(self, host, python, script):
        return self.command(['ssh', '-o', 'ConnectTimeout=20', host, shlex.join([python, '-c', script])],
                            timeout=90, capture=True)

    def synchronize(self, names, kind):
        if not names:
            return
        a = self.args
        listing = self.control / f'{kind}_files.txt'
        listing.write_text(''.join(name + '\n' for name in sorted(set(names))))
        options = ['rsync', '-a', '--stats', '--timeout=45', '--partial-dir=.partial_rsync',
                   '--files-from=' + str(listing)]
        if kind == 'immutable':
            options.append('--ignore-existing')
        self.command(options + [a.remote_host + ':' + a.remote_root + '/', str(a.local_root) + '/'])
        self.command(options + [str(a.local_root) + '/', a.canonical_host + ':' + a.canonical_root + '/'])

    def snapshot(self):
        a = self.args
        script = REMOTE_SNAPSHOT.replace('ROOT_VALUE', repr(a.remote_root)).replace('WORKER_VALUES', repr(a.workers))
        return json.loads(self.ssh(a.remote_host, a.remote_python, script))

    def export_status(self):
        a = self.args
        script = 'import json,subprocess;from pathlib import Path;p=Path(' + repr(a.canonical_root) + ');'
        script += 'f=p/"export/conditional_scaling_manifest.json";e=p/"logs/final_export.exit";'
        script += 'live=subprocess.run(["tmux","has-session","-t","conditional_fixed_0915_export"],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0;'
        script += 'print(json.dumps({"manifest":json.loads(f.read_text()) if f.exists() else None,"exit":e.read_text().strip() if e.exists() else None,"live":live}))'
        return json.loads(self.ssh(a.canonical_host, a.canonical_python, script))

    def start_export(self):
        a = self.args
        argv = ['env', 'PYTHONDONTWRITEBYTECODE=1', 'OMP_NUM_THREADS=2', 'OPENBLAS_NUM_THREADS=2',
                a.canonical_python, a.canonical_code + '/plot/export_conditional_sample_scaling.py',
                '--results', a.canonical_root, '--inputs', a.canonical_root + '/inputs/poisson',
                '--selection', a.canonical_root + '/provenance/selection.json',
                '--output', a.canonical_root + '/export']
        shell = shlex.join(argv) + ' > ' + shlex.quote(a.canonical_root + '/logs/final_export.log')
        shell += ' 2>&1; stage_exit=$?; printf "%s\\n" "$stage_exit" > ' + shlex.quote(a.canonical_root + '/logs/final_export.exit')
        shell += '; exit "$stage_exit"'
        self.command(['ssh', a.canonical_host, shlex.join(['tmux', 'new-session', '-d', '-s',
                     'conditional_fixed_0915_export', 'bash', '-lc', shell])], timeout=60)
        write(self.control / 'export_started.json', dict(argv=argv, started_unix=time.time()))

    def finish(self, remote_manifest):
        a = self.args
        assert remote_manifest['status'] == 'pass' and remote_manifest['complete']
        assert remote_manifest['conditional_trajectories'] == 96000
        assert len(remote_manifest['completed_jobs']) == 96
        export = a.local_root / 'export'; export.mkdir(exist_ok=True)
        self.command(['rsync', '-a', '--ignore-existing', '--stats', '--timeout=45',
                      a.canonical_host + ':' + a.canonical_root + '/export/', str(export) + '/'])
        local_manifest = json.loads((export / 'conditional_scaling_manifest.json').read_text())
        assert local_manifest == remote_manifest
        for name, expected_hash in remote_manifest['export_files'].items():
            assert digest(export / name) == expected_hash
        self.command([sys.executable, str(ROOT / 'plot/plot_conditional_sample_scaling.py'),
                      '--exports', str(export), '--output', str(a.paper / 'source_data/conditional_fixed_0915'),
                      '--figures', str(a.paper / 'figures'), '--figure-prefix', 'revision_0915_conditional_fixed'])
        source = a.paper / 'source_data/conditional_fixed_0915'
        final = json.loads((source / 'conditional_scaling_final_manifest.json').read_text())
        assert final['complete'] and final['rows'] == 480 and final['canonical_trajectories'] == 96000
        for fig in final['figure_files']:
            assert digest(fig['path']) == fig['sha256']
        self.command(['rsync', '-a', '--ignore-existing', str(source) + '/',
                      a.canonical_host + ':' + a.canonical_root + '/paper_export/source_data/'])
        for fig in final['figure_files']:
            self.command(['rsync', '-a', '--ignore-existing', fig['path'],
                          a.canonical_host + ':' + a.canonical_root + '/paper_export/figures/'])
        write(a.local_root / 'completion.json', dict(status='audited_and_plotted', unique_cases=96,
            predictions=96000, prefix_records=480, completed_unix=time.time(),
            paper_manifest=str(source / 'conditional_scaling_final_manifest.json'),
            paper_manifest_sha256=digest(source / 'conditional_scaling_final_manifest.json'),
            remaining='Manuscript integration and final paper review by the parent task.'))
        return True

    def cycle(self):
        a = self.args
        snapshot = self.snapshot()
        state = dict(observed_unix=time.time(), completed_cases=len(snapshot['completed']),
                     completed_draws=1000 * len(snapshot['completed']), workers=snapshot['workers'], status='collecting')
        write(self.control / 'status.json', state)
        self.synchronize(snapshot['immutable_files'], 'immutable')
        self.synchronize(snapshot['mutable_files'], 'mutable')
        if len(snapshot['completed']) < 96:
            if not any(worker['live'] for worker in snapshot['workers']):
                raise RuntimeError('All samplers are terminal but fewer than 96 complete cases are available')
            state['status'] = 'running'; write(self.control / 'status.json', state)
            return False
        status = self.export_status()
        if status['exit'] not in [None, '0']:
            raise RuntimeError('Strict final export failed: exit ' + status['exit'])
        if status['manifest'] is not None:
            if status['exit'] != '0':
                state['status'] = 'auditing'; write(self.control / 'status.json', state)
                return False
            result = self.finish(status['manifest'])
            state['status'] = 'audited_and_plotted'; write(self.control / 'status.json', state)
            return result
        if not (self.control / 'export_started.json').exists():
            self.start_export()
        elif not status['live']:
            raise RuntimeError('Final audit process is missing without a successful exit receipt')
        state['status'] = 'auditing'; write(self.control / 'status.json', state)
        return False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--local-root', type=Path, required=True)
    p.add_argument('--paper', type=Path, required=True)
    p.add_argument('--remote-host', default='server216')
    p.add_argument('--remote-root', required=True)
    p.add_argument('--remote-python', default='/data1/zjinzxf2025/miniconda3/envs/fm4pde/bin/python')
    p.add_argument('--canonical-host', default='server197')
    p.add_argument('--canonical-root', required=True)
    p.add_argument('--canonical-code', required=True)
    p.add_argument('--canonical-python', default='/research_data/users/zhangxifeng/.conda/envs/fm4pde/bin/python')
    p.add_argument('--workers', nargs='+', default=['g0', 'g2', 'g3', 'g4'])
    p.add_argument('--once', action='store_true')
    p.add_argument('--interval', type=int, default=60)
    args = p.parse_args()
    assert args.local_root.is_absolute() and args.interval >= 10
    collector = Collector(args)
    if (args.local_root / 'completion.json').exists():
        previous = json.loads((args.local_root / 'completion.json').read_text())
        assert previous['status'] == 'audited_and_plotted' and previous['predictions'] == 96000
        print('ALREADY_AUDITED_AND_PLOTTED', flush=True)
        return
    with (collector.control / 'collector.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                done = collector.cycle()
                print('COLLECTED', time.time(), 'complete' if done else 'running', flush=True)
                if done or args.once:
                    return
            except (subprocess.SubprocessError, OSError) as exc:
                write(collector.control / 'network_retry.json', dict(status='retry', error=repr(exc), time=time.time()))
                print('RETRY', repr(exc), flush=True)
                if args.once:
                    raise
            except BaseException as exc:
                write(collector.control / 'status.json', dict(status='blocked', error=repr(exc), time=time.time()))
                raise
            time.sleep(args.interval)


if __name__ == '__main__':
    main()
