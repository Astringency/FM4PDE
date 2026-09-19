"""One-time, reviewed device handoff for the 20260919 Burgers continuation."""
from pathlib import Path
import json
import os
import shlex
import signal
import subprocess
import time

import torch

from experiments.ema_timesteps.audit import checkpoint
from experiments.optimizer_diagnostics.study import sha, write


ROOT = Path('/data1/zjinzxf2025/C01Python/FM4PDE/reproducibility/ema_timesteps_20260919')
FOLDER = ROOT / 'runs/burger/stratified_uniform'
SCHEDULE = ROOT / 'scheduling/burger_stratified_gpu6'
OUT = SCHEDULE / 'handoff'
PYTHON = '/data1/zjinzxf2025/miniconda3/envs/fm4pde/bin/python'
OLD_PARENT, OLD_CHILD = 3615245, 3627360


def process(pid, module, cwd):
    path = Path('/proc') / str(pid)
    command = (path / 'cmdline').read_bytes().decode().replace('\0', ' ')
    assert module in command and '--pde burger --arm stratified_uniform' in command
    assert Path(os.readlink(path / 'cwd')) == cwd
    return dict(pid=pid, command=command, cwd=str(cwd), stat=(path / 'stat').read_text())


def run():
    torch.set_num_threads(4)
    assert not OUT.exists(), 'This one-time handoff already has a record'
    gate = json.loads((SCHEDULE / 'gate/result.json').read_text())
    assert gate['passed'] and all(gate['checks'].values())
    assert gate['actual']['step'] == 2110
    assert json.loads((SCHEDULE / 'gate_exit.json').read_text())['exit_code'] == 0
    while True:
        process(OLD_CHILD, 'experiments.ema_timesteps.study run', ROOT / 'code_control')
        progress = json.loads((FOLDER / 'progress.json').read_text())
        # A first-step diagnostic remains visible until the next 64-step log.
        # If the epoch-4 window is missed, wait for epoch 5 rather than interrupt
        # a nearly complete epoch. No process is stopped while waiting.
        epoch = progress.get('epoch', 0) - 1
        if epoch in (4, 5) and progress.get('state') == 'training' and progress['step'] == epoch * 703 + 1:
            free = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free',
                                           '--format=csv,noheader,nounits'], text=True).splitlines()
            if int(free[6]) >= 65000:
                break
        if epoch > 5:
            raise RuntimeError('Reviewed handoff window passed; original training remains running')
        time.sleep(1)

    OUT.mkdir()
    parent = process(OLD_PARENT, 'experiments.ema_timesteps.extend', ROOT / 'code_scheduling')
    child = process(OLD_CHILD, 'experiments.ema_timesteps.study run', ROOT / 'code_control')
    saved = torch.load(FOLDER / 'last.pth', map_location='cpu', mmap=True, weights_only=False)
    assert saved['ema_timesteps']['completed_epochs'] == epoch
    assert saved['ema_timesteps']['additional_updates'] == epoch * 703
    assert int(saved['model_for_resume']['num_updates']) == epoch * 703
    del saved
    snapshot = FOLDER / f'epoch_{epoch:04d}.pth'
    if not snapshot.exists():
        snapshot.hardlink_to(FOLDER / 'last.pth')
    digest = sha(snapshot)
    assert sha(FOLDER / 'last.pth') == digest
    ready = snapshot.with_suffix('.ready.json')
    if ready.exists():
        assert json.loads(ready.read_text())['sha256'] == digest
    else:
        write(ready, dict(checkpoint=str(snapshot), sha256=digest, pde='burger',
                         arm='stratified_uniform', epoch=epoch, steps=epoch * 703))
    audited = checkpoint(ROOT, 'burger', 'stratified_uniform', epoch)
    assert audited['checkpoint_sha256'] == digest
    original = json.loads((FOLDER / 'progress.json').read_text())
    assert original['step'] == epoch * 703 + 1, 'First-step window expired before stopping'
    old_progress_mtime = (FOLDER / 'progress.json').stat().st_mtime_ns
    log_sha = sha(FOLDER / 'training.jsonl')
    extension = ROOT / 'extensions/burger/stratified_uniform'
    for name in ['launch.json', 'screen_audits.json', 'progress.json']:
        (OUT / ('previous_' + name)).write_bytes((extension / name).read_bytes())
    write(OUT / 'original_first_update.json', original)
    record = dict(
        old_parent=parent, old_child=child, resume_epoch=epoch, checkpoint_sha256=digest,
        gate_result_sha256=sha(SCHEDULE / 'gate/result.json'), old_gpu=4, new_gpu=6,
        observed_free_mib=int(free[6]), training_log_sha256_before=log_sha,
        reason='Shared GPU4 update median 2.748s versus dedicated GPU6 1.283s; recorded update replay passed exactly',
        partial_epoch_policy='Only complete checkpoint and log prefix retained. Observed first update of the next epoch will be replayed; additional in-memory partial updates are discarded.',
        scientific_checkout=str(ROOT / 'code_control'), scientific_parameters_unchanged=True,
    )
    write(OUT / 'before_stop.json', record)
    # Kill only the verified task parent and child. Child-held locks prevent a
    # second worker until it has exited. The old shell records intentional 143.
    process(OLD_PARENT, 'experiments.ema_timesteps.extend', ROOT / 'code_scheduling')
    process(OLD_CHILD, 'experiments.ema_timesteps.study run', ROOT / 'code_control')
    os.kill(OLD_PARENT, signal.SIGTERM)
    os.kill(OLD_CHILD, signal.SIGTERM)
    for _ in range(100):
        if not (Path('/proc') / str(OLD_CHILD)).exists():
            break
        time.sleep(.1)
    assert not (Path('/proc') / str(OLD_CHILD)).exists()
    assert sha(FOLDER / 'last.pth') == digest
    assert sha(FOLDER / 'training.jsonl') == log_sha
    record['old_worker_exited_and_complete_state_retained'] = True
    write(OUT / 'stopped.json', record)

    command = [PYTHON, '-u', '-m', 'experiments.ema_timesteps.extend', '--root', str(ROOT),
               '--pde', 'burger', '--arm', 'stratified_uniform', '--gpu', '6', '--epochs', '10',
               '--checkout', str(ROOT / 'code_control'), '--min-free-mib', '65000', '--resume-epoch', str(epoch)]
    script = OUT / 'continue.sh'
    script.write_text('cd ' + shlex.quote(str(ROOT / 'code_handoff_e43bb3e')) + '\n' +
                      shlex.join(command) + ' > ' + shlex.quote(str(OUT / 'continue.log')) + ' 2>&1\n' +
                      'code=$?\nprintf \'%s\\n\' "$code" > ' + shlex.quote(str(OUT / 'continue.exit')) + '\nexit "$code"\n')
    subprocess.run(['tmux', 'new-session', '-d', '-s', 'fm_ema_extend_burger_stratified_gpu6',
                    shlex.join(['bash', str(script)])], check=True)
    write(OUT / 'new_launch.json', dict(command=command, controller_checkout=str(ROOT / 'code_handoff_e43bb3e'),
                                      resume_epoch=epoch, checkpoint_sha256=digest))
    for _ in range(600):
        current = json.loads((extension / 'progress.json').read_text())
        if current.get('state') == 'training' and current.get('gpu') == 6:
            new_child = process(current['child_pid'], 'experiments.ema_timesteps.study run', ROOT / 'code_control')
            if (FOLDER / 'progress.json').stat().st_mtime_ns != old_progress_mtime:
                observed = json.loads((FOLDER / 'progress.json').read_text())
                if observed.get('state') == 'training':
                    break
        time.sleep(1)
    else:
        raise RuntimeError('New worker first-update observation timed out; inspect it without restarting')
    assert observed['step'] == original['step']
    keys = ['loss', 'lr', 'grad_norm', 'first_parameter_relative_update', 'input_ids_sha256', 'target_sha256', 'ema_updates']
    checks = {key: observed[key] == original[key] for key in keys}
    write(OUT / 'first_resumed_update.json', dict(actual=observed, reference=original, checks=checks,
          passed=all(checks.values()), process=new_child,
          scope='One actual update scalar diagnostics and input/noise identities; not all tensor or future-update equality'))
    assert all(checks.values()), checks
    print(json.dumps(dict(status='resumed_and_first_update_matched', epoch=epoch, child_pid=current['child_pid'],
                          checkpoint_sha256=digest)), flush=True)


if __name__ == '__main__':
    run()
