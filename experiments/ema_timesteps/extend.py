"""Continue a planned arm after screening, preserving its complete saved state."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from experiments.ema_timesteps.audit import checkpoint
from experiments.optimizer_diagnostics.study import write, sha

ARMS = ['uniform', 'stratified_uniform', 'logit_normal', 'beta1_05']


def run(root, pde, arm, gpu, epochs, checkout, min_free_mib, resume_epoch=2):
    out = root/'extensions'/pde/arm
    out.mkdir(parents=True, exist_ok=True)
    plan = json.loads((root/'extension_plan.json').read_text())
    decision = plan['pdes'][pde][arm]
    assert decision['target_epochs'] == epochs and epochs > 2
    assert decision['role'] in ['control', 'candidate']
    assert 2 <= resume_epoch < epochs
    overlap = decision.get('overlap_screening', False)
    if overlap:
        # A prespecified control can use an otherwise idle GPU. Adaptive recipe
        # extensions still wait for every screening arm and its state audit.
        assert arm == 'uniform' and decision['role'] == 'control'
    torch.set_num_threads(4)

    def status(state, **kwargs):
        write(out/'progress.json', dict(state=state, pde=pde, arm=arm,
              pid=os.getpid(), gpu=gpu, epochs=epochs, **kwargs))

    locks = root/'runtime_locks'
    locks.mkdir(exist_ok=True)
    with (locks/f'{pde}_{arm}.lock').open('a+') as arm_lock, (locks/f'gpu_{gpu}.lock').open('a+') as gpu_lock:
        fcntl.flock(arm_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                status('waiting_other_continuation_on_gpu')
                time.sleep(30)
        while True:
            queue_path = root/'queues'/pde/'progress.json'
            queue = json.loads(queue_path.read_text()) if queue_path.exists() else {}
            for candidate in ARMS:
                exit_path=root/'queues'/pde/f'{candidate}_02.exit.json'
                if exit_path.exists() and json.loads(exit_path.read_text())['exit_code'] != 0:
                    raise RuntimeError(f'Screening failed: {pde}/{candidate}; inspect its log before continuation')
            if overlap:
                completed = root/'queues'/pde/f'{arm}_02.exit.json'
                if (completed.exists() and queue.get('arm') != arm
                        and (root/'runs'/pde/arm/'complete_02.json').exists()):
                    break
                status('waiting_control_screening', stage1_queue=queue)
                time.sleep(30)
                continue
            session = subprocess.run(['tmux','has-session','-t',f'fm_ema_train_{pde}'],capture_output=True)
            ready = all((root/'runs'/pde/a/'complete_02.json').exists() for a in ARMS)
            if queue.get('state') == 'complete' and queue.get('epochs') == 2 and ready and session.returncode != 0:
                break
            status('waiting_stage1_training', stage1_queue=queue)
            time.sleep(30)
        # Check the actual saved states and logs before the first longer update.
        assert json.loads((root/'extension_plan.json').read_text())['pdes'][pde][arm] == decision
        audits = {a:checkpoint(root,pde,a,2) for a in ([arm] if overlap else ARMS)}
        write(out/'screen_audits.json', audits)
        resume_audit = (audits[arm] if resume_epoch == 2 else
                        checkpoint(root, pde, arm, resume_epoch))
        resume_path = root/'runs'/pde/arm/'last.pth'
        assert sha(resume_path) == resume_audit['checkpoint_sha256'], 'Resume state differs from reviewed snapshot'
        write(out/'resume_audit.json', resume_audit)
        while True:
            free = subprocess.check_output(['nvidia-smi','--query-gpu=memory.free',
                    '--format=csv,noheader,nounits'],text=True).splitlines()
            if int(free[gpu]) >= min_free_mib:
                break
            status('waiting_gpu_memory', free_mib=int(free[gpu]), minimum_mib=min_free_mib)
            time.sleep(30)
        command = [sys.executable,'-u','-m','experiments.ema_timesteps.study','run',
                   '--root',str(root),'--pde',pde,'--arm',arm,'--epochs',str(epochs)]
        record = dict(decision=decision, plan_sha256=sha(root/'extension_plan.json'),
            checkout=str(checkout), command=command, observed_free_mib=int(free[gpu]),
            git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=checkout,text=True).strip(),
            original_stage1_queue=queue, resume_snapshot_sha256=resume_audit['checkpoint_sha256'],
            resume_path=str(resume_path), source_epoch=resume_epoch, target_epoch=epochs, overlaps_remaining_screening=overlap,
            audited_screening_arms=list(audits),
            all_arm_audits_required_before_recipe_selection=True)
        write(out/'launch.json',record)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4')
        with (out/'train.log').open('a') as log:
            child=subprocess.Popen(command,cwd=checkout,env=env,stdout=log,stderr=subprocess.STDOUT,
                                   pass_fds=(arm_lock.fileno(),gpu_lock.fileno()))
            status('training',child_pid=child.pid)
            code=child.wait()
        write(out/'exit.json',dict(exit_code=code,child_pid=child.pid))
        if code:
            raise RuntimeError(f'Continuation failed for {pde}/{arm}: {code}')
        result=checkpoint(root,pde,arm,epochs)
        status('complete',audit=result)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--pde',required=True)
    parser.add_argument('--arm',choices=ARMS,default='uniform')
    parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--checkout',type=Path,required=True)
    parser.add_argument('--min-free-mib',type=int,default=70000)
    parser.add_argument('--resume-epoch',type=int,default=2,
                        help='Reviewed complete snapshot to resume; last.pth must have the same SHA')
    args=parser.parse_args()
    run(args.root,args.pde,args.arm,args.gpu,args.epochs,args.checkout,args.min_free_mib,args.resume_epoch)
