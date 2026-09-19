"""Evaluate the paired 512-update checkpoints after initial sampling finishes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from experiments.optimizer_diagnostics.study import write,sha


def main(root):
    out=root/'hard_sampling'
    while True:
        p=out/'queue.json'
        if p.exists() and json.loads(p.read_text()).get('state')=='complete':
            for name in ['original','lr_control_128','beta2_128']:
                assert json.loads((out/f'{name}.exit.json').read_text())['exit_code']==0
            break
        print('WAIT_INITIAL_SAMPLING',flush=True);time.sleep(30)
    for variant,arm in [('beta2_512','selected_resume'),('lr_control_512','lr_control')]:
        checkpoint=root/'continuation/nsnonbounded'/arm/'step_0512.pth'
        receipt=checkpoint.with_suffix('.json')
        while not receipt.exists():
            print('WAIT_SNAPSHOT',arm,flush=True);time.sleep(30)
        assert sha(checkpoint)==json.loads(receipt.read_text())['sha256']
        cmd=[sys.executable,'-u','-m','experiments.optimizer_diagnostics.hard_sampling','run',
            '--root',str(root),'--variant',variant,'--checkpoint',str(checkpoint)]
        with (out/f'{variant}.log').open('a') as stream:
            child=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT)
            write(out/'continuation_queue.json',dict(state='running',variant=variant,child_pid=child.pid,time=time.time()))
            code=child.wait()
        write(out/f'{variant}.exit.json',dict(exit_code=code,child_pid=child.pid,time=time.time()))
        if code: raise RuntimeError(f'Sampling failed for {variant}: {code}')
        subprocess.run([sys.executable,'-u','-m','experiments.optimizer_diagnostics.hard_sampling','report',
            '--root',str(root)],check=True)
    write(out/'continuation_queue.json',dict(state='complete',time=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    main(p.parse_args().root)
