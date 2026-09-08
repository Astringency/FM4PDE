"""Sequential per-GPU queue: diagnostics, frozen choice, rerun, seed ensemble."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
REVISED={'poisson','nsnonbounded','reaction_diffusion','shallow_water','heat',
         'wave','advection_diffusion','steady_heat_conduction'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pdes',nargs='+',required=True)
    p.add_argument('--wait-for-complete',nargs='*',default=[])
    a=p.parse_args()
    deadline=time.monotonic()+7200
    while any(not (a.output/pde/'diagnose_complete.json').exists() for pde in a.wait_for_complete):
        if time.monotonic()>deadline:
            raise TimeoutError('Prerequisite diagnostics have not completed; no GPU work started')
        time.sleep(10)
    for pde in a.pdes:
        shared=['--inputs',str(a.inputs),'--output',str(a.output),'--pdes',pde]
        target=a.output/pde
        if pde in REVISED and not (target/'diagnose_complete.json').exists():
            subprocess.run([sys.executable,'-u',str(HERE/'run_paper_ablation_revision.py'),'diagnose',*shared],check=True)
        subprocess.run([sys.executable,'-u',str(HERE/'select_paper_ablation_revision.py'),*shared],check=True)
        for mode in (['rerun','ensemble'] if pde in REVISED else ['ensemble']):
            if not (target/f'{mode}_complete.json').exists():
                subprocess.run([sys.executable,'-u',str(HERE/'run_paper_ablation_revision.py'),mode,*shared],check=True)
    print('QUEUE COMPLETE',a.pdes,flush=True)


if __name__=='__main__':main()
