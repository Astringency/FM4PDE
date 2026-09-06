"""Copy new sampling receipts/diagnostics without deleting source or target files.

Collection may be repeated while a job runs. Only complete receipts are used by
the separate strict validator; an interrupted copy is not evaluation evidence.
"""
from pathlib import Path
import argparse
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remote',default='server193')
    parser.add_argument('--remote-root',default='/home/zhangxf/C01Python/FM4PDE_jmlr_20260906/revision_results')
    parser.add_argument('--dest',type=Path,required=True)
    parser.add_argument('--pdes',nargs='+',choices=['poisson','darcy','nsnonbounded','burger'],default=['poisson','darcy','nsnonbounded','burger'])
    parser.add_argument('--with-tensors',action='store_true')
    args=parser.parse_args()
    names=['receipt.json','environment.json','*_complete.json','curves.csv',
           'metrics_step_per_sample.csv','metrics_per_sample.csv','resolved_config.yaml','worker.json']
    if args.with_tensors:names+=['result.pt','masks.pt']
    for pde in args.pdes:
        destination=args.dest/pde;destination.mkdir(parents=True,exist_ok=True)
        cmd=['rsync','-a','--include=*/']+['--include='+name for name in names]+['--exclude=*',
             f'{args.remote}:{args.remote_root}/{pde}/',str(destination)+'/']
        subprocess.run(cmd,check=True)
        print('COPIED',pde,destination,flush=True)


if __name__=='__main__':main()
