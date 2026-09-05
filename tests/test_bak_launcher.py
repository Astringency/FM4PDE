from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('fail_worker', [False, True])
def test_two_gpu_launcher_shards_and_stops_on_failure(tmp_path, fail_worker):
    stub=tmp_path/'python_stub'
    stub.write_text(f'#!{sys.executable}\n'+'''
import json,os,pathlib,sys,time
args=sys.argv[1:]
root=pathlib.Path(args[args.index('--output')+1])
if '--worker-index' in args:
    index=args[args.index('--worker-index')+1]
    (root/f'worker{index}.args.json').write_text(json.dumps(args))
    if os.environ.get('TEST_FAIL_WORKER')=='1':
        if index=='1':
            time.sleep(.1)
            sys.exit(7)
        time.sleep(30)
elif '--summarize-only' in args:
    (root/'summary.csv').write_text('summary completed')
''')
    stub.chmod(0o755)
    output=tmp_path/'results'
    env={**os.environ,'BAK_PYTHON':str(stub),'BAK_OUTPUT':str(output),
         'TEST_FAIL_WORKER':'1' if fail_worker else '0'}
    proc=subprocess.run(['bash','scripts/run_bak_comparison_a100.sh'],env=env,
                        capture_output=True,text=True,timeout=10)
    assert proc.returncode==(1 if fail_worker else 0),proc.stderr
    assert (output/'summary.csv').is_file()
    for index in (0,1):
        args=json.loads((output/f'worker{index}.args.json').read_text())
        assert args[args.index('--device')+1]==f'cuda:{index}'
        assert args[args.index('--worker-index')+1]==str(index)
        assert args[args.index('--num-workers')+1]=='2'
