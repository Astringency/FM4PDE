#!/usr/bin/env python3
"""Wait for the archive's verified entry, then launch one CPU-only NS audit."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time


def save(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def ssh(command, **kwargs):
    return subprocess.run(['ssh', 'server197', command], text=True, check=True, **kwargs)


def main(args):
    audit = args.plan.parent
    plan = json.loads(args.plan.read_text())
    marker = audit / 'launch_attempt.json'
    assert not marker.exists(), 'An earlier launch attempt must be inspected, never silently restarted'
    while True:
        state = json.loads(args.archive_status.read_text())
        record = state.get('entries', {}).get('ns_main_complete_local_ns_study', {})
        if record.get('status') == 'held':
            raise RuntimeError('Required archive entry is held: ' + json.dumps(record))
        if record.get('status') == 'verified':
            break
        print('WAITING_FOR_VERIFIED_NS_ARCHIVE', state.get('current_entry'), flush=True)
        time.sleep(30)
    expected = record['receipt_sha256']
    assert expected == record['execute']['receipt_sha256'] == record['verify']['receipt_sha256']
    assert record['verify']['destination'] == plan['remote_study']
    save(audit/'archive_dependency_receipt.json', record)
    resources = ssh('hostname; uptime; free -h; getconf _NPROCESSORS_ONLN; '
        'nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader',
        capture_output=True).stdout
    (audit/'remote_resources_before.txt').write_text(resources)
    command = ['env', *[key+'='+value for key,value in plan['environment'].items()],
        'PYTHONUNBUFFERED=1', 'nice', '-n', '10', plan['python'],
        plan['remote_checkout']+'/reproducibility/revision_20260909/verify_ns_archived_main.py',
        '--study', plan['remote_study'], '--output', plan['remote_output'], '--receipt-sha256', expected]
    log, code = plan['remote_root']+'/run.log', plan['remote_root']+'/exit_code'
    shell = shlex.join(command)+' > '+shlex.quote(log)+' 2>&1\nstatus=$?\nprintf "%s\\n" "$status" > '+shlex.quote(code)+'\nexit "$status"'
    tmux = ['tmux','new-session','-d','-s',plan['tmux_session'],'bash','-lc',shell]
    attempt = dict(status='launching', command=command, tmux=tmux, receipt_sha256=expected,
        code_commit=plan['code_commit'], archive_record=record,
        launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    save(marker, attempt)
    # A transport failure here is deliberately not retried: the remote launch
    # may have succeeded. A human or parent agent must inspect the saved attempt.
    ssh(shlex.join(tmux))
    attempt['status']='running';save(marker, attempt)
    print('CPU_VERIFICATION_STARTED', plan['tmux_session'], flush=True)
    check = 'if test -f '+shlex.quote(code)+'; then cat '+shlex.quote(code)+'; elif '+shlex.join(
        ['tmux','has-session','-t',plan['tmux_session']])+' 2>/dev/null; then echo RUNNING; else echo INDETERMINATE; fi'
    while True:
        try:
            observed = ssh(check, capture_output=True).stdout.strip()
        except subprocess.CalledProcessError:
            print('SSH_OBSERVATION_RETRY', flush=True);time.sleep(30);continue
        if observed == 'RUNNING':
            print('CPU_VERIFICATION_RUNNING', flush=True);time.sleep(30);continue
        if observed == 'INDETERMINATE':
            raise RuntimeError('Remote audit has neither a session nor an exit receipt; inspect without restarting')
        assert observed.lstrip('-').isdigit(), observed
        attempt['returncode']=int(observed)
        attempt['status']='complete' if observed=='0' else 'failed'
        save(marker, attempt)
        subprocess.run(['scp','server197:'+log,str(audit/'remote_run.log')],check=True)
        if observed!='0':
            raise RuntimeError('CPU archive verification failed; see remote_run.log')
        break
    assert not (audit/'recomputed').exists()
    subprocess.run(['scp','-r','server197:'+plan['remote_output'],str(audit/'recomputed')],check=True)
    result=json.loads((audit/'recomputed/verification_complete.json').read_text())
    assert result['status']=='pass' and result['complete'] and result['examples']==15000
    print('ARCHIVED_NS_RECOMPUTATION_PASS', flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--archive-status',type=Path,required=True)
    main(parser.parse_args())
