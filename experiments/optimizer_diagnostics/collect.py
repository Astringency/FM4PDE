"""Deliver a finished study and clean only its verified temporary workspaces."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from experiments.optimizer_diagnostics.relay import CACHE, CANONICAL, ssh

LOCAL=Path('/home/tat512/share/reports/FM4PDE_optimizer_20260919')
WORKTREE='/home/tat512/C01Python/FM4PDE_optimizer_20260919'
PDES=['poisson','helmholtz','darcy','nsnonbounded','burger']


def remote_python(host,source):
    return ssh(host,"python3 - <<'PY'\n"+source+"\nPY",capture_output=True,text=True).stdout


def ready():
    result=remote_python('server197',f'''
from pathlib import Path
import json,subprocess
r=Path({CANONICAL!r})
p=r/'finalization.json'
status=json.loads(p.read_text()) if p.exists() else {{'state':'waiting'}}
active=subprocess.run(['tmux','has-session','-t','fm_opt_0919_finalize'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0
if status.get('state')!='complete' and not active:
    status={{'state':'failed','reason':'Finalizer exited before completion','log_tail':(r/'finalize.log').read_text()[-3000:]}}
if status.get('state')=='complete':
    for pde in {PDES!r}:
        run=r/'runs'/pde
        assert json.loads((run/'audit.json').read_text())['status']=='verified'
        assert json.loads((run/'run.exit.json').read_text())['exit_code']==0
        assert json.loads((run/'inspect.exit.json').read_text())['exit_code']==0
    assert json.loads((r/'report'/'summary.json').read_text())['status']=='complete'
    if active:
        status={{'state':'finalizer_exiting'}}
    else:
        sample=r/'hard_sampling/finalization.json'
        sampling=json.loads(sample.read_text()) if sample.exists() else {{'state':'waiting'}}
        if sampling.get('state')!='complete':
            status={{'state':'sampling_pending','pending':sampling.get('pending',{PDES!r})}}
        else:
            assert sampling['continuation_verified']
            assert json.loads((r/'continuation/nsnonbounded/audit.json').read_text())['status']=='verified'
            for pde in {PDES!r}:
                out=r/'hard_sampling' if pde=='nsnonbounded' else r/'hard_sampling'/pde
                audit=json.loads((out/'audit.json').read_text())
                assert audit['status']=='verified'
                assert len(audit['variants'])==(5 if pde=='nsnonbounded' else 3)
                assert json.loads((out/'figures/manifest.json').read_text())['all_16_cases_shown']
            sessions=subprocess.check_output(['tmux','list-sessions','-F','#{{session_name}}'],text=True).splitlines()
            if any(s.startswith(('fm_optsample','fm_optextend')) for s in sessions):
                status={{'state':'sampling_finalizer_exiting'}}
print(json.dumps(status))
''')
    return json.loads(result)


def finish():
    assert ready().get('state')=='complete'
    # Relay sessions exit only after canonical SHA256 verification and publication.
    for pde in ['helmholtz','darcy']:
        while subprocess.run(['tmux','has-session','-t',f'fm_optrelay_0919_{pde}'],
                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:
            time.sleep(30)
        ssh('server197',f'cd {CANONICAL}/runs/{pde} && sha256sum -c SHA256SUMS > {CANONICAL}/incoming_{pde}/final_verification.log')
        log=Path(f'/tmp/fm_optrelay_0919_{pde}.log')
        if log.exists():
            saved=LOCAL/f'relay_{pde}.log'
            saved.write_bytes(log.read_bytes())
            subprocess.run(['scp',str(saved),f'server197:{CANONICAL}/remote_execution/relay_{pde}.log'],check=True)
            log.unlink()
    for pde in ['helmholtz','darcy']:
        while subprocess.run(['tmux','has-session','-t',f'fm_optsamplerelay_0919_{pde}'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:
            time.sleep(30)
        ssh('server197',f'cd {CANONICAL}/hard_sampling/{pde} && sha256sum -c SHA256SUMS > {CANONICAL}/incoming_sampling/{pde}_final_verification.log')
        log=Path(f'/tmp/fm_optsamplerelay_0919_{pde}.log')
        if log.exists():
            subprocess.run(['scp',str(log),f'server197:{CANONICAL}/remote_execution/sampling_relay_{pde}.log'],check=True)
            log.unlink()
    subprocess.run(['scp','-r',f'server197:{CANONICAL}/report/.',str(LOCAL)],check=True)
    source=subprocess.Popen(['ssh','server197',f'tar -C {CANONICAL} --exclude=*.pth --exclude=full_train.pt -cf - hard_sampling continuation'],stdout=subprocess.PIPE)
    target=subprocess.Popen(['tar','-C',str(LOCAL),'-xf','-'],stdin=source.stdout)
    source.stdout.close()
    assert target.wait()==0 and source.wait()==0
    # Archive the final branch, including every training, sampling and audit revision.
    commit=subprocess.check_output(['git','-C',WORKTREE,'rev-parse','HEAD'],text=True).strip()
    subprocess.run(['git','-C','/home/tat512/C01Python/FM4PDE','merge-base','--is-ancestor',commit,'HEAD'],check=True)
    bundle=Path('/tmp/fm4pde_optimizer_20260919_final_source.bundle')
    subprocess.run(['git','-C',WORKTREE,'bundle','create',str(bundle),'experiments/optimizer-diagnostics-20260919'],check=True)
    subprocess.run(['scp',str(bundle),f'server197:{CANONICAL}/source_next.bundle'],check=True)
    ssh('server197',f'git -C {CANONICAL}/code_final bundle verify {CANONICAL}/source_next.bundle && mv {CANONICAL}/source_next.bundle {CANONICAL}/source.bundle')
    # All training source versions are ancestors in this complete Git bundle.
    remote_python('server197',f'''
from pathlib import Path
import json,subprocess,shutil,time
r=Path({CANONICAL!r})
archive=r/'source.bundle'
assert archive.exists()
verification=r/'source_verify.git'
assert not verification.exists()
subprocess.run(['git','clone','--bare',str(archive),str(verification)],check=True)
subprocess.run(['git','-C',str(verification),'fsck','--full'],check=True)
assert not (verification/'objects'/'info'/'alternates').exists()
subprocess.run(['git','-C',str(verification),'cat-file','-e',{commit!r}+':experiments/optimizer_diagnostics/finalize_sampling.py'],check=True)
for pde in {PDES!r}:
    commit=json.loads((r/'runs'/pde/'protocol.json').read_text())['git_commit']
    subprocess.run(['git','-C',str(verification),'cat-file','-e',commit+':experiments/optimizer_diagnostics/study.py'],check=True)
for folder in ['code','code_v3','code_final']:
    p=r/folder
    if p.exists():
        assert not subprocess.check_output(['git','-C',str(p),'status','--porcelain'],text=True).strip()
        for proc in Path('/proc').glob('[0-9]*'):
            try: cwd=(proc/'cwd').resolve(strict=True)
            except (OSError,PermissionError): continue
            assert not cwd.is_relative_to(p),str(cwd)
shutil.rmtree(verification)
for folder in ['code','code_v3','code_final']:
    p=r/folder
    if p.exists(): shutil.rmtree(p)
for p in r.glob('code*.bundle'): p.unlink()
for p in r.glob('sampling*.bundle'): p.unlink()
(r/'source_archive.json').write_text(json.dumps(dict(bundle=str(archive),independent_restore_verified=True,all_training_commits_present=True,time=time.time()),indent=2))
''')
    remote_python('server216',f'''
from pathlib import Path
import shutil,subprocess
r=Path({CACHE!r})
for pde in ['helmholtz','darcy']:
    assert subprocess.run(['tmux','has-session','-t',f'fm_opt_0919_{{pde}}'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode!=0
    assert subprocess.run(['tmux','has-session','-t',f'fm_optsample_0919_{{pde}}'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode!=0
for proc in Path('/proc').glob('[0-9]*'):
    try: cwd=(proc/'cwd').resolve(strict=True)
    except (OSError,PermissionError): continue
    assert not cwd.is_relative_to(r),str(cwd)
if r.exists(): shutil.rmtree(r)
''')
    subprocess.run(['git','-C','/home/tat512/C01Python/FM4PDE','worktree','remove',WORKTREE],check=True)
    for p in Path('/tmp').glob('fm4pde_optimizer_20260919*.bundle'):
        p.unlink()
    for p in Path('/tmp').glob('fm4pde_optimizer_sampling_20260919*.bundle'):
        p.unlink()
    record=dict(state='complete',canonical_root=CANONICAL,local_report=str(LOCAL/'README.md'),
        temporary_server216_cache_removed=True,temporary_code_checkouts_removed=True,
        source_bundle=f'{CANONICAL}/source.bundle',time=time.time())
    record['sampling_report']=str(LOCAL/'SAMPLING.md')
    record['five_pde_sampling_and_ns_continuation_verified']=True
    (LOCAL/'delivery.json').write_text(json.dumps(record,indent=2)+'\n')
    ssh('server197',f'cat > {CANONICAL}/delivery.json',input=json.dumps(record,indent=2),text=True)
    print('DELIVERED',json.dumps(record),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--once',action='store_true',help='Read readiness once without publishing or cleaning')
    args=parser.parse_args()
    LOCAL.mkdir(parents=True,exist_ok=True)
    previous=None
    while True:
        try:
            status=ready()
        except subprocess.CalledProcessError as exc:
            print('POLL_RETRY',str(exc),flush=True)
            if args.once: raise
            time.sleep(30)
            continue
        print('STATUS',json.dumps(status),flush=True)
        if args.once: return
        if status.get('state')=='failed':
            raise RuntimeError(json.dumps(status))
        marker=(status.get('state'),tuple(status.get('pending',[])))
        if marker!=previous:
            subprocess.run(['scp','-r',f'server197:{CANONICAL}/report/.',str(LOCAL)],check=True)
            previous=marker
        (LOCAL/'delivery.json').write_text(json.dumps(dict(state='waiting',remote_status=status),indent=2)+'\n')
        if status.get('state')=='complete': break
        time.sleep(30)
    finish()


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        LOCAL.mkdir(parents=True,exist_ok=True)
        (LOCAL/'delivery.json').write_text(json.dumps(dict(state='failed',error=repr(exc)),indent=2)+'\n')
        raise
