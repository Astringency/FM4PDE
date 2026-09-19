"""Deliver a finished study and clean only its verified temporary workspaces."""
import argparse
import hashlib
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
            replica=r/'seed_replicates'
            if replica.exists():
                receipt=replica/'combine.exit.json'
                if not receipt.exists():
                    status={{'state':'additional_seeds_pending','pending':['darcy']}}
                else:
                    assert json.loads(receipt.read_text())['exit_code']==0
                    assert json.loads((r/'hard_sampling/darcy/four_seed_summary.json').read_text())['status']=='verified'
print(json.dumps(status))
''')
    return json.loads(result)


def finish():
    assert ready().get('state')=='complete'
    ssh('server197',f'cd {CANONICAL}/code_final && /research_data/users/zhangxifeng/.conda/envs/fm4pde/bin/python -m experiments.optimizer_diagnostics.conclusion --root {CANONICAL}')
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
        remote_python('server197',f'''
from pathlib import Path
import hashlib,json
r=Path({CANONICAL!r})/'hard_sampling'/{pde!r}
assert json.loads((r/'publication.json').read_text())['all_files_sha256_verified']
derived={{'./summary.json','./README.md','./per_sample.csv'}}
checked=[]
for line in (r/'SHA256SUMS').read_text().splitlines():
    expected,name=line.split('  ',1)
    if name in derived: continue  # Recomputed and independently audited on server197.
    assert hashlib.sha256((r/name).read_bytes()).hexdigest()==expected,name
    checked.append(name)
(Path({CANONICAL!r})/'incoming_sampling'/{(pde+'_final_verification.log')!r}).write_text('\\n'.join(checked)+'\\n')
''')
        log=Path(f'/tmp/fm_optsamplerelay_0919_{pde}.log')
        if log.exists():
            subprocess.run(['scp',str(log),f'server197:{CANONICAL}/remote_execution/sampling_relay_{pde}.log'],check=True)
            log.unlink()
    while subprocess.run(['tmux','has-session','-t','fm_optsamplereprelay_0919_darcy'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:
        time.sleep(30)
    replica_log=Path('/tmp/fm_optsamplereprelay_0919_darcy.log')
    if replica_log.exists():
        subprocess.run(['scp',str(replica_log),f'server197:{CANONICAL}/remote_execution/sampling_replicate_relay_darcy.log'],check=True)
        replica_log.unlink()
    subprocess.run(['scp','-r',f'server197:{CANONICAL}/report/.',str(LOCAL)],check=True)
    source=subprocess.Popen(['ssh','server197',f'tar -C {CANONICAL} --exclude=*.pth --exclude=full_train.pt -cf - hard_sampling continuation seed_replicates/hard_sampling seed_replicates/combine.log seed_replicates/combine.exit.json'],stdout=subprocess.PIPE)
    target=subprocess.Popen(['tar','-C',str(LOCAL),'-xf','-'],stdin=source.stdout)
    source.stdout.close()
    assert target.wait()==0 and source.wait()==0
    manifest=json.loads(remote_python('server197',f'''
from pathlib import Path
import hashlib,json
r=Path({CANONICAL!r}); rows=[]
for folder,prefix in [('report',''),('hard_sampling','hard_sampling'),('continuation','continuation'),('seed_replicates/hard_sampling','seed_replicates/hard_sampling')]:
    base=r/folder
    for p in sorted(base.rglob('*')):
        if not p.is_file() or p.suffix=='.pth' or p.name=='full_train.pt': continue
        h=hashlib.sha256()
        with p.open('rb') as stream:
            for block in iter(lambda:stream.read(8<<20),b''): h.update(block)
        rows.append(dict(path=str(Path(prefix)/p.relative_to(base)),sha256=h.hexdigest()))
print(json.dumps(rows))
'''))
    for entry in manifest:
        h=hashlib.sha256()
        with (LOCAL/entry['path']).open('rb') as stream:
            for block in iter(lambda:stream.read(8<<20),b''): h.update(block)
        assert h.hexdigest()==entry['sha256'],entry['path']
    (LOCAL/'artifact_verification.json').write_text(json.dumps(dict(status='verified',files=manifest),indent=2)+'\n')
    subprocess.run(['scp',str(LOCAL/'artifact_verification.json'),f'server197:{CANONICAL}/local_report_verification.json'],check=True)
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
sampling_commits=set()
for base in [r/'hard_sampling',r/'seed_replicates/hard_sampling']:
    for identity in base.glob('**/identity.json'):
        commit=json.loads(identity.read_text())['environment']['code_commit']
        sampling_commits.add(commit)
        subprocess.run(['git','-C',str(verification),'cat-file','-e',commit+':experiments/optimizer_diagnostics/hard_sampling.py'],check=True)
for p in (r/'continuation').glob('**/protocol.json'):
    commit=json.loads(p.read_text())['git_commit']
    subprocess.run(['git','-C',str(verification),'cat-file','-e',commit+':experiments/optimizer_diagnostics/extend_ns.py'],check=True)
reference=sorted(sampling_commits)[0]
for commit in sampling_commits:
    subprocess.run(['git','-C',str(verification),'diff','--quiet',reference,commit,'--','sampling','models','experiments/aligned_sampling'],check=True)
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
(r/'source_archive.json').write_text(json.dumps(dict(bundle=str(archive),independent_restore_verified=True,all_training_commits_present=True,sampling_commits=sorted(sampling_commits),sampling_core_identical_across_revisions=True,time=time.time()),indent=2))
''')
    remote_python('server216',f'''
from pathlib import Path
import shutil,subprocess
r=Path({CACHE!r})
for pde in ['helmholtz','darcy']:
    assert subprocess.run(['tmux','has-session','-t',f'fm_opt_0919_{{pde}}'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode!=0
    assert subprocess.run(['tmux','has-session','-t',f'fm_optsample_0919_{{pde}}'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode!=0
assert subprocess.run(['tmux','has-session','-t','fm_optsamplerep_0919_darcy'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode!=0
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
