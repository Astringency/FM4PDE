"""Publish completed remote hard-case sampling with per-file SHA256 checks."""
import argparse
import json
import time
from experiments.optimizer_diagnostics.relay import CACHE, CANONICAL, ssh, pipe


def main(pde):
    source=f'{CACHE}/hard_sampling/{pde}'
    while True:
        response=ssh('server216',f"python3 - <<'PY'\nfrom pathlib import Path\nimport json,subprocess\np=Path({source!r})\nf=p/'queue.json'\ns=json.loads(f.read_text()) if f.exists() else {{'state':'waiting'}}\nfor n in ['original','lr_control_128','selected_128']:\n e=p/(n+'.exit.json')\n if e.exists(): assert json.loads(e.read_text())['exit_code']==0\nactive=subprocess.run(['tmux','has-session','-t','fm_optsample_0919_{pde}'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0\nassert active or s.get('state')=='complete',s\nprint(json.dumps(dict(ready=s.get('state')=='complete' and not active,status=s)))\nPY",capture_output=True,text=True)
        status=json.loads(response.stdout)
        print('SAMPLING_REMOTE_STATUS',pde,status,flush=True)
        if status['ready']: break
        time.sleep(30)
    ssh('server216',f'cd {source} && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS')
    stage=f'{CANONICAL}/incoming_sampling/{pde}'
    ssh('server197',f'mkdir -p {stage}')
    pipe('server216',f'tar -C {source} -cf - .','server197',f'tar -C {stage} -xf -')
    ssh('server197',f'cd {stage} && sha256sum -c SHA256SUMS > ../{pde}_verification.log')
    ssh('server197',f"python3 - <<'PY'\nfrom pathlib import Path\nimport hashlib,json,time\ns=Path({stage!r}); d=Path({CANONICAL!r})/'hard_sampling'/{pde!r}\ndef sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()\nfor p in d.iterdir():\n assert p.name in ['selection.json','selected_reference.pt'],p\n assert sha(p)==sha(s/p.name),p\nfor p in list(s.iterdir()):\n target=d/p.name\n if target.exists(): p.unlink()\n else: p.replace(target)\ns.rmdir()\n(d/'publication.json').write_text(json.dumps(dict(source_host='server216',temporary_source={source!r},canonical=str(d),all_files_sha256_verified=True,time=time.time()),indent=2)+'\\n')\nPY")
    print('SAMPLING_PUBLISHED',pde,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pde',choices=['helmholtz','darcy'],required=True)
    main(p.parse_args().pde)
