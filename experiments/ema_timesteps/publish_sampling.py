"""Verify and atomically return completed sampling variants to canonical storage."""
import argparse
import json
import subprocess
import time

from experiments.ema_timesteps.transport import ssh, pipe
from experiments.ema_timesteps.transfer_inputs import CANONICAL, CACHE
from experiments.ema_timesteps.relay_sampling import remote_python


def publish(pde, variant):
    relative = f'evaluation/{pde}/variants/{variant}'
    state = json.loads(remote_python('server197', f'''
from pathlib import Path
import json
p=Path({(CANONICAL+'/'+relative+'/publication.json')!r})
print(p.read_text() if p.exists() else 'null')
'''))
    if state is not None:
        assert state['status'] == 'verified'
        return
    manifest = json.loads(remote_python('server216', f'''
from pathlib import Path
import hashlib,json
r=Path({(CACHE+'/'+relative)!r})
assert (r/'complete.json').exists()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
files={{p.name:sha(p) for p in sorted(r.iterdir()) if p.is_file() and p.name!='transfer_manifest.json'}}
assert not any(name.endswith('.tmp') for name in files)
(r/'transfer_manifest.json').write_text(json.dumps(files,indent=2))
print(json.dumps(files))
'''))
    stage = f'{CANONICAL}/incoming_sampling_{pde}_{variant}'
    ssh('server197', f'mkdir -p {stage}')
    pipe('server216', f'tar -C {CACHE}/evaluation/{pde}/variants -cf - {variant}',
         'server197', f'tar -C {stage} -xf -')
    print(remote_python('server197', f'''
from pathlib import Path
import hashlib,json
stage=Path({stage!r});src=stage/{variant!r}
expected=json.loads({json.dumps(manifest)!r})
assert json.loads((src/'transfer_manifest.json').read_text())==expected
assert {{p.name for p in src.iterdir()}}==set(expected)|{{'transfer_manifest.json'}}
for name,digest in expected.items():
 h=hashlib.sha256()
 with (src/name).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 assert h.hexdigest()==digest,name
destination=Path({(CANONICAL+'/'+relative)!r});destination.parent.mkdir(parents=True,exist_ok=True)
assert not destination.exists()
(src/'publication.json').write_text(json.dumps(dict(status='verified',source={CACHE!r},files=expected),indent=2))
src.replace(destination);stage.rmdir()
print('SAMPLING_PUBLISHED',{pde!r},{variant!r},len(expected))
'''), flush=True)
    ssh('server197', f'cd {CANONICAL}/code_sampling && '
        '/research_data/users/zhangxifeng/.conda/envs/fm4pde/bin/python '
        f'-m experiments.ema_timesteps.sampling report --root {CANONICAL} --pde {pde}')


def run(pdes, watch, epoch, required_variants=None):
    while True:
        try:
            states = json.loads(remote_python('server216', f'''
from pathlib import Path
import json
r=Path({CACHE!r});result={{}}
for pde in {pdes!r}:
 folder=r/'evaluation'/pde
 variants=sorted([p.parent.name for p in (folder/'variants').glob('*/complete.json')],
                 key=lambda name:(name!='original',name))
 queue=json.loads((folder/'queue.json').read_text()) if (folder/'queue.json').exists() else {{}}
 result[pde]=dict(variants=variants,queue=queue)
print(json.dumps(result))
'''))
            for pde, state in states.items():
                for variant in state['variants']:
                    publish(pde, variant)
            finished = (all(set(required_variants) <= set(s['variants']) for s in states.values())
                        if required_variants else
                        all(s['queue'].get('state') == 'complete' and s['queue'].get('epoch') == epoch
                            for s in states.values()))
            if not watch or finished:
                return
        except subprocess.CalledProcessError as exc:
            if not watch or exc.returncode != 255:
                raise
            print('SSH_DISCONNECTED_RETRY_FROM_PUBLICATION_MARKERS', str(exc), flush=True)
        except RuntimeError as exc:
            if not watch or not str(exc).startswith('Transfer failed source='):
                raise
            print('TRANSFER_INTERRUPTED_RETRY_VERIFIED_PUBLICATION', str(exc), flush=True)
        time.sleep(30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdes', nargs='+', default=['nsnonbounded','poisson','helmholtz','darcy','burger'])
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--epoch', type=int, default=2)
    parser.add_argument('--required-variants', nargs='+',
                        help='Wait for these completed variants for every PDE, including later epochs')
    args = parser.parse_args()
    run(args.pdes, args.watch, args.epoch, args.required_variants)
