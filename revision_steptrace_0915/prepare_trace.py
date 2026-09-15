"""Freeze original controlled-timing inputs, weights and exact sampler sources."""
import argparse, gzip, hashlib, json, pathlib, shutil, subprocess, tarfile

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=pathlib.Path,required=True);ap.add_argument('--fm-repo',type=pathlib.Path,required=True);ap.add_argument('--audit',type=pathlib.Path,required=True);a=ap.parse_args()
 root=a.output;root.mkdir(parents=True,exist_ok=True)
 if (root/'manifest.json').exists():raise FileExistsError(root/'manifest.json')
 audit=json.load(gzip.open(a.audit,'rt'));manifest={'version':1,'scope':'5 PDE x 20 fixed Smooth examples x 2 methods x 1000 steps; extra pilot ID500','cells':{},'artifacts':[],'timing_note':'Instrumented sampler wall time excludes synchronized metric callbacks; original uninstrumented resident-model timings are retained separately.'}
 def freeze(src,rel):
  src=pathlib.Path(src);dst=root/rel;dst.parent.mkdir(parents=True,exist_ok=True)
  if not dst.exists():shutil.copy2(src,dst)
  assert sha(src)==sha(dst)
  manifest['artifacts'].append({'path':str(rel),'sha256':sha(dst),'source':str(src)})
  return str(rel)
 for pde,info in audit['pde_protocols'].items():
  inp=pathlib.Path(info['inputs']);results=pathlib.Path(info['results'])/pde;old=json.load(open(inp/'protocol.json'));sub=pathlib.Path('inputs')/pde
  freeze(inp/'protocol.json',sub/'protocol.json');freeze(inp/'source/timing_truths.npz',sub/'truths.npz');freeze(inp/'masks.npz',sub/'masks.npz')
  fm=freeze(inp/'weights'/f'fm_{pde}.pth',sub/'fm.pth')
  dmitem=next(x for x in old['artifacts'] if x['path'].endswith('.pkl') and (pde.replace('burger','burgers').replace('nsnonbounded','ns-nonbounded') in x['path']))
  dm=freeze(inp/dmitem['path'],sub/'diffusion.pkl')
  source=freeze(results/'diffusion_effective_sampler.py',sub/'diffusion_effective_sampler.py')
  for method in ['FM4PDE','DiffusionPDE']:
   for n in [100,1000]:
    for i in old['evaluation_ids']:
     for ext in ['pt','json']:freeze(results/f'{method}_{n}_{i}.{ext}',sub/'original_timing'/f'{method}_{n}_{i}.{ext}')
  commit=old['fm_commit'];codedir=root/'sources'/commit
  if not codedir.exists():
   codedir.mkdir(parents=True);archive=root/'sources'/f'{commit}.tar'
   with open(archive,'wb') as f:subprocess.run(['git','archive',commit],cwd=a.fm_repo,stdout=f,check=True)
   with tarfile.open(archive) as f:f.extractall(codedir,filter='data')
   manifest['artifacts'].append({'path':str(archive.relative_to(root)),'sha256':sha(archive),'source':f'git:{a.fm_repo}@{commit}'})
  manifest['cells'][pde]={'input_root':str(sub),'fm_weights':fm,'diffusion_weights':dm,'diffusion_source':source,'fm_code':str(codedir.relative_to(root)),'fm_commit':commit,'diffusion_commit':old['diffusion_commit'],'original_protocol_sha256':sha(inp/'protocol.json'),'evaluation_ids':old['evaluation_ids'],'pilot_id':old['pilot_id']}
 (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(root/'manifest.json')
if __name__=='__main__':main()
