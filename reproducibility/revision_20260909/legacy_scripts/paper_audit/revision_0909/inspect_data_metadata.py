"""Read-only metadata audit; no complete field arrays are loaded."""
from pathlib import Path
import json
import numpy as np
import h5py
from scipy.io import loadmat, whosmat
root = Path('/large_storage/zhangxf/PDEdata')
families = ['poisson','helmholtz','darcy','nsnonbounded','burgers','reaction_diffusion','shallow_water','heat','wave','advection_diffusion','steady_heat_conduction']
def conv(x):
    if isinstance(x, bytes): return x.decode('utf8',errors='replace')
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,np.generic): return x.item()
    return x
for family in families:
    paths = sorted(p for p in (root/family).glob('*') if p.suffix in ('.mat','.h5'))
    test = [p for p in paths if any(p.stem.endswith('_'+s) for s in ['id','smooth','rough'])]
    train = [p for p in paths if 'test' not in p.name and ('_new' in p.name if family == 'nsnonbounded' else True)]
    rec={'family':family,'training_files':[p.name for p in train], 'test_files':[p.name for p in test], 'inspected':[]}
    for p in (train+test):
        info={'path':str(p)}
        try:
            if h5py.is_hdf5(p):
                with h5py.File(p,'r') as f:
                    info['attrs']={k:conv(v) for k,v in f.attrs.items()}
                    info['shapes']={k:list(f[k].shape) for k in f.keys() if not k.isdigit() and isinstance(f[k],h5py.Dataset)}
                    groups=[k for k in f if k.isdigit()]
                    if groups:
                        info['n_samples']=len(groups)
                        g=f[sorted(groups)[0]]
                        info['first_sample_attrs']={k:conv(v) for k,v in g.attrs.items()}
                        if 'grid' in g:
                            info['grid']={k:conv(g['grid'][k][()]) for k in g['grid'] if k in ['t']}
                            info['grid_bounds']={k:[float(g['grid'][k][0]),float(g['grid'][k][-1]),len(g['grid'][k])] for k in g['grid'] if k in ['x','y']}
                        if 'data' in g:
                            d=g['data']; info['sample_shapes']=list(d.shape) if isinstance(d,h5py.Dataset) else {k:list(v.shape) for k,v in d.items()}
                    for k in ['t','T','dt','tspan','grf_alpha','grf_tau','grf_gamma','k','viscosity','dataset_type','domain','boundary_condition','initial_grf_smoothness','initial_grf_tau']:
                        if k in f and isinstance(f[k],h5py.Dataset) and f[k].size<200: info[k]=conv(f[k][()])
            else:
                shapes=whosmat(p); info['shapes']={k:list(shape) for k,shape,_ in shapes}
                names=[k for k,shape,_ in shapes if np.prod(shape)<500]
                d=loadmat(p,variable_names=names,squeeze_me=True)
                info['metadata']={k:conv(v) for k,v in d.items() if not k.startswith('__')}
        except Exception as e: info['error']=str(e)
        rec['inspected'].append(info)
    print(json.dumps(rec,default=conv),flush=True)
