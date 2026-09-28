"""Verified physical inputs from the published controlled timing experiment."""
import json
import os
from pathlib import Path

from experiments.paper.run import ROOT, digest


class TimingInputs:
    def __init__(self, pde):
        import numpy as np
        root=Path(os.environ.get('TIMING_INPUT_ROOT_'+pde.upper(),
                  os.environ.get('TIMING_INPUT_ROOT',ROOT/'datasets/paper/sampling_time')))
        self.pde=pde
        self.root=root
        self.protocol=json.loads((root/'protocol.json').read_text())
        self.truth_path=root/'source/timing_truths.npz'
        self.mask_path=root/'masks.npz'
        self.weight=root/'weights'/f'fm_{pde}.pth'
        self.hashes={}
        expected={x['path']:x['sha256'] for x in self.protocol['artifacts']}
        for path in [self.truth_path,self.mask_path,self.weight]:
            actual=digest(path)
            if actual!=expected[str(path.relative_to(root))]:
                raise ValueError(f'Frozen timing input checksum mismatch: {path}')
            self.hashes[str(path.relative_to(root))]=actual
        self.hashes['protocol.json']=digest(root/'protocol.json')
        self.data=np.load(self.truth_path)
        self.masks=np.load(self.mask_path)
        self.ids=self.protocol['evaluation_ids']
        self.pilot=self.protocol['pilot_id']
        self.order=[*self.ids,self.pilot]

    def overrides(self):
        # Only scientific controls retained in the cleaned implementation.
        old=self.protocol['fm_configs'][self.pde]
        keys=['guidance_components','loss_state','sampler_phase','switch_ratio',
              'clip_threshold','pde_guidance_start_ratio','stochastic_guidance_coeff',
              'zeta_obs_a','zeta_obs_u','zeta_pde','time_grid','time_grid_eta',
              'enforce_boundary_conditions','boundary_condition_mode','bc_weight',
              'endpoint_bc_weight','boundary_residual_normalization','allow_unknown_boundary_conditions']
        values={k:old[k] for k in keys if k in old}
        if self.pde == 'helmholtz':
            # Use the current joint clipping threshold; the archived timing
            # protocol contains the superseded 1e10 threshold.
            values.pop('clip_threshold', None)
        values.update(checkpoint_path=str(self.weight),data_path=str(self.truth_path),
                      model_profile='auto',shared_mask=False)
        if self.pde=='burger':
            values.update(sensor_mode='sensor_column',num_sensor_columns=5,num_obs=640)
        return values

    def case(self, index, device):
        import torch
        from data.specs import get_pde_spec
        from sampling.data import PDEGroundTruth
        from sampling.masks import PairMasks
        j=self.order.index(index);pde=self.pde;spec=get_pde_spec(pde)
        a=torch.from_numpy(self.data[pde+'_a'][j:j+1]).to(device)
        u=torch.from_numpy(self.data[pde+'_u'][j:j+1]).to(device)
        ma=torch.from_numpy(self.masks[f'{pde}_{index}_a'])[None,None].to(device)
        mu=torch.from_numpy(self.masks[f'{pde}_{index}_u'])[None,None].to(device)
        metadata=dict(sample_ids=[str(index)],sample_offsets=[index],offset=index,
                      batch_size=1,synthetic=False,endpoint_pair=pde!='burger')
        truth=PDEGroundTruth(pde,a,u,a if pde=='burger' else torch.cat([a,u],1),{},
                             list(spec.coef_channel_names),list(spec.sol_channel_names),metadata)
        return truth,PairMasks(ma,mu,dict(source='published controlled timing inputs'))
