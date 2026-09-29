"""Fixed timing inputs, reconstructed from the released data by default."""
import hashlib
import json
import os
from pathlib import Path

from experiments.paper.run import ROOT, digest


def scientific_controls(pde):
    """Use manuscript joint-guidance settings, with stochastic timing updates."""
    from sampling.config import load_config
    main=load_config(ROOT/f'configs/main/both/{pde}.yaml').asdict()
    keys=['guidance_components','loss_state','sampler_phase','switch_ratio',
          'clip_threshold','pde_guidance_start_ratio','stochastic_guidance_coeff',
          'zeta_obs_a','zeta_obs_u','zeta_pde','time_grid','time_grid_eta',
          'enforce_boundary_conditions','boundary_condition_mode','bc_weight',
          'endpoint_bc_weight','boundary_residual_normalization','allow_unknown_boundary_conditions']
    values={k:main[k] for k in keys}
    values.update(model_profile='auto',shared_mask=False,sampler_phase='stochastic')
    if pde=='burger':
        values.update(sensor_mode='time_slices',num_sensor_columns=5,num_obs=640)
    return values


def plan_overrides(pde):
    """Expose the actual timing controls without loading data or a model."""
    legacy=os.environ.get('TIMING_INPUT_ROOT_'+pde.upper(),os.environ.get('TIMING_INPUT_ROOT'))
    values=scientific_controls(pde)
    if legacy:
        values.update(checkpoint_path=str(Path(legacy)/'weights'/f'fm_{pde}.pth'),
                      data_path=str(Path(legacy)/'source/timing_truths.npz'))
    return values


class TimingInputs:
    def __init__(self, pde):
        import numpy as np
        self.pde=pde
        legacy_root=os.environ.get('TIMING_INPUT_ROOT_'+pde.upper(),
                                   os.environ.get('TIMING_INPUT_ROOT'))
        self.public=not legacy_root
        if self.public:
            self._from_released_data()
            return
        root=Path(legacy_root)
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
        self._verify_random_selection()

    def _verify_random_selection(self):
        """Recreate the original random draw; no predictions enter this check."""
        import numpy as np
        seed=self.protocol['seed']
        drawn=np.random.default_rng(seed).choice(1000,21,replace=False).tolist()
        if self.order!=drawn:
            raise ValueError('Timing IDs do not follow the recorded random selection rule')
        for index in self.order:
            rng=np.random.default_rng(seed+index)
            a=np.zeros((128,128),dtype=np.float32)
            u=np.zeros_like(a)
            if self.pde=='burger':
                levels=rng.choice(128,5,replace=False)
                layout=self.protocol.get('burger_observation_layout','sensor_column')
                if layout=='time_slices':
                    u[levels,:]=1
                elif layout=='sensor_column':
                    u[:,levels]=1
                else:
                    raise ValueError(f'Unknown Burgers timing mask layout: {layout}')
                a=u.copy()
            else:
                a.flat[rng.choice(128*128,500,replace=False)]=1
                u.flat[rng.choice(128*128,500,replace=False)]=1
            for field,expected in [('a',a),('u',u)]:
                if not np.array_equal(self.masks[f'{self.pde}_{index}_{field}'],expected):
                    raise ValueError(f'Timing observation mask violates the recorded random rule: {self.pde}/{index}/{field}')

    def _from_released_data(self):
        import numpy as np
        from sampling.config import load_config
        self.root=ROOT/'configs/experiments/comparison'
        protocol_path=self.root/'timing_protocol.json'
        self.protocol=json.loads(protocol_path.read_text())
        source=self.protocol['inputs'][self.pde]
        self.distribution=source['distribution']
        self.input_config=load_config(ROOT/f'configs/main/both/{self.pde}.yaml',
                                     dict(test_type=self.distribution,device='cpu',batch_size=1))
        self.truth_path=Path(self.input_config.data_path)
        self.mask_path=self.root/'timing_masks.npz'
        self.weight=Path(self.input_config.checkpoint_path)
        self.hashes={'timing_protocol.json':digest(protocol_path)}
        expected=[(self.truth_path,source['sha256']),
                  (self.mask_path,self.protocol['masks_sha256']),
                  (self.weight,self.protocol['fm_weights'][self.pde]['sha256'])]
        for path,wanted in expected:
            actual=digest(path)
            if actual!=wanted:
                raise ValueError(f'Published timing asset checksum mismatch: {path}')
            self.hashes[str(path)]=actual
        self.masks=np.load(self.mask_path)
        self.ids=self.protocol['evaluation_ids']
        self.pilot=self.protocol['pilot_id']
        self.order=[*self.ids,self.pilot]
        self._verify_random_selection()

    def diffusion_checkpoint(self, default):
        """Resolve and verify the original baseline weights for either input source."""
        if self.public:
            path=Path(default)
            expected=self.protocol['diffusion_weights'][self.pde]['sha256']
        else:
            path=self.root/'weights'/Path(default).name
            artifacts={x['path']:x['sha256'] for x in self.protocol['artifacts']}
            expected=artifacts[str(path.relative_to(self.root))]
        actual=digest(path)
        if actual!=expected:
            raise ValueError(f'Published DiffusionPDE timing checkpoint checksum mismatch: {path}')
        self.hashes[str(path)]=actual
        return path

    def overrides(self):
        values=scientific_controls(self.pde)
        values.update(checkpoint_path=str(self.weight),data_path=str(self.truth_path),
                      model_profile='auto',shared_mask=False)
        return values

    def observation_mask(self, index, field):
        """Return manuscript masks, also when reading an old spatial-column bundle."""
        mask=self.masks[f'{self.pde}_{index}_{field}']
        if self.pde=='burger' and self.protocol.get('burger_observation_layout','sensor_column')=='sensor_column':
            mask=mask.T.copy()
        return mask

    def case(self, index, device):
        import torch
        from data.specs import get_pde_spec
        from sampling.data import PDEGroundTruth
        from sampling.masks import PairMasks
        j=self.order.index(index);pde=self.pde;spec=get_pde_spec(pde)
        if self.public:
            from dataclasses import replace
            from sampling.data import load_ground_truth
            source=load_ground_truth(replace(self.input_config,offset=index))
            for field,tensor in [('a',source.coef),('u',source.sol)]:
                actual=hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()
                expected=self.protocol['truth_sha256'][pde][str(index)][field]
                if actual!=expected:
                    raise ValueError(f'Timing physical input differs from the recorded field: {pde}/{index}/{field}')
            a,u=source.coef.to(device),source.sol.to(device)
        else:
            a=torch.from_numpy(self.data[pde+'_a'][j:j+1]).to(device)
            u=torch.from_numpy(self.data[pde+'_u'][j:j+1]).to(device)
        ma=torch.from_numpy(self.observation_mask(index,'a'))[None,None].to(device)
        mu=torch.from_numpy(self.observation_mask(index,'u'))[None,None].to(device)
        metadata=dict(sample_ids=[str(index)],sample_offsets=[index],offset=index,
                      batch_size=1,synthetic=False,endpoint_pair=pde!='burger')
        truth=PDEGroundTruth(pde,a,u,a if pde=='burger' else torch.cat([a,u],1),{},
                             list(spec.coef_channel_names),list(spec.sol_channel_names),metadata)
        return truth,PairMasks(ma,mu,dict(source='controlled timing inputs',
                                        sensor_mode='time_slices' if pde=='burger' else 'random'))
