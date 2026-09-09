"""Read effective configurations from completed NS batches; stage settings only."""
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import torch

PAPER = Path(__file__).resolve().parents[2]
STUDY = Path('/home/tat512/C01Python/audit/ns_main_revision_0909')
OUT = PAPER / 'audit/revision_0909/ns_effective_settings'
OUT.mkdir(exist_ok=True)
KEYS = ['checkpoint_path','zeta_obs_a','zeta_obs_u','zeta_pde','clip_mode',
        'clip_threshold','pde_guidance_start_ratio','pde_guidance_ramp_ratio',
        'shared_mask','mask_seed','sample_seed','num_obs','num_steps',
        'sampler_phase','time_grid','step_method','loss_state',
        'gradient_target','residual_mode','ns_operator_mode',
        'stochastic_guidance_coeff','stochastic_guidance_time','dtype']
selection = json.loads((STUDY/'selection.json').read_text())
records = []
by_setting = defaultdict(list)
for dist in ['id','smooth','rough']:
    for setting in ['full_forward','full_inverse','sparse_forward','sparse_inverse','sparse_joint']:
        path = STUDY/'main_results'/dist/setting/'offset2700.pt'
        receipt = json.loads(path.with_suffix('.json').read_text())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert receipt['result_sha256'] == digest
        assert receipt['protocol_sha256'] == selection['protocol_sha256']
        assert receipt['selection_sha256'] == hashlib.sha256((STUDY/'selection.json').read_bytes()).hexdigest()
        assert receipt['worker'] == 1 and receipt['ids'] == list(range(2700,2716))
        d = torch.load(path,map_location='cpu',weights_only=False)
        # The file hash is added after serialization and cannot be embedded in
        # the serialized file itself; every other receipt field must match.
        assert d['receipt'] == {k:v for k,v in receipt.items() if k!='result_sha256'}
        c = {k:d['config'][k] for k in KEYS}
        assert c['num_steps'] == receipt['nfe'] == 100
        assert c['pde_guidance_start_ratio'] == .8
        assert c['pde_guidance_ramp_ratio'] == 0
        assert c['shared_mask'] is False
        assert c['residual_mode'] == 'endpoint_secant'
        expected=(0.,2400000.,300.,100.) if 'inverse' in setting else (60000.,60000.,.1,1e10)
        assert tuple(c[k] for k in ['zeta_obs_a','zeta_obs_u','zeta_pde','clip_threshold']) == expected
        assert c['num_obs'] == (16384 if setting.startswith('full') else 500)
        by_setting[setting].append(c)
        records.append(dict(distribution=dist,setting=setting,source=str(path),
                            source_sha256=digest,effective_configuration=c,
                            runtime_batch_size=receipt['batch_size'],runtime_tf32=receipt['tf32']))
for setting, configs in by_setting.items():
    assert len(configs) == 3 and configs[0] == configs[1] == configs[2],setting
(OUT/'effective_settings.json').write_text(json.dumps(dict(
    status='prepared_from_validated_batch_receipts',scope='Settings only; this is not the complete 15000-example accuracy audit.',
    model_identity_source=str(STUDY/'MODEL_AND_METHOD_FACTS.md'),
    batch_note='The saved scalar configuration has batch_size=1. Actual inference batching is controlled separately and recorded in each receipt; do not infer runtime batch size from that scalar field.',
    records=records),indent=2)+'\n')
rows = [
 r'Navier--Stokes & F & Full & I/S/R & $6\!\times\!10^{4}$ & $6\!\times\!10^{4}$ & $0.1$ & $10^{10}$ & Late \\',
 r'Navier--Stokes & I & Full & I/S/R & $0$ & $2.4\!\times\!10^{6}$ & $300$ & $100$ & Late \\',
 r'Navier--Stokes & J & Random & I/S/R & $6\!\times\!10^{4}$ & $6\!\times\!10^{4}$ & $0.1$ & $10^{10}$ & Late \\',
 r'Navier--Stokes & F & Random & I/S/R & $6\!\times\!10^{4}$ & $6\!\times\!10^{4}$ & $0.1$ & $10^{10}$ & Late \\',
 r'Navier--Stokes & I & Random & I/S/R & $0$ & $2.4\!\times\!10^{6}$ & $300$ & $100$ & Late \\',
]
(OUT/'main_hyperparameter_rows.tex').write_text('\n'.join(rows)+'\n')
print('Prepared five setting rows from 15 effective configurations. Main accuracy gate remains required.')
