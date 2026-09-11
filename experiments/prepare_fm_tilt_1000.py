"""Freeze and independently verify all 1000 real main-experiment inputs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from scripts.train.main_resume_sampling import historical_inputs, tensor_sha
from scripts.train.resume_study import file_sha, write


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--record', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    assert args.output.is_absolute() and '/outputs/' in str(args.output)
    out = args.output / 'inputs'
    out.mkdir(parents=True, exist_ok=True)
    if (out/'protocol.json').exists():
        raise FileExistsError(out/'protocol.json')
    torch.set_num_threads(2)
    record = json.loads(args.record.read_text())
    assert record['n'] == 1000 and len(set(record['sample_ids'])) == 1000
    cases, rows, sources = {}, {}, []
    start = time.monotonic()
    for batch in record['batches']:
        selected = sorted([r for r in record['rows'] if r['result_path'] == batch['result_path']], key=lambda r:r['sample_id'])
        gt, masks, ids = historical_inputs(record, selected, 'cpu')
        payload = torch.load(batch['result_path'], map_location='cpu', weights_only=False)
        pred = (payload['predictions'] if record['pde']=='nsnonbounded' else
            payload['sol_final'] if record['pde']=='burger' else
            torch.cat([payload['coef_final'],payload['sol_final']],1))
        for i, row in enumerate(selected):
            sid = row['sample_id']
            assert sid not in cases
            prediction = pred[row['result_row']:row['result_row']+1]
            truth = gt.pair[i:i+1]
            errors = (torch.linalg.vector_norm((prediction-truth).double().flatten(2),dim=2)
                /torch.linalg.vector_norm(truth.double().flatten(2),dim=2))[0].tolist()
            expected = [row['error_u']] if record['pde']=='burger' else [row['error_a'],row['error_u']]
            assert np.allclose(errors, expected, rtol=2e-6, atol=2e-8)
            from scripts.train.main_resume_sampling import slice_params
            cases[sid] = dict(truth=truth, baseline=prediction,
                masks=dict(coef=masks.coef[i:i+1], sol=masks.sol[i:i+1]),
                params=slice_params(gt.pde_params,[i],len(ids),'cpu'))
            rows[sid] = dict(sample_id=sid, baseline_errors=errors, truth_sha256=tensor_sha(truth),
                mask_sha256=tensor_sha(torch.cat([masks.coef[i:i+1],masks.sol[i:i+1]],1)),
                baseline_sha256=tensor_sha(prediction), source=row)
        sources.append(dict(path=batch['result_path'], sha256=batch['result_sha256'], sample_ids=ids))
        print('VERIFIED',len(cases),flush=True)
    assert sorted(cases)==sorted(record['sample_ids'])
    torch.save(cases, out/'cases.pt')
    weights = Path(record['inference_checkpoint'])
    protocol = dict(pde=record['pde'], task=record['task'], distribution=record['dist'],
        sample_ids=record['sample_ids'], n=1000, config=record['config'],
        checkpoint_path=str(weights), checkpoint_sha256=file_sha(weights),
        source_record=str(args.record), source_record_sha256=file_sha(args.record),
        cases_sha256=file_sha(out/'cases.pt'), sources=sources,
        rows=[rows[i] for i in record['sample_ids']],
        verification='All 1000 distinct truths, masks and saved predictions verified against source hashes; every historical error recomputed in FP64.',
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        seconds=time.monotonic()-start)
    write(out/'protocol.json',protocol)
    print('COMPLETE',protocol['n'],protocol['checkpoint_sha256'],flush=True)


if __name__=='__main__':
    main()
