"""Locate historical sampling discrepancies before attributing model improvements."""
import json
from pathlib import Path
import torch
from experiments.optimizer_diagnostics.hard_sampling import REFERENCE
from experiments.optimizer_diagnostics.study import write
from experiments.aligned_sampling.input_sources import load_cell
from experiments.aligned_sampling.run_inference import configure_runtime,effective_config,observation_batch,infer,relative_differences
from experiments.rough_stress.noise import NoiseBank
from sampling.model_io import load_fm4pde_checkpoint_bundle


def main(root):
    out=root/'hard_sampling'
    configure_runtime('cuda:0',False,4)
    selection=json.loads((out/'selection.json').read_text())
    data=load_cell(REFERENCE,selection['cell'])
    reference=torch.load(out/'selected_reference.pt',map_location='cpu',weights_only=False)
    bank=NoiseBank('/large_storage/zhangxf/outputs/FM4PDE/rough_stress_20260918/setup/noise_a100_seed0')
    cp=REFERENCE/selection['cell']['checkpoint']['path']
    bundle=load_fm4pde_checkpoint_bundle(str(cp),'nsnonbounded','cuda:0',prefer_ema=False)
    results={};predictions={}
    for size,replay in [(1,False),(1,True),(4,True)]:
        ids=selection['indices'][:size]
        cfg=effective_config(selection['cell'],cp,'cuda:0',ids,1000,out)
        gt,masks,hashes=observation_batch(data,cfg,ids,'cuda:0')
        if replay:
            with bank.replay(ids): pred,meta=infer(cfg,bundle,gt,masks,ids)
        else: pred,meta=infer(cfg,bundle,gt,masks,ids)
        key=f'batch{size}_bank{int(replay)}'
        predictions[key]=pred
        result=dict(archived_difference=relative_differences(pred,reference['prediction'][:size]),runtime=meta,input_hashes=hashes)
        if key!='batch1_bank0': result['first_case_vs_native']=relative_differences(pred[:1],predictions['batch1_bank0'])
        results[key]=result;write(out/'replay_diagnosis.json',results)
        torch.save(pred,out/f'replay_{key}.pt')
        print(key,json.dumps(result),flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    main(p.parse_args().root)
