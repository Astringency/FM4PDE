"""Run single-input ablations, parameter selection trials and seed ensembles.

Prepare from the read-only original archive on server197. Run in an isolated
Git checkout. No original configuration, checkpoint, or output is overwritten.
Each reported score retains coefficient and solution fields separately.
"""
from __future__ import annotations

import argparse
import copy
import csv
import dataclasses
import fcntl
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from contextlib import redirect_stderr, redirect_stdout

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sampling.study_models import add_model_arguments, model_config, bind_model, print_model_plan
PDES = ['poisson', 'helmholtz', 'darcy', 'nsnonbounded', 'burger',
        'reaction_diffusion', 'shallow_water', 'heat', 'wave',
        'advection_diffusion', 'steady_heat_conduction']
EXTRA = PDES[5:]
REVISED = ['poisson', 'nsnonbounded'] + EXTRA
EXCLUDED_GROUPS = {'temporal_residual_mode', 'deterministic_endpoint_bt'}
TUNING_IDS = [1100, 1101, 1102, 1103]
EVALUATION_IDS = list(range(1500, 1532))
PARAM_KEYS = ['zeta_obs_a', 'zeta_obs_u', 'zeta_pde', 'clip_mode', 'clip_threshold']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def prepare(args):
    import torch
    import scipy.io
    from sampling.config import load_config
    from sampling.data import load_ground_truth
    from scripts.training.export_checkpoint import make_inference_checkpoint
    target = args.inputs
    target.mkdir(parents=True, exist_ok=True)
    # The MATLAB loader otherwise reads the entire multi-GB file for each ID.
    # Reuse one immutable raw file while extracting a PDE's fixed input cohort.
    original_loadmat = scipy.io.loadmat
    scipy.io.loadmat = functools.lru_cache(maxsize=1)(original_loadmat)
    recommendation_file = args.archive / 'outputs/tuning/six_pde_sampling_refine/recommended_params_standard.json'
    recommendations = json.loads(recommendation_file.read_text())
    summary_file = args.archive / 'outputs/ablations/summary_latest_unique.csv'
    rows = list(csv.DictReader(summary_file.open()))
    for pde in args.pdes:
        protocol_path = target / pde / 'protocol.json'
        if protocol_path.exists():
            print('EXISTS', protocol_path, flush=True)
            continue
        dest = target / pde
        dest.mkdir(parents=True, exist_ok=True)
        archived = []
        keys = {f.name for f in dataclasses.fields(__import__('sampling.config', fromlist=['AblationConfig']).AblationConfig)}
        for index, row in enumerate(rows):
            if row['pde'] != pde:
                continue
            path = args.archive / row['run_dir'] / 'resolved_config.yaml'
            cfg = load_config(path).asdict()
            assert not set(cfg) - keys
            archived.append(dict(id=f'archive_{index:04d}', config=cfg,
                                 source_config=str(path), source_config_sha256=digest(path),
                                 source_result=str(path.parent / 'result.pt'),
                                 old_errors={f:float(row['rel_l2_'+f]) for f in ['a','u']}))
        base = {task:load_config(ROOT/f'configs/main/{task}/{pde}.yaml').asdict()
                for task in (['both'] if pde == 'burger' else ['both','forward','inverse'])}
        ckpt = Path(model_config(args, pde).checkpoint_path)
        weight = dest/'weights.pth'
        make_inference_checkpoint(ckpt, weight)
        truths = {}
        for i in [0]+TUNING_IDS+EVALUATION_IDS:
            cfg = load_config(ROOT/f'configs/main/both/{pde}.yaml', overrides=dict(
                offset=i, batch_size=1, device='cpu', allow_synthetic_data=False))
            truths[i] = load_ground_truth(cfg)
            assert truths[i].metadata['synthetic'] is False
            print('CACHE', pde, i, flush=True)
        cache = dest/'truths.pt'
        torch.save(truths, cache)
        # Check the original main-ablation input rather than assume index equality.
        old = torch.load(archived[0]['source_result'], map_location='cpu', weights_only=False)
        assert torch.equal(old['coef_ground_truth'], truths[0].coef), pde
        assert torch.equal(old['sol_ground_truth'], truths[0].sol), pde
        del old
        protocol = dict(version=1, pde=pde, original_main_id=0, tuning_ids=TUNING_IDS,
            evaluation_ids=EVALUATION_IDS, inference_seeds=[0,1,2], base_configs=base,
            archived=archived, recommendations=[r for r in recommendations if r['pde']==pde],
            recommendation_source=str(recommendation_file), recommendation_sha256=digest(recommendation_file),
            archive_summary_sha256=digest(summary_file), weights_sha256=digest(weight),
            truth_sha256=digest(cache), preparation_commit=subprocess.check_output(
                ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            revision_scope='Archive main input and masks retained; calibrated observation weights and clipping only. '
                           'Archive solver, phase, grid, residual and ablated variables are retained.',
            tuning_scope='Four ID inputs disjoint from archived main ID 0 and the new 32-input ensemble cohort. '
                         'No claim of independence from unknown historical training/calibration decisions.',
            ensemble_scope='32 predeclared ID physical inputs, three seeds, both task, 100 steps. '
                           'Physical fields are averaged before scoring. Report a/u separately; Burgers u only.',
            no_rerun_groups=sorted(EXCLUDED_GROUPS),
            clipping_candidates=[50.,100.,500.,1000.] if pde in ['poisson','nsnonbounded'] else [],
            tuning_observation_counts=[50,100,500] if pde in ['poisson','nsnonbounded'] else [500])
        write(protocol_path, protocol)
        scipy.io.loadmat.cache_clear()
        print('PREPARED', pde, len(archived), flush=True)


def physical_errors(pred, truth):
    import torch
    p = pred.detach().cpu().double().flatten(1)
    t = truth.detach().cpu().double().flatten(1)
    return (torch.linalg.vector_norm(p-t,dim=1)/torch.linalg.vector_norm(t,dim=1).clamp_min(1e-12)).tolist()


def run(args):
    import torch
    from sampling.config import AblationConfig
    from sampling.model_io import load_fm4pde_checkpoint_bundle
    from sampling.batching import combine_truths
    import sampling.runner as runner
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    for pde in args.pdes:
        source = args.inputs/pde
        protocol = json.loads((source/'protocol.json').read_text())
        ph = digest(source/'protocol.json')
        assert digest(source/'truths.pt') == protocol['truth_sha256']
        target = args.output/pde
        target.mkdir(parents=True, exist_ok=True)
        with (target/f'{args.mode}.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            selected_model = model_config(args, pde)
            bind_model(target, selected_model)
            write(target/f'environment_{args.mode}.json',dict(host=socket.gethostname(), pid=os.getpid(),
                protocol_sha256=ph, python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(), visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                tf32=False, commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
            truths = torch.load(source/'truths.pt',map_location='cpu',weights_only=False)
            bundle = load_fm4pde_checkpoint_bundle(selected_model.checkpoint_path,pde,'cuda:0',model_profile=selected_model.model_profile)
            calls = {'count':0}
            def hook(*_): calls['count'] += 1
            handle = bundle[0].model.register_forward_hook(hook)

            def one(label, conf, ids, stage):
                folder = target/stage/label
                receipt = folder/'receipt.json'
                if receipt.exists():
                    row = json.loads(receipt.read_text())
                    assert row['protocol_sha256']==ph and row['sample_ids']==ids
                    return row
                folder.mkdir(parents=True, exist_ok=True)
                c = copy.deepcopy(conf)
                c.update(checkpoint_path=selected_model.checkpoint_path, model_profile=selected_model.model_profile, device='cuda:0', output_dir=str(folder),
                         batch_size=len(ids), offset=ids[0], save_plots=False,save_intermediate=False,
                         save_per_sample_curves=True,ablation_name='paper_revision',allow_synthetic_data=False)
                cfg = AblationConfig(**c)
                cfg.validate()
                gt = combine_truths(truths, ids, 'cuda:0')
                calls['count'] = 0
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                start = time.perf_counter()
                with (folder/'run.log').open('w') as log,redirect_stdout(log),redirect_stderr(log):
                    result = runner.run_single_ablation(cfg,checkpoint_bundle=bundle,ground_truth=gt)
                torch.cuda.synchronize()
                seconds = time.perf_counter()-start
                result_path = Path(result['run_dir'])/'result.pt'
                payload = torch.load(result_path,map_location='cpu',weights_only=False)
                # Independently recompute from saved physical tensors, retaining both fields.
                errors = {f:physical_errors(payload[k],payload[truth_key])
                          for f,k,truth_key in [('a','coef_final','coef_ground_truth'),('u','sol_final','sol_ground_truth')]}
                finite = all(math.isfinite(x) for values in errors.values() for x in values)
                errors = {f:[x if math.isfinite(x) else None for x in values] for f,values in errors.items()}
                masks = torch.load(Path(result['run_dir'])/'masks.pt',map_location='cpu',weights_only=False)
                mask_hash = hashlib.sha256(masks['coef'].numpy().tobytes()+masks['sol'].numpy().tobytes()).hexdigest()
                row = dict(protocol_sha256=ph, stage=stage, label=label, sample_ids=ids, config=c,
                    result_path=str(result_path), result_sha256=digest(result_path),
                    mask_sha256=mask_hash, errors=errors, finite=finite, status=result['status'],
                    seconds=seconds, nfe=calls['count'], peak_bytes=torch.cuda.max_memory_allocated())
                write(receipt,row)
                print('DONE',pde,stage,label,f'{seconds:.2f}s',errors,flush=True)
                return row

            def anchor(task='both'):
                rows=[x['config'] for x in protocol['archived'] if x['config']['task']==task and
                      x['config']['ablation_group']=='guidance_components' and x['config']['guidance_components']=='obs_pde']
                assert len(rows)==1,(pde,task,len(rows))
                return copy.deepcopy(rows[0])

            def proposal(task):
                c = anchor(task)
                r = next((r for r in protocol['recommendations'] if r['task']==task), None)
                if r:
                    # The recommendation already reverts failed holdout candidates.
                    c.update({k:r[k] for k in PARAM_KEYS})
                return c

            if args.mode == 'diagnose':
                # Batch one first: no unmeasured batch-size extrapolation on a 24GB GPU.
                rows=[]
                tasks = list(protocol['base_configs'])
                for task in tasks:
                    base = anchor(task)
                    candidates=[('archived',base)]
                    if pde in EXTRA:
                        candidates.append(('recommended',proposal(task)))
                    if pde in ['poisson','nsnonbounded']:
                        candidates += [(f'clip{clip:g}',dict(base,clip_threshold=clip))
                                       for clip in protocol['clipping_candidates']]
                    counts=protocol['tuning_observation_counts'] if task=='both' else [500]
                    for count in counts:
                        for name,c in candidates:
                            for i in protocol['tuning_ids']:
                                c=copy.deepcopy(c)
                                c.update(num_obs=count,sample_seed=20260912+i,mask_seed=20260912+i)
                                rows.append(one(f'{task}/{name}/obs{count}/sample{i}',c,[i],'diagnostic'))
                write(target/'diagnose_complete.json',dict(protocol_sha256=ph,runs=len(rows)))
            else:
                selection = json.loads((target/'selection.json').read_text())
                assert selection['protocol_sha256']==ph
                if args.mode == 'rerun':
                    for item in protocol['archived']:
                        c = copy.deepcopy(item['config'])
                        if c['ablation_group'] in EXCLUDED_GROUPS:
                            continue
                        c.update(selection['task_updates'][c['task']])
                        one(item['id'],c,[0],'main')
                elif args.mode == 'ensemble':
                    c=anchor()
                    c.update(selection['task_updates']['both'])
                    for seed in protocol['inference_seeds']:
                        for i in protocol['evaluation_ids']:
                            c.update(sample_seed=20260912+10000*seed+i,mask_seed=20260912+i)
                            one(f'seed{seed}/sample{i}',c,[i],'ensemble')
                write(target/f'{args.mode}_complete.json',dict(protocol_sha256=ph,completed_unix=time.time()))
            handle.remove()
            del bundle,truths
            torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','diagnose','rerun','ensemble'])
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--archive',type=Path)
    parser.add_argument('--pdes',nargs='+',choices=PDES,default=PDES)
    add_model_arguments(parser)
    args=parser.parse_args()
    if print_model_plan(args): return
    args.inputs=args.inputs.resolve()
    if args.mode=='prepare':
        if not args.archive: parser.error('--archive required for preparation')
        prepare(args)
    else:
        if not args.output: parser.error('--output required for sampling')
        args.output=args.output.resolve()
        run(args)


if __name__=='__main__':
    main()
