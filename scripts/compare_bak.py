#!/usr/bin/env python3
"""Paired legacy evaluation against archived outputs/main, without raw MAT files.

Sample IDs are chosen independently of errors. Prefer each complete tuned1
1000-sample group, otherwise use the complete original group. Only stochastic
100-step sparse results from the original formal checkpoints are eligible.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import time

PDES = ("poisson", "helmholtz", "darcy", "nsnonbounded")
BURGER_SENSORS = ("random", "sensor_column")


def suite_pdes(suite):
    return ("burger",) if suite == "burgers" else PDES


def cell_key(row):
    key = tuple(row[k] for k in ("pde", "test_type", "task"))
    return (*key, row["sensor_mode"]) if row["pde"] == "burger" else key


def result_key(row):
    return (*cell_key(row), row["sample_id"])


def is_priority(row):
    return row["test_type"] == "rough" and (row["task"] == "inverse" or row["pde"] == "burger")


def worker_for_entry(entry, num_workers):
    # Burgers is one PDE: split the two observation layouts across the GPUs.
    index = BURGER_SENSORS.index(entry["sensor_mode"]) if entry["pde"] == "burger" else PDES.index(entry["pde"])
    return index % num_workers


def write_worker_status(args, *, total, completed, status, label=""):
    path = args.output / f"worker{args.worker_index}.status.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(dict(pid=os.getpid(), device=args.device, total=total,
                                        completed=completed, status=status, label=label)))
    temporary.replace(path)


def archive_path(value):
    path = Path(value)
    if path.exists():
        return path
    parts = path.parts
    if parts[:1] == ("outputs",) and parts[1].startswith("MAIN"):
        path = Path("outputs/main", *parts[1:])
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def select_groups(root, suite="four_pde"):
    selected = {}
    for distribution in ("rough", "id", "smooth"):
        for suffix in ("", "_tuned1"):
            source = root / f"MAIN1000_100_TEST_{distribution}{suffix}" / "metrics_per_sample_all.csv"
            if not source.exists():
                continue
            groups = defaultdict(list)
            for row in csv.DictReader(source.open()):
                tasks = {"both"} if suite == "burgers" else {"both", "forward", "inverse"}
                if row["pde"] in suite_pdes(suite) and row["task"] in tasks:
                    groups[(row["pde"], row["task"], row["ablation_group_key"])].append(row)
            eligible = defaultdict(list)
            for (pde, task, _), rows in groups.items():
                if len(rows) == 1000 and {int(r["sample_id"]) for r in rows} == set(range(1000)):
                    if all(int(r["num_obs"]) == 500 and int(r["num_steps"]) == 100
                           and r["sampler_phase"] == "stochastic" for r in rows):
                        key = (pde, distribution, task)
                        if pde == "burger":
                            modes = {r["sensor_mode"] for r in rows}
                            if len(modes) != 1 or not modes <= set(BURGER_SENSORS):
                                continue
                            mode = next(iter(modes))
                            if mode == "sensor_column":
                                columns = {int(r["num_sensor_columns"]) for r in rows}
                                if len(columns) != 1 or min(columns) <= 0:
                                    raise ValueError(f"Inconsistent Burgers sensor columns: {source}")
                            key = (*key, mode)
                        eligible[key].append(rows)
            for key, groups_for_key in eligible.items():
                if len(groups_for_key) != 1:
                    raise ValueError(f"Ambiguous complete baseline groups for {key}: {source}")
                selected[key] = (source, groups_for_key[0])
    expected = 6 if suite == "burgers" else 36
    if len(selected) != expected:
        raise ValueError(f"Expected all {expected} {suite} comparison groups, found {len(selected)}")
    return selected


def make_manifest(root, samples, priority_samples, seed, suite="four_pde"):
    import numpy as np
    selected = select_groups(root, suite)
    rng = np.random.default_rng(seed)
    # Same prespecified sample ID order in every cell; no selection by baseline errors.
    ids = rng.permutation(1000).tolist()
    manifest = []
    for key, (source, rows) in selected.items():
        pde, distribution, task = key[:3]
        by_id = {int(row["sample_id"]): row for row in rows}
        count = priority_samples if distribution == "rough" and (task == "inverse" or pde == "burger") else samples
        for order, sample_id in enumerate(ids[:count]):
            row = by_id[sample_id]
            entry = dict(pde=pde, test_type=distribution, task=task, sample_id=sample_id,
                         selection_order=order, baseline_csv=str(source), baseline=row)
            if pde == "burger":
                entry.update(sensor_mode=key[3], num_sensor_columns=(
                    int(row["num_sensor_columns"]) if key[3] == "sensor_column" else None))
            manifest.append(entry)
    return sorted(manifest, key=lambda r: (
        0 if is_priority(r) else 1,
        suite_pdes(suite).index(r["pde"]), ("rough", "id", "smooth").index(r["test_type"]),
        ("inverse", "both", "forward").index(r["task"]), r.get("sensor_mode", ""), r["selection_order"],
    ))


def _slice_params(value, indices, batch_size, device):
    import torch
    if torch.is_tensor(value):
        if value.ndim and value.shape[0] == batch_size:
            value = value[indices]
        return value.to(device)
    if isinstance(value, dict):
        return {k: _slice_params(v, indices, batch_size, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_slice_params(v, indices, batch_size, device) for v in value]
    return value


def load_archived_sample(entry, device):
    gt, masks, config, indices, batch_size = load_archived_batch([entry], device)
    return gt, masks, config, indices[0], batch_size


def load_archived_batch(entries, device):
    import torch
    from sampling.data import PDEGroundTruth
    from sampling.masks import PairMasks
    entry = entries[0]
    folder = archive_path(entry["baseline"]["run_dir"])
    if any(e['baseline']['run_dir'] != entry['baseline']['run_dir'] for e in entries):
        raise ValueError("A replay batch must come from the same archived run")
    payload = torch.load(folder / "result.pt", weights_only=False, mmap=True, map_location="cpu")
    config = payload["config"]
    expected_settings = dict(pde=entry['pde'], task=entry['task'], num_steps=100, num_obs=500,
                             sampler_phase='stochastic')
    if entry['pde'] == 'burger':
        expected_settings['sensor_mode'] = entry['sensor_mode']
        if entry['sensor_mode'] == 'sensor_column':
            expected_settings['num_sensor_columns'] = entry['num_sensor_columns']
    if any(config.get(key) != value for key,value in expected_settings.items()):
        raise ValueError(f'Archived configuration does not match the selected comparison cell: {folder}')
    expected_checkpoint = {
        "poisson": "260621-104344", "helmholtz": "260625-150653",
        "darcy": "260625-150746", "nsnonbounded": "260712-121023",
        "burger": "260625-150439",
    }[entry["pde"]]
    if expected_checkpoint not in config["checkpoint_path"]:
        raise ValueError(f"Baseline is not the original 50k-data formal checkpoint: {config['checkpoint_path']}")
    indices = [int(e["baseline"]["sample_index"]) for e in entries]
    batch_size = int(payload["coef_ground_truth"].shape[0])
    metadata = dict(payload["ground_truth_metadata"])
    if any(str(metadata["sample_ids"][i]) != str(e["sample_id"]) for i,e in zip(indices,entries)):
        raise ValueError("Archive sample identity does not match the summary row")
    a = payload["coef_ground_truth"][indices].to(device)
    u = payload["sol_ground_truth"][indices].to(device)
    metadata.update(batch_size=len(entries), sample_ids=[str(e["sample_id"]) for e in entries],
                    sample_offsets=[e["sample_id"] for e in entries], offset=entry["sample_id"],
                    comparison_source_artifact=str(folder / "result.pt"))
    if entry["pde"] == "burger" and not torch.equal(a, u):
        raise ValueError("Burgers archive must store the same single field in coef and sol")
    gt = PDEGroundTruth(entry["pde"], a, u, u if entry["pde"] == "burger" else torch.cat([a, u], 1),
                        _slice_params(payload.get("pde_params", {}), indices, batch_size, device),
                        metadata["channel_names_coef"], metadata["channel_names_sol"], metadata)
    saved_masks = payload["masks"]
    masks = PairMasks(saved_masks["coef"][indices].to(device),
                      saved_masks["sol"][indices].to(device), dict(saved_masks["metadata"]))
    if entry["pde"] == "burger":
        expected_count = entry["num_sensor_columns"] * u.shape[-2] if entry["sensor_mode"] == "sensor_column" else 500
        if not bool((masks.sol.flatten(1).sum(1) == expected_count).all()):
            raise ValueError("Burgers archive mask does not match its observation budget")
        if entry["sensor_mode"] == "sensor_column" and not torch.equal(masks.sol, masks.sol[..., :1, :].expand_as(masks.sol)):
            raise ValueError("Burgers sensor_column archive does not contain full columns")
    # Verify CSV metrics against the actual archived predictions before pairing.
    for side, target in (("a", a), ("u", u)):
        field = "coef_final" if side == "a" else "sol_final"
        prediction = payload[field][indices].to(device)
        error = torch.linalg.vector_norm((prediction-target).flatten(1),dim=1) / torch.linalg.vector_norm(target.flatten(1),dim=1)
        if any(abs(float(value) - float(e["baseline"][f"rel_l2_{side}"])) > 2e-5 for value,e in zip(error,entries)):
            raise ValueError("Baseline CSV and archived prediction error disagree")
    return gt, masks, config, indices, batch_size


def read_results(output):
    paths = [output / 'paired_results.jsonl', *sorted(output.glob('paired_results.worker*.jsonl'))]
    records = {}
    for path in paths:
        if not path.exists():continue
        for line in path.read_text().splitlines():
            row=json.loads(line)
            key=result_key(row)
            if key not in records or row['status']=='ok' or records[key]['status']!='ok':
                records[key]=row  # A successful retry supersedes failures even if worker count changes.
    return list(records.values())


def summarize(output):
    import numpy as np
    from scipy.stats import wilcoxon
    records = read_results(output)
    groups = defaultdict(list)
    for row in records:
        groups[cell_key(row)].append(row)
    burgers_only = bool(records) and all(row["pde"] == "burger" for row in records)
    family_size = 6 if burgers_only else 72
    if any(row["pde"] == "burger" for row in records) and not burgers_only:
        raise ValueError("Burgers and four-PDE results must use separate output directories")
    summaries = []
    rng = np.random.default_rng(20260905)
    for key, rows in sorted(groups.items()):
        ok = [r for r in rows if r["status"] == "ok"]
        # Burgers coef and sol are aliases of one predicted field, not two outcomes.
        for side in (("u",) if key[0] == "burger" else ("a", "u")):
            item = dict(zip(("pde", "test_type", "task"), key))
            if key[0] == "burger":
                item.update(sensor_mode=key[3], num_sensor_columns=rows[0]["num_sensor_columns"])
            item.update(metric=f"rel_l2_{side}", n=len(ok), failures=len(rows)-len(ok))
            if ok:
                current = np.array([r[f"current_{side}"] for r in ok])
                legacy = np.array([r[f"bak_{side}"] for r in ok])
                diff = legacy-current
                boot = diff[rng.integers(0, len(diff), size=(10000, len(diff)))].mean(axis=1)
                item.update(current_mean=float(current.mean()), bak_mean=float(legacy.mean()),
                            mean_difference=float(diff.mean()), ci95_low=float(np.quantile(boot,.025)),
                            ci95_high=float(np.quantile(boot,.975)), bak_win_rate=float((diff<0).mean()),
                            p_value=float(wilcoxon(diff).pvalue) if (diff!=0).any() and len(diff)>1 else 1.0)
            summaries.append(item)
    eligible = [r for r in summaries if "p_value" in r]
    # Include the full prespecified family, even while a run is incomplete.
    running = 0.
    for rank, row in enumerate(sorted(eligible,key=lambda r:r["p_value"])):
        running = max(running, min(1., row["p_value"]*(family_size-rank)))
        row[f"p_holm_{family_size}"] = running
        row["significant"] = running < .05 and row["failures"] == 0
    if summaries:
        columns = list(dict.fromkeys(k for row in summaries for k in row))
        with (output / "summary.csv").open("w",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=columns);writer.writeheader();writer.writerows(summaries)
    return summaries


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("four_pde", "burgers"), default="four_pde")
    parser.add_argument("--main-root",type=Path,default=Path("outputs/main"))
    parser.add_argument("--output",type=Path,default=Path("outputs/bak_comparison"))
    parser.add_argument("--samples",type=int,default=20)
    parser.add_argument("--priority-samples",type=int,default=100)
    parser.add_argument("--selection-seed",type=int,default=20260905)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--batch-size",type=int,default=1)
    parser.add_argument("--worker-index",type=int,default=0)
    parser.add_argument("--num-workers",type=int,default=1)
    parser.add_argument("--retry-failed",action="store_true")
    parser.add_argument("--pde",choices=(*PDES, "burger"))
    parser.add_argument("--sensor-mode",choices=BURGER_SENSORS,help="Filter Burgers observation layout")
    parser.add_argument("--priority-only",action="store_true")
    parser.add_argument("--limit",type=int)
    parser.add_argument("--steps",type=int,default=100,help="Use a separate output for short smoke runs")
    parser.add_argument("--prepare-only",action="store_true")
    parser.add_argument("--summarize-only",action="store_true")
    args=parser.parse_args(argv)
    if args.batch_size < 1 or not 0 <= args.worker_index < args.num_workers <= 4:
        raise ValueError("Require batch-size >= 1 and 0 <= worker-index < num-workers <= 4")
    if args.pde and args.pde not in suite_pdes(args.suite):
        raise ValueError("The selected PDE does not belong to --suite; use --suite burgers for burger")
    if args.suite == "burgers" and args.num_workers > 2:
        raise ValueError("Burgers supports one or two workers (one per observation layout)")
    if args.sensor_mode and args.suite != "burgers":
        raise ValueError("--sensor-mode is only supported by --suite burgers")
    args.output.mkdir(parents=True,exist_ok=True)
    if args.summarize_only:
        summarize(args.output);return
    if not 1 <= args.samples <=1000 or not 1 <= args.priority_samples <=1000:
        raise ValueError("Sample counts must be in 1..1000")
    manifest=make_manifest(args.main_root,args.samples,args.priority_samples,args.selection_seed,args.suite)
    manifest_path=args.output/"manifest.json"
    protocol=dict(steps=args.steps,samples=args.samples,priority_samples=args.priority_samples,
                  selection_seed=args.selection_seed,comparison="legacy weights + legacy guidance vs archived current system",
                  rng="historical seed and source batch row replay; historical RNG tensors were not saved",
                  uncertainty="paired bootstrap mean CI; Wilcoxon and Holm over 72 outcomes; failures retained")
    if args.suite == "burgers":
        protocol.update(suite="burgers", uncertainty="paired bootstrap mean CI; Wilcoxon and Holm over 6 u outcomes; failures retained",
                        observations="reuse archived random500 and sensor_column masks/counts; legacy loss denominators retained")
    protocol['bak_config_sha256']={str(path):hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in sorted(Path('configs/bak').glob('*/*.yaml'))
                                   if path.stem in suite_pdes(args.suite)}
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Existing output has a different sample manifest; use a new output directory")
    protocol_path=args.output/"protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Existing output has a different protocol; use a new output directory")
    if not manifest_path.exists():manifest_path.write_text(json.dumps(manifest,indent=2))
    if not protocol_path.exists():protocol_path.write_text(json.dumps(protocol,indent=2))
    if args.prepare_only:
        assets={str(entry['baseline_csv']) for entry in manifest}
        run_dirs={entry['baseline']['run_dir'] for entry in manifest}
        assets.update(str(archive_path(folder)/'result.pt') for folder in run_dirs)
        from sampling.config import load_config
        assets.update(load_config(f"configs/bak/{entry['task']}/{entry['pde']}.yaml").checkpoint_path
                      for entry in {cell_key(e): e for e in manifest}.values())
        missing=[path for path in sorted(assets) if not Path(path).is_file()]
        if missing:raise FileNotFoundError(f'Missing required benchmark assets: {missing[:5]}')
        (args.output/'required_assets.txt').write_text('\n'.join(sorted(assets))+'\n')
        print(f"Prepared {len(manifest)} paired samples across {len({cell_key(e) for e in manifest})} cells");return
    import torch
    from sampling.config import load_config
    from sampling.model_io import load_fm4pde_checkpoint_bundle
    from sampling.runner import run_single_ablation
    torch.set_num_threads(4)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA is unavailable; refusing a silent CPU benchmark")
    results_path=args.output/('paired_results.jsonl' if args.num_workers==1 else f'paired_results.worker{args.worker_index}.jsonl')
    completed=set()
    for row in read_results(args.output):
        if row['status']!='ok' and not args.retry_failed:
            raise ValueError('A recorded failure requires inspection; use --retry-failed after resolving it')
        if row['status']=='ok':completed.add(result_key(row))
    pending=defaultdict(list)
    already_done=0
    for entry in manifest:
        key=result_key(entry)
        if args.pde and entry['pde']!=args.pde:continue
        if worker_for_entry(entry,args.num_workers) != args.worker_index:continue
        if args.priority_only and not is_priority(entry):continue
        if args.sensor_mode and entry.get('sensor_mode') != args.sensor_mode:continue
        if key in completed:
            already_done+=1
            continue
        pending[(*cell_key(entry),entry['baseline']['run_dir'])].append(entry)
    jobs=[entries[start:start+args.batch_size] for entries in pending.values()
          for start in range(0,len(entries),args.batch_size)]
    pending_count=sum(len(entries) for entries in jobs)
    total=already_done+(min(pending_count,args.limit) if args.limit is not None else pending_count)
    progress_path=args.output/f'worker{args.worker_index}.progress.json'
    progress_path.unlink(missing_ok=True)
    os.environ['FM4PDE_SWEEP_PROGRESS_FILE']=str(progress_path)
    write_worker_status(args,total=total,completed=already_done,status='loading')
    bundle=None;loaded_pde=None;count=0
    for entries in jobs:
        if args.limit is not None and count>=args.limit:break
        if args.limit is not None:entries=entries[:args.limit-count]
        entry=entries[0]
        label='/'.join(cell_key(entry))
        write_worker_status(args,total=total,completed=already_done+count,status='sampling',label=label)
        cfg=load_config(f"configs/bak/{entry['task']}/{entry['pde']}.yaml")
        if loaded_pde!=entry['pde']:
            bundle=None;gc.collect();torch.cuda.empty_cache()
            bundle=load_fm4pde_checkpoint_bundle(cfg.checkpoint_path,cfg.pde,args.device,model_profile='legacy')
            loaded_pde=entry['pde']
        start=time.monotonic()
        results=[{k:e[k] for k in ('pde','test_type','task','sample_id')} for e in entries]
        if entry['pde'] == 'burger':
            for result,e in zip(results,entries):
                result.update(sensor_mode=e['sensor_mode'], num_sensor_columns=e['num_sensor_columns'])
        print(f"START {'/'.join(cell_key(entry))} ids={[e['sample_id'] for e in entries]}",flush=True)
        try:
            gt,masks,source,indices,batch_size=load_archived_batch(entries,args.device)
            cfg.test_type=entry['test_type'];cfg.offset=entry['sample_id'];cfg.sample_seed=int(source['sample_seed'])
            cfg.initial_noise_source_batch_size=batch_size;cfg.initial_noise_source_indices=indices
            cfg.device=args.device;cfg.num_steps=args.steps;cfg.batch_size=len(entries)
            if entry['pde'] == 'burger':
                cfg.sensor_mode=source['sensor_mode'];cfg.num_sensor_columns=source.get('num_sensor_columns')
            name=f"bak_{entry['test_type']}_{entry['task']}_{entry['sample_id']}"
            if entry['pde'] == 'burger':
                name+=f"_{entry['sensor_mode']}" + (f"{cfg.num_sensor_columns}" if cfg.sensor_mode=='sensor_column' else "500")
            cfg.output_dir=str(args.output/'runs');cfg.ablation_group=name;cfg.ablation_name=name
            final=run_single_ablation(cfg,bundle,ground_truth=gt,observation_masks=masks)
            sample_rows=list(csv.DictReader((Path(final['run_dir'])/'metrics_per_sample.csv').open()))
            if len(sample_rows) != len(entries):raise ValueError('Output sample count changed')
            for result, sample in zip(results,sample_rows):
                if str(result['sample_id']) != sample['sample_id']:raise ValueError('Output sample ID changed')
                result.update(status='ok',bak_a=float(sample['rel_l2_a']),bak_u=float(sample['rel_l2_u']),run_dir=final['run_dir'])
        except Exception as exc:
            import traceback
            traceback.print_exc()
            for result in results:result.update(status='failed',error=str(exc))
        for result,entry in zip(results,entries):
            result.update(current_a=float(entry['baseline']['rel_l2_a']),current_u=float(entry['baseline']['rel_l2_u']),
                          baseline_run=entry['baseline']['run_dir'],seconds=(time.monotonic()-start)/len(entries),steps=args.steps)
            with results_path.open('a') as handle:handle.write(json.dumps(result)+'\n')
            print('DONE '+json.dumps(result),flush=True)
        count+=len(entries)
        failed=any(result['status']!='ok' for result in results)
        write_worker_status(args,total=total,completed=already_done+count,
                            status='failed' if failed else 'sampling',label=label)
        if args.num_workers==1:summarize(args.output)
        if failed:
            raise RuntimeError('Comparison paused after a failed sample; inspect the recorded failure before resuming')
    write_worker_status(args,total=total,completed=already_done+count,status='complete')


if __name__=='__main__':main()
