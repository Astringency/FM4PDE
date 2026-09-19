"""Matched optimizer continuation probes, with layerwise measured parameter updates.

The screening pool is a subset of the original TRAIN shard. It preserves the
original 50,000-example split, and cannot establish a global model/data limit.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import time

import numpy as np
import torch

from data.load import PDEloader
from data.specs import get_pde_spec
from data.training_manifest import load_training_file_manifest
from data.transform import PDEStandardizer
from models.model_configs import instantiate_model
from train import _train_val_split_indices
from training.load_and_save import _load_optimizer_state_preserving_runtime_options

PDES = ("poisson", "helmholtz", "darcy", "nsnonbounded", "burger")
ROOT = Path(__file__).resolve().parents[2]


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tensor_sha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def cpu_tree(x):
    if torch.is_tensor(x):
        return x.detach().cpu()
    if isinstance(x, dict):
        return {k: cpu_tree(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(cpu_tree(v) for v in x)
    return x


def select_source(root, pde):
    paths = sorted((root / "formal" / pde).glob("*/fm4*-checkpoint.pth"))
    date = "260904" if pde == "nsnonbounded" else "260621" if pde == "poisson" else "260625"
    matches = [p for p in paths if p.parent.name.startswith(date)]
    if len(matches) != 1:
        raise RuntimeError(f"Ambiguous source for {pde}: {matches}")
    return matches[0]


def prepare(args):
    out = Path(args.output).resolve() / args.pde
    out.mkdir(parents=True, exist_ok=True)
    if (out / "prepared.json").exists():
        print("ALREADY_PREPARED", args.pde, flush=True)
        return
    torch.set_num_threads(4)
    ckpt = select_source(Path(args.pretrained_root), args.pde)
    source = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
    assert source.get("optimizer") and not source.get("use_ema")
    saved_args = source["args"] if isinstance(source["args"], dict) else vars(source["args"])
    seed = int(saved_args.get("seed", 0))
    normalizer = PDEStandardizer.from_state_dict(source["normalizer"])
    files = load_training_file_manifest(ROOT / "configs/training_data.yaml",
        data_root=args.data_root, pde_names=[args.pde])[args.pde]
    loader = PDEloader(args.pde)
    raw, _ = loader.load_data_files([files[0]])
    assert len(raw) == 10000 and "test" not in str(files[0]).lower()
    train_ids, val_ids = _train_val_split_indices(50000,
        seed=seed + get_pde_spec(args.pde).label_id * 1009, val_ratio=0.1)
    train_ids = train_ids[train_ids < 10000]
    val_ids = val_ids[val_ids < 10000]
    generator = torch.Generator().manual_seed(20260919)
    train_ids = train_ids[torch.randperm(len(train_ids), generator=generator)[:4096]]
    val_ids = val_ids[torch.randperm(len(val_ids), generator=generator)]
    dev_ids, confirm_ids = val_ids[:256], val_ids[256:768]
    assert len(confirm_ids) == 512 and not set(train_ids.tolist()) & set(val_ids.tolist())
    data = {"train": normalizer.transform(raw[train_ids]).contiguous(),
        "development": normalizer.transform(raw[dev_ids]).contiguous(),
        "confirmation": normalizer.transform(raw[confirm_ids]).contiguous(),
        "train_ids": train_ids, "development_ids": dev_ids, "confirmation_ids": confirm_ids,
        "physical_development": raw[dev_ids[:32]].contiguous()}
    torch.save(data, out / "data.pt")
    # Retain all restart information, deduplicating raw model state for transfer.
    source["model"] = source["model_for_resume"]
    torch.save(source, out / "source.pth")
    groups = [{k: v for k, v in g.items() if k != "params"}
        for g in source["optimizer"]["param_groups"]]
    record = dict(pde=args.pde, source_checkpoint=str(ckpt), source_sha256=sha(ckpt),
        source_epoch=int(source["epoch"]), source_optimizer_groups=groups,
        optimizer_steps=sorted({int(s["step"]) for s in source["optimizer"]["state"].values()}),
        source_file=str(files[0]), source_file_sha256=sha(files[0]), source_file_size=files[0].stat().st_size,
        data_sha256=sha(out / "data.pt"), checkpoint_sha256=sha(out / "source.pth"),
        tensor_sha256={k: tensor_sha(v) for k, v in data.items()},
        sample_counts={k: len(data[k]) for k in ("train", "development", "confirmation")},
        split_policy="Original 50k deterministic split restricted to first training shard; no test samples",
        historical_caveat="June checkpoints may have used the full pool in earlier pretraining; this continuation is disjoint",
        checkpoint_normalizer_retained=True, checkpoint_architecture_retained=True,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    write(out / "prepared.json", record)
    print("PREPARED", args.pde, json.dumps(record["sample_counts"]), flush=True)


def restore(model, optimizer, source, lr=None, betas=None):
    model.load_state_dict(source["model_for_resume"], strict=True)
    # Deepcopy avoids CPU state aliasing too; every candidate starts identically.
    _load_optimizer_state_preserving_runtime_options(optimizer, copy.deepcopy(source["optimizer"]))
    for group in optimizer.param_groups:
        if lr is not None:
            group["lr"] = group["initial_lr"] = float(lr)
        if betas is not None:
            group["betas"] = tuple(betas)
    optimizer.zero_grad(set_to_none=True)


def draw_batch(data, ids, generator, device):
    sample = data[ids].to(device)
    noise = torch.randn(sample.shape, generator=generator).to(device)
    t = torch.rand(len(sample), generator=generator).clamp(1e-5, 1-1e-5).to(device)
    return noise.lerp(sample, t[:, None, None, None]), t, sample - noise


def backward_batch(model, xt, t, target, microbatch):
    total = 0.0
    for start in range(0, len(xt), microbatch):
        end = min(start + microbatch, len(xt))
        error = model(xt[start:end], t[start:end], extra={}) - target[start:end]
        loss = error.square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite loss")
        weight = (end-start) / len(xt)
        (weight * loss).backward()
        total += float(loss.detach()) * weight
    return total


def layer_stats(model, optimizer, before):
    records = []
    for name, p in model.named_parameters():
        g = p.grad
        if g is None:
            records.append(dict(name=name, elements=p.numel(), grad_missing=True))
            continue
        old = before[name]
        delta = p.detach() - old
        state = optimizer.state[p]
        gn, wn, dn = g.norm(), old.norm(), delta.norm()
        record = dict(name=name, elements=p.numel(), grad_missing=False,
            grad_norm=float(gn), weight_norm=float(wn), update_norm=float(dn),
            grad_zero_fraction=float((g == 0).float().mean()),
            update_zero_fraction=float((delta == 0).float().mean()),
            grad_finite=bool(torch.isfinite(g).all()),
            relative_update=float(dn / wn.clamp_min(1e-30)),
            descent_alignment=float(-(g * delta).sum() / (gn * dn).clamp_min(1e-30)),
            exp_avg_norm=float(state["exp_avg"].norm()),
            second_moment_mean=float(state["exp_avg_sq"].mean()),
            current_grad_square_mean=float(g.square().mean()))
        records.append(record)
    valid = [r for r in records if not r["grad_missing"]]
    elements = sum(r["elements"] for r in valid)
    gn = math.sqrt(sum(r["grad_norm"]**2 for r in valid))
    wn = math.sqrt(sum(r["weight_norm"]**2 for r in valid))
    dn = math.sqrt(sum(r["update_norm"]**2 for r in valid))
    summary = dict(grad_norm=gn, weight_norm=wn, update_norm=dn, relative_update=dn/max(wn,1e-30),
        missing_grad_tensors=len(records)-len(valid),
        grad_zero_fraction=sum(r["elements"]*r["grad_zero_fraction"] for r in valid)/elements,
        update_zero_fraction=sum(r["elements"]*r["update_zero_fraction"] for r in valid)/elements,
        descent_alignment=sum(r["descent_alignment"]*r["grad_norm"]*r["update_norm"] for r in valid)/max(gn*dn,1e-30),
        moment_to_current_g2=sum(r["elements"]*r["second_moment_mean"] for r in valid)/max(sum(r["elements"]*r["current_grad_square_mean"] for r in valid),1e-30))
    return dict(summary=summary, layers=records)


@torch.no_grad()
def evaluate(model, data, microbatch, seed=190900, repeats=4):
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    values, by_time = [], []
    # Independent stream; validation does not alter the training/dropout stream.
    with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
        for rep in range(repeats):
            gen = torch.Generator().manual_seed(seed + rep)
            t = (torch.rand(len(data), generator=gen) + rep) / repeats
            errors = []
            for start in range(0, len(data), microbatch):
                sample = data[start:start+microbatch].to(device)
                noise = torch.randn(sample.shape, generator=gen).to(device)
                tb = t[start:start+len(sample)].to(device)
                xt = noise.lerp(sample, tb[:,None,None,None])
                error = model(xt, tb, extra={}) - (sample-noise)
                errors.append(error.square().mean((2,3)).cpu())
            loss = torch.cat(errors)
            values.append(loss)
            by_time.append(float(loss.mean()))
    model.train(was_training)
    per_input_channel = torch.stack(values).mean(0)
    if not torch.isfinite(per_input_channel).all():
        raise FloatingPointError("Nonfinite validation")
    return dict(mean=float(per_input_channel.mean()), by_time_quartile=by_time,
        by_channel=per_input_channel.mean(0).tolist(), per_input=per_input_channel.mean(1).tolist(),
        seed=seed, repeats=repeats)


def paired(candidate, baseline):
    diff = np.asarray(candidate["per_input"]) - np.asarray(baseline["per_input"])
    se = float(diff.std(ddof=1)/math.sqrt(len(diff)))
    return dict(change=float(diff.mean()), ci95=[float(diff.mean()-1.96*se),float(diff.mean()+1.96*se)],
        relative_change_pct=100*(candidate["mean"]/baseline["mean"]-1), count=len(diff))


def frozen_gradients(model, optimizer, data, microbatch, draws=8):
    """Estimate minibatch gradient signal/noise without any parameter update."""
    model.train()
    device=next(model.parameters()).device
    sums={n:torch.zeros_like(p) for n,p in model.named_parameters()}
    square_sums={n:0.0 for n,_ in model.named_parameters()}
    per_batch=[]
    gen=torch.Generator().manual_seed(61919)
    with torch.random.fork_rng(devices=[device.index or 0]):
        torch.manual_seed(61919)
        for batch_index in range(draws):
            optimizer.zero_grad(set_to_none=True)
            ids=torch.randperm(len(data),generator=gen)[:64]
            xt,t,target=draw_batch(data,ids,gen,device)
            loss=backward_batch(model,xt,t,target,microbatch)
            norm2=0.
            missing=[]
            for name,p in model.named_parameters():
                if p.grad is None:
                    missing.append(name)
                    continue
                sums[name].add_(p.grad)
                g2=float(p.grad.square().sum())
                square_sums[name]+=g2
                norm2+=g2
            per_batch.append(dict(loss=loss,grad_norm=math.sqrt(norm2),missing_grad_names=missing))
            del xt,t,target
    layers=[]
    dot_total=0.
    moment_norm2=0.
    for name,p in model.named_parameters():
        s2=float(sums[name].square().sum())
        expected2=square_sums[name]/draws
        mean2=s2/draws**2
        signal2=(s2-square_sums[name])/(draws*(draws-1))
        variance=(expected2-mean2)*draws/(draws-1)
        m=optimizer.state[p].get("exp_avg",torch.zeros_like(p))
        dot=float((sums[name]*m).sum())/draws
        m2=float(m.square().sum())
        dot_total+=dot
        moment_norm2+=m2
        layers.append(dict(name=name,mean_gradient_norm=math.sqrt(max(mean2,0)),
            mean_batch_gradient_norm_squared=expected2,unbiased_signal_squared=signal2,
            noise_trace=variance,signal_to_noise=max(signal2,0)/max(variance,1e-30),
            mean_gradient_momentum_cosine=dot/max(math.sqrt(max(mean2*m2,0)),1e-30)))
    mean_norm2=sum(r["mean_gradient_norm"]**2 for r in layers)
    signal=sum(r["unbiased_signal_squared"] for r in layers)
    noise=sum(r["noise_trace"] for r in layers)
    optimizer.zero_grad(set_to_none=True)
    return dict(draws=draws,batch_size=64,per_batch=per_batch,layers=layers,
        summary=dict(mean_gradient_norm=math.sqrt(mean_norm2),unbiased_signal_squared=signal,
            noise_trace=noise,signal_to_noise=max(signal,0)/max(noise,1e-30),
            mean_gradient_momentum_cosine=dot_total/max(math.sqrt(mean_norm2*moment_norm2),1e-30)),
        caveat="Eight independent training-mode minibatches, including dropout and time/noise variation; noisy local estimate, not proof of global convergence")


def probe_batch(model, optimizer, source, data, maximum_gib, out):
    device = next(model.parameters()).device
    records=[]
    for batch in (1,2,4,8,16,32):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated()
        free, _ = torch.cuda.mem_get_info()
        budget = min(maximum_gib*2**30, base + free - 12*2**30)
        if records:
            prior=records[-1]
            projected=base+2.5*(prior["peak_allocated"]-prior["base"])+2*2**30
            if projected > budget:
                break
        elif free < 12*2**30:
            raise RuntimeError("Insufficient memory for conservative batch probe")
        torch.cuda.reset_peak_memory_stats()
        gen=torch.Generator().manual_seed(919)
        xt,t,target=draw_batch(data, torch.arange(batch), gen, device)
        begin=time.monotonic()
        backward_batch(model,xt,t,target,batch)
        optimizer.step()
        torch.cuda.synchronize()
        elapsed=time.monotonic()-begin
        records.append(dict(batch=batch,base=base,peak_allocated=torch.cuda.max_memory_allocated(),
            peak_reserved=torch.cuda.max_memory_reserved(),seconds=elapsed,samples_per_second=batch/elapsed))
        write(out/"batch_probe.json",records)
        del xt,t,target
    eligible=[r for r in records if r["peak_allocated"] < maximum_gib*2**30]
    if not eligible:
        raise RuntimeError("No batch fits the memory limit")
    # These are one-step probes; retain conservative largest fitting batch.
    return max(r["batch"] for r in eligible)


def save_resume(path, source, model, optimizer, config, steps, prepared):
    payload={k:v for k,v in source.items() if k not in ("model","model_for_resume","model_ema","optimizer","lr_schedule","scaler")}
    state=cpu_tree(model.state_dict())
    args=copy.copy(source["args"] if isinstance(source["args"],dict) else vars(source["args"]))
    group=optimizer.param_groups[0]
    args.update(lr=group["lr"],optimizer_betas=list(group["betas"]),lr_scheduler="constant",warmup_epochs=0)
    scheduler=torch.optim.lr_scheduler.ConstantLR(optimizer,factor=1,total_iters=1)
    payload.update(model=state,model_for_resume=state,model_ema=None,optimizer=cpu_tree(optimizer.state_dict()),
        lr_schedule=scheduler.state_dict(),scaler=None,args=argparse.Namespace(**args),
        lr_scheduler="constant",resolved_lr_scheduler="constant",
        optimizer_diagnostics=dict(config=config,additional_updates=steps,source_sha256=prepared["source_sha256"],
            epoch_policy="Partial-data update study; original complete-epoch index retained",exact_restart_entry="experiments.optimizer_diagnostics.study"))
    temporary=path.with_suffix(".tmp")
    torch.save(payload,temporary)
    temporary.replace(path)


def run(args):
    out=Path(args.output).resolve()/args.pde
    out.mkdir(parents=True,exist_ok=True)
    if (out/"complete.json").exists():
        raise RuntimeError("Completed study cannot be overwritten")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=False
    device=torch.device("cuda:0")
    input_dir=Path(args.inputs)/args.pde
    prepared=json.loads((input_dir/"prepared.json").read_text())
    assert sha(input_dir/"source.pth")==prepared["checkpoint_sha256"]
    assert sha(input_dir/"data.pt")==prepared["data_sha256"]
    source=torch.load(input_dir/"source.pth",map_location="cpu",weights_only=False,mmap=True)
    data=torch.load(input_dir/"data.pt",map_location="cpu",weights_only=False,mmap=True)
    model=instantiate_model(args.pde,use_ema=False,model_config=source["model_config"]).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,fused=True)
    restore(model,optimizer,source)
    source_group=source["optimizer"]["param_groups"][0]
    assert tuple(source_group["betas"])==(0.9,0.999)
    batch=probe_batch(model,optimizer,source,data["train"],args.max_memory_gib,out)
    restore(model,optimizer,source)
    write(out/"frozen_gradient_probe.json",frozen_gradients(model,optimizer,data["train"],batch))
    configs=[dict(name="source_lr",lr=float(source_group["lr"]),betas=[.9,.999]),
        dict(name="lr1e5",lr=1e-5,betas=[.9,.999]),
        dict(name="lr3e5",lr=3e-5,betas=[.9,.999]),
        dict(name="beta1_08",lr=1e-5,betas=[.8,.999]),
        dict(name="beta2_099",lr=1e-5,betas=[.9,.99])]
    write(out/"protocol.json",dict(prepared=prepared,configs=configs,steps=args.steps,effective_batch=64,
        microbatch=batch,host=socket.gethostname(),pid=os.getpid(),gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,precision="FP32 with TF32",moment_policy="all original moments and counters retained before every candidate; betas override applied after loading",
        scheduler_policy="constant, explicitly set after loading; one-factor comparisons at lr1e-5",
        git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()))
    baseline=evaluate(model,data["development"],batch)
    write(out/"baseline_development.json",baseline)
    baseline_confirm=evaluate(model,data["confirmation"],batch,seed=290900)
    write(out/"baseline_confirmation.json",baseline_confirm)
    all_results=[]
    best=float("inf")
    for config in configs:
        restore(model,optimizer,source,lr=config["lr"],betas=config["betas"])
        assert optimizer.param_groups[0]["lr"]==config["lr"]
        assert list(optimizer.param_groups[0]["betas"])==config["betas"]
        torch.manual_seed(20260919)
        generator=torch.Generator().manual_seed(20260919)
        order_generator=torch.Generator().manual_seed(3919)
        order=[]
        while len(order)<args.steps*64:
            order.extend(torch.randperm(len(data["train"]),generator=order_generator).tolist())
        model.train()
        records=[]
        started=time.monotonic()
        for step in range(1,args.steps+1):
            optimizer.zero_grad(set_to_none=True)
            xt,t,target=draw_batch(data["train"],order[(step-1)*64:step*64],generator,device)
            loss=backward_batch(model,xt,t,target,batch)
            grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),float("inf"),error_if_nonfinite=True)
            diagnose=step in {1,8,32,64,128,args.steps}
            before={n:p.detach().clone() for n,p in model.named_parameters()} if diagnose else None
            optimizer.step()
            record=dict(step=step,loss=loss,grad_norm=float(grad_norm),lr=optimizer.param_groups[0]["lr"],
                betas=list(optimizer.param_groups[0]["betas"]),elapsed_seconds=time.monotonic()-started)
            if diagnose:
                diagnostic=layer_stats(model,optimizer,before)
                record.update(diagnostic["summary"])
                write(out/config["name"]/f"layers_step_{step:04d}.json",diagnostic)
                del before
            records.append(record)
            if step % 16==0 or step==1:
                write(out/"progress.json",dict(pde=args.pde,candidate=config["name"],**record))
                print("PROGRESS",args.pde,config["name"],step,f"loss={loss:.7f}",flush=True)
            del xt,t,target
        validation=evaluate(model,data["development"],batch)
        result=dict(config=config,steps=args.steps,seconds=time.monotonic()-started,
            training=records,development=validation,paired_development=paired(validation,baseline))
        write(out/config["name"]/"result.json",result)
        all_results.append(result)
        if config["name"]=="lr1e5":
            save_resume(out/"lr_control.pth",source,model,optimizer,config,args.steps,prepared)
        if config["name"]!="source_lr" and validation["mean"]<best:
            best=validation["mean"]
            save_resume(out/"selected_resume.pth",source,model,optimizer,config,args.steps,prepared)
        print("CANDIDATE_DONE",args.pde,config["name"],result["paired_development"],flush=True)
    # Confirmation is opened only after development selection is complete.
    confirmations={}
    for name in ("lr_control","selected_resume"):
        candidate=torch.load(out/f"{name}.pth",map_location="cpu",weights_only=False,mmap=True)
        model.load_state_dict(candidate["model_for_resume"],strict=True)
        val=evaluate(model,data["confirmation"],batch,seed=290900)
        confirmations[name]=dict(metrics=val,paired_original=paired(val,baseline_confirm),
            config=candidate["optimizer_diagnostics"]["config"],sha256=sha(out/f"{name}.pth"))
        del candidate
    confirmations["selected_vs_lr_control"]=paired(confirmations["selected_resume"]["metrics"],confirmations["lr_control"]["metrics"])
    write(out/"complete.json",dict(status="completed_optimizer_screen",pde=args.pde,
        confirmations=confirmations,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        interpretation="A matched partial-data continuation screen, not evidence of a global optimization limit or downstream sampling improvement"))
    print("COMPLETE",args.pde,json.dumps(confirmations["selected_vs_lr_control"]),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=["prepare","run"])
    parser.add_argument("--pde",choices=PDES,required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--inputs")
    parser.add_argument("--pretrained-root")
    parser.add_argument("--data-root",default="/large_storage/zhangxf/PDEdata")
    parser.add_argument("--steps",type=int,default=128)
    parser.add_argument("--max-memory-gib",type=float,default=26)
    args=parser.parse_args()
    if not Path(args.output).is_absolute():
        parser.error("Use an explicit absolute output directory")
    (prepare if args.mode=="prepare" else run)(args)


if __name__=="__main__":
    main()
