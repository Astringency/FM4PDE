"""Recheck optimizer state, paired statistics, and matched initial gradients."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.optimizer_diagnostics.study import PDES, paired, sha, write


def audit(root,pde):
    run=root/"runs"/pde
    prepared=json.loads((root/"inputs"/pde/"prepared.json").read_text())
    protocol=json.loads((run/"protocol.json").read_text())
    complete=json.loads((run/"complete.json").read_text())
    exit_record=json.loads((run/"run.exit.json").read_text())
    assert exit_record["exit_code"]==0
    reference=torch.load(root/"inputs"/pde/"source.pth",map_location="cpu",weights_only=False,mmap=True)
    initial=reference["model_for_resume"]
    source_steps={int(x["step"]) for x in reference["optimizer"]["state"].values()}
    assert len(source_steps)==1
    original_step=next(iter(source_steps))
    baseline=json.loads((run/"baseline_development.json").read_text())
    diagnostics=[]
    for config in protocol["configs"]:
        result=json.loads((run/config["name"]/"result.json").read_text())
        assert result["steps"]==protocol["steps"]==len(result["training"])
        expected=paired(result["development"],baseline)
        np.testing.assert_allclose(expected["ci95"],result["paired_development"]["ci95"],rtol=1e-12,atol=1e-12)
        rows=result["training"]
        assert [x["step"] for x in rows]==list(range(1,protocol["steps"]+1))
        assert all(x["lr"]==config["lr"] and x["betas"]==config["betas"] for x in rows)
        diagnostics.append(json.loads((run/config["name"]/"layers_step_0001.json").read_text()))
    # Every knob comparison starts at identical weights, minibatch, time/noise and dropout.
    reference_grads=np.asarray([x["grad_norm"] for x in diagnostics[0]["layers"]])
    for record in diagnostics[1:]:
        np.testing.assert_allclose(reference_grads,[x["grad_norm"] for x in record["layers"]],rtol=2e-6,atol=1e-10)
    checkpoints=[]
    for name in ("lr_control","selected_resume"):
        path=run/f"{name}.pth"
        payload=torch.load(path,map_location="cpu",weights_only=False,mmap=True)
        assert sha(path)==complete["confirmations"][name]["sha256"]
        config=payload["optimizer_diagnostics"]["config"]
        states=payload["optimizer"]["state"]
        steps={int(x["step"]) for x in states.values()}
        assert steps=={original_step+protocol["steps"]}
        assert payload["model_config"]==reference["model_config"]
        for k,value in reference["normalizer"].items():
            actual=payload["normalizer"][k]
            assert torch.equal(value,actual) if torch.is_tensor(value) else value==actual
        for g in payload["optimizer"]["param_groups"]:
            assert g["lr"]==config["lr"] and list(g["betas"])==config["betas"]
        changed=0
        delta2=0.
        weight2=0.
        for k,value in payload["model_for_resume"].items():
            assert torch.isfinite(value).all()
            changed+=int(not torch.equal(value,initial[k]))
            if value.is_floating_point():
                delta2+=float((value.double()-initial[k].double()).square().sum())
                weight2+=float(initial[k].double().square().sum())
        assert changed>0
        for state in states.values():
            assert torch.isfinite(state["exp_avg"]).all()
            assert torch.isfinite(state["exp_avg_sq"]).all()
            assert (state["exp_avg_sq"]>=0).all()
        checkpoints.append(dict(name=name,config=config,optimizer_step=next(iter(steps)),
            optimizer_state_count=len(states),changed_weight_tensors=changed,relative_weight_change=(delta2/weight2)**.5,
            sha256=sha(path),normalizer_and_architecture_preserved=True,weights_and_moments_finite=True))
    record=dict(pde=pde,status="verified",initial_gradients_matched=True,actual_hyperparameters_verified=True,
        candidate_update_count=protocol["steps"],source_optimizer_step=original_step,checkpoints=checkpoints,
        original_source_sha256_unchanged=sha(prepared["source_checkpoint"])==prepared["source_sha256"])
    assert record["original_source_sha256_unchanged"]
    write(run/"audit.json",record)
    print(json.dumps(record),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--pdes",nargs="+",default=list(PDES))
    args=parser.parse_args()
    torch.set_num_threads(4)
    for pde in args.pdes:
        audit(args.root,pde)


if __name__=="__main__":
    main()
