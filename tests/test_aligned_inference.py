"""CPU-only kernel/recovery checks; real checkpoint/GPU pilots run separately."""
import dataclasses
import json
from pathlib import Path
import subprocess
import sys
import types

import pytest
import torch

from experiments.aligned_sampling.run_inference import (
    ROOT, digest, effective_config, existing_batches, infer,
    json_digest, observation_batch, pilot_family, tensor_digest, production, validate_protocol,
)


class Velocity(torch.nn.Module):
    def forward(self, x, t, extra=None):
        return .17 * x + .003 * torch.sin(t)


class Identity:
    def __init__(self, channels):
        self.mean = torch.zeros(1, channels, 1, 1)

    def inverse_transform(self, x):
        return x


def fixture(pde="poisson", clip="global_norm", task="both"):
    from sampling.config import AblationConfig
    rng = torch.Generator().manual_seed(11)
    a = .2 * torch.randn(4, 1, 8, 8, generator=rng)
    u = a if pde == "burger" else .2 * torch.randn(4, 1, 8, 8, generator=rng)
    ma = (torch.rand(4, 1, 8, 8, generator=rng) > .5).to(torch.uint8)
    mu = (torch.rand(4, 1, 8, 8, generator=rng) > .5).to(torch.uint8)
    if task == "forward":
        mu.zero_()
    if task == "inverse":
        ma.zero_()
    if pde == "burger":
        ma.zero_()
    params = dict(nu=.001, T=1.) if pde == "nsnonbounded" else {}
    data = dict(coef_ground_truth=a, sol_ground_truth=u, masks=dict(coef=ma, sol=mu),
                sample_ids=[f"f.mat:{i}" for i in range(4)], source_indices=torch.arange(4),
                pde_params=params)
    c = AblationConfig(pde=pde, task=task, sample_seed=0, num_steps=100,
                       pde_guidance_start_ratio=0., zeta_obs_a=2., zeta_obs_u=3.,
                       zeta_pde=1e-6 if clip == "none" else .2, clip_mode=clip, clip_threshold=.2,
                       residual_mode="endpoint_secant" if pde == "nsnonbounded" else "auto",
                       coef_positive_mode="none", num_obs=32, img_resolution=8)
    cell = dict(cell_id=f"supervised/{pde}/id/sparse_joint", config=c.asdict(),
                pde=pde, setting="sparse_joint")
    return cell, data, (Velocity(), Identity(1 if pde == "burger" else 2), {})


@pytest.mark.parametrize("pde", ["poisson", "helmholtz", "darcy", "nsnonbounded", "burger"])
@pytest.mark.parametrize("clip", ["global_norm", "none", "per_component_norm"])
def test_stock_repeat_and_batch_partition(pde, clip):
    torch.set_num_threads(1)
    cell, data, bundle = fixture(pde, clip)
    def run(ids):
        cfg = effective_config(cell, "unused.pth", "cpu", ids, 4)
        gt, masks, _ = observation_batch(data, cfg, ids, "cpu")
        return infer(cfg, bundle, gt, masks, ids, steps=4)[0]
    reference = run([0, 2, 3])
    assert torch.equal(run([0, 2, 3]), reference)
    reordered = torch.cat([run([3]), run([2, 0])])
    torch.testing.assert_close(reordered, reference[[2, 1, 0]], rtol=2e-5, atol=2e-6)


def test_kernel_always_uses_stock_component_gradients(monkeypatch):
    import sampling.runner as stock
    cell, data, bundle = fixture()
    cfg = effective_config(cell, "unused", "cpu", [0, 2], 4)
    gt, masks, _ = observation_batch(data, cfg, [0, 2], "cpu")
    original = stock.compute_guidance_gradient
    calls = []
    def record(losses, target, schedule, config):
        result = original(losses, target, schedule, config)
        calls.append(result.metadata)
        return result
    monkeypatch.setattr(stock, "compute_guidance_gradient", record)
    _, metadata = infer(cfg, bundle, gt, masks, [0, 2], steps=4)
    assert len(calls) == 4
    assert all(c["loss_gradient_batch_reduction"] == "sum_of_per_sample" for c in calls)
    assert all(c["clip_scope"] == "per_sample" for c in calls)
    assert metadata["fused_weighted_gradient"] is False
    assert metadata["gradient_operator"] == "stock_separate_component_gradients_v1"


@pytest.mark.parametrize("task", ["forward", "inverse", "both"])
def test_observation_only_access_and_actual_tensor_hashes(task):
    cell, data, bundle = fixture(task=task)
    cfg = effective_config(cell, "unused", "cpu", [1, 3], 4)
    gt, masks, hashes = observation_batch(data, cfg, [1, 3], "cpu")
    assert not torch.count_nonzero(gt.coef * (1 - masks.coef))
    assert not torch.count_nonzero(gt.sol * (1 - masks.sol))
    assert hashes["observed_coef"] == tensor_digest(gt.coef)
    altered = dict(data, coef_ground_truth=data["coef_ground_truth"] *
                   (1 + 7 * (1 - data["masks"]["coef"])))
    hidden_gt, _, hidden_hashes = observation_batch(altered, cfg, [1, 3], "cpu")
    assert hashes["coef_truth"] != hidden_hashes["coef_truth"]
    assert hashes["observed_coef"] == hidden_hashes["observed_coef"]
    baseline = infer(cfg, bundle, gt, masks, [1, 3], steps=3)[0]
    hidden = infer(cfg, bundle, hidden_gt, masks, [1, 3], steps=3)[0]
    assert torch.equal(baseline, hidden)


def test_resume_refuses_changed_bindings_and_corruption(tmp_path):
    identity = dict(protocol="frozen")
    folder = tmp_path / "batch_abc"
    folder.mkdir()
    prediction = folder / "prediction.pt"
    torch.save(dict(prediction=torch.ones(2, 1, 2, 2)), prediction)
    (folder / "receipt.json").write_text(json.dumps(dict(identity=identity,
        indices=[4, 8], prediction_sha256=digest(prediction))))
    assert existing_batches(tmp_path, identity) == {4, 8}
    with pytest.raises(ValueError, match="bindings"):
        existing_batches(tmp_path, dict(protocol="different"))
    prediction.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        existing_batches(tmp_path, identity)


def test_family_reuses_gpu_model_not_uuid_or_distribution_but_checks_task():
    cell, _, _ = fixture()
    bindings = dict(checkpoint_sha256="weights")
    environment = dict(torch="t", cuda="c", gpu="A100", gpu_uuid="first", tf32=False)
    family = pilot_family(cell, bindings, environment, {})
    changed = dict(cell, cell_id="supervised/poisson/rough/sparse_joint",
                   config=dict(cell["config"], zeta_pde=80., clip_threshold=10.))
    assert pilot_family(changed, bindings, dict(environment, gpu_uuid="second"), {}) == family
    changed["config"]["task"] = "inverse"
    assert pilot_family(changed, bindings, environment, {}) != family


def protocol_fixture(status, cells=57):
    result = dict(status=status, cells=[dict(cell_id=f"cell{i}", count=1000) for i in range(cells)])
    result["content_sha256"] = json_digest(result)
    return result


def test_protocol_status_and_complete_production_scope():
    assert len(validate_protocol(protocol_fixture("frozen"), "run")) == 57
    assert len(validate_protocol(protocol_fixture("pilot_only", 1), "pilot")) == 1
    assert len(validate_protocol(protocol_fixture("frozen"), "pilot")) == 57
    with pytest.raises(ValueError, match="not allowed"):
        validate_protocol(protocol_fixture("pilot_only"), "run")
    with pytest.raises(ValueError, match="57-cell/57000-row"):
        validate_protocol(protocol_fixture("frozen", 56), "run")
    with pytest.raises(ValueError, match="not allowed"):
        validate_protocol(protocol_fixture("draft", 1), "pilot")
    corrupt = protocol_fixture("pilot_only", 1)
    corrupt["cells"][0]["count"] = 999
    with pytest.raises(ValueError, match="content hash"):
        validate_protocol(corrupt, "pilot")
    corrupt.pop("content_sha256")
    corrupt["content_sha256"] = json_digest(corrupt)
    with pytest.raises(ValueError, match="exactly 1000"):
        validate_protocol(corrupt, "pilot")


def test_completed_atomic_batches_resume_without_new_inference(tmp_path, monkeypatch):
    import experiments.aligned_sampling.run_inference as runner
    cell, data, bundle = fixture()
    bindings = dict(protocol_sha256="protocol", checkpoint_sha256="weights",
                    input_truth_sha256="truth", input_masks_sha256="mask")
    env = dict(torch="t", cuda=None, gpu=None, gpu_uuid=None, tf32=False)
    code = {"runner": "a" * 64}
    cert = tmp_path / "pilot.json"
    cert.write_text(json.dumps(dict(status="pass", family=pilot_family(cell, bindings, env, code),
                                    max_batch_size=3)))
    args = types.SimpleNamespace(pilot_certificate=[cert], batch_size=2, device="cpu", start=0, stop=4)
    folder = tmp_path / "results"
    folder.mkdir()
    production(args, cell, data, bundle, "unused", bindings, env, code, folder)
    committed = {p: digest(p) for p in folder.glob("batch_*/*")}
    assert len(committed) == 4
    assert not list(folder.glob(".partial_*"))
    for p in folder.glob("batch_*/prediction.pt"):
        saved = torch.load(p, weights_only=False)
        assert saved["runtime_input_hashes"]["observed_coef"]
        assert saved["effective_config"]["output_dir"] == str(folder.resolve())
    def forbidden(*args, **kwargs):
        raise AssertionError("Completed rows must not be resampled")
    monkeypatch.setattr(runner, "infer", forbidden)
    args.batch_size = 3
    production(args, cell, data, bundle, "unused", bindings, env, code, folder)
    assert {p: digest(p) for p in committed} == committed


def historical_module(commit, filename):
    source = subprocess.check_output(["git", "show", f"{commit}:sampling/{filename}.py"],
                                     cwd=ROOT, text=True)
    name = f"_historical_{commit}_{filename}"
    module = types.ModuleType(name)
    sys.modules[name] = module
    exec(compile(source, name, "exec"), module.__dict__)
    return module


@pytest.mark.parametrize("pde", ["poisson", "helmholtz", "darcy", "nsnonbounded", "burger"])
def test_f377_active_mse_residual_and_gradient_are_unchanged(pde):
    from sampling.losses import compute_guidance_losses
    from sampling.guidance import compute_guidance_gradient, make_zeta_schedule
    from sampling.state import SplitState
    old_losses = historical_module("f377", "losses")
    old_guidance = historical_module("f377", "guidance")
    cell, data, _ = fixture(pde)
    cfg = effective_config(cell, "unused", "cpu", [0, 2], 4)
    gt, masks, _ = observation_batch(data, cfg, [0, 2], "cpu")
    x = torch.randn(2, 1 if pde == "burger" else 2, 8, 8, requires_grad=True)
    state = SplitState(x[:, :1], x[:, :1] if pde == "burger" else x[:, 1:2])
    t, tn = torch.tensor(.9), torch.tensor(.91)
    new = compute_guidance_losses(state, gt, masks, cfg)
    old = old_losses.compute_guidance_losses(state, gt, masks, cfg)
    assert torch.equal(new.L_pde, old.L_pde)
    assert torch.equal(new.L_obs_a, old.L_obs_a)
    assert torch.equal(new.L_obs_u, old.L_obs_u)
    ng = compute_guidance_gradient(new, x, make_zeta_schedule(cfg, t, tn, t), cfg)
    og = old_guidance.compute_guidance_gradient(old, x, old_guidance.make_zeta_schedule(cfg, t, tn, t), cfg)
    assert torch.equal(ng.grad_total, og.grad_total)
