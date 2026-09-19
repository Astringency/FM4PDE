"""Measure e_k and counterfactual corrections on the original sampler path.

Endpoint loss differences are observable. They are not p_k, which also needs
unknown time-dependent local objective minima. Extra calls are excluded from
the production timing comparison.
"""
import argparse
import copy
from pathlib import Path

import torch

from experiments.proposal_guidance.study import prepared, config_batch
from experiments.aligned_sampling.run_inference import tensor_digest
from sampling.sampler_wrappers import _call_velocity_model
from sampling.time_grid import endpoint_from_velocity
from sampling.metrics import write_json
import sampling.runner as r


def raw_gradient(gradient, schedule):
    return (schedule.zeta_obs_a_t * gradient.grad_obs_a
            + schedule.zeta_obs_u_t * gradient.grad_obs_u
            + schedule.zeta_pde_t * gradient.grad_pde)


def objective(losses, schedule):
    return float((schedule.zeta_obs_a_t * losses.guidance_L_obs_a
                  + schedule.zeta_obs_u_t * losses.guidance_L_obs_u
                  + schedule.zeta_pde_t * losses.guidance_L_pde).detach().cpu())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    args = parser.parse_args()
    p, data, bundle = prepared(args)
    indices = p["indices"][:p["batch_size"]]
    cfg, gt, masks, hashes = config_batch(p, data, args.task, indices, 0, "original", args.root)
    net, normalizer, _ = bundle
    r._set_seed(cfg.sample_seed)
    grid = r.make_time_grid(cfg.time_grid, cfg.num_steps, cfg.device)
    x = r._sample_initial_noise(cfg, gt, torch.device(cfg.device))
    initial_hash = tensor_digest(x)
    trace = []

    def endpoint(state, t):
        return state if float(t) == 1 else endpoint_from_velocity(state, _call_velocity_model(net, state, t, None), t)

    def loss(state):
        physical = r._physical_from_model_state(state, cfg, normalizer)
        return r.compute_guidance_losses(physical, gt, masks, cfg)

    selected = {0, 1, 5, 10, 25, 50, 75, 79, 80, 90, 95, 98, 99}
    for k in range(cfg.num_steps):
        cur = x.detach().requires_grad_(True)
        t, tn = grid[k], grid[k + 1]
        out = r.sampler_step(net, cur, t, tn, "stochastic", "euler", "endpoint", device=cfg.device,
            stochastic_noise_source_batch_size=1000, stochastic_noise_source_indices=indices)
        losses = loss(out.x_loss_state)
        affine = r.affine_coefficients(r.scheduler_coefficients(t), training="velocity")
        schedule = r.make_zeta_schedule(cfg, t, tn, affine.b_t)
        old = r.compute_guidance_gradient(losses, cur, schedule, cfg)
        guided = r.apply_guidance_update(out.x_raw_next, old, out, schedule, cfg).detach()
        if k in selected:
            proposal = out.x_raw_next.detach().requires_grad_(True)
            fresh_losses = loss(endpoint(proposal, tn))
            fresh = r.compute_guidance_gradient(fresh_losses, proposal, schedule, cfg)
            counterfactual = r.apply_guidance_update(proposal, fresh, out, schedule, cfg).detach()
            a, b = raw_gradient(old, schedule).flatten(1), raw_gradient(fresh, schedule).flatten(1)
            na, nb = a.norm(dim=1), b.norm(dim=1)
            ne = (a - b).norm(dim=1)
            with torch.no_grad():
                stale_value = objective(loss(endpoint(guided, tn)), schedule)
                fresh_value = objective(loss(endpoint(counterfactual, tn)), schedule)
            trace.append(dict(step=k, t=float(t), t_next=float(tn),
                objective_before_proposal=objective(losses, schedule),
                objective_at_proposal=objective(fresh_losses, schedule),
                objective_after_stale=stale_value, objective_after_fresh=fresh_value,
                gradient_cosine=((a*b).sum(1)/(na*nb).clamp_min(1e-30)).cpu().tolist(),
                e_norm=ne.cpu().tolist(), e_over_gplus=(ne/nb.clamp_min(1e-30)).cpu().tolist(),
                gminus_norm=na.cpu().tolist(), gplus_norm=nb.cpu().tolist(),
                old_clip_scale=old.clip_scale, fresh_clip_scale=fresh.clip_scale))
            print(args.task, trace[-1], flush=True)
        x = guided
    write_json(args.root / "diagnostics" / f"{args.task}.json", dict(
        indices=indices, seed=0, initial_noise_sha256=initial_hash, input_hashes=hashes,
        endpoint_evaluations_time="t_next", same_schedule_for_both_gradients=True,
        path="original sampler; post-proposal correction is a one-step counterfactual",
        final_prediction_sha256=tensor_digest(torch.cat([
            r._physical_from_model_state(x, cfg, normalizer).coef,
            r._physical_from_model_state(x, cfg, normalizer).sol], 1)), trace=trace))


if __name__ == "__main__":
    main()
