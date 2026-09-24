"""Fixed-mask, canonical-noise conditional draws used in the manuscript."""
import copy, hashlib, json, time

def tensor_digest(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(str(value.dtype).encode() + json.dumps(list(value.shape), separators=(',', ':')).encode()
                          + value.numpy().tobytes()).hexdigest()

def fixed_observations(cfg, truth):
    """Generate one mask pair per physical input, before expanding any draws."""
    import torch
    from sampling.masks import make_pair_masks
    pair = torch.cat([truth.coef, truth.sol], 1).detach().cpu()
    assert pair.shape == (1, 2, 128, 128)
    masks = make_pair_masks(truth.coef.shape, truth.sol.shape, cfg.num_obs,
                            cfg.sensor_mode, cfg.shared_mask, cfg.mask_seed, device='cpu')
    mask = torch.cat([masks.coef, masks.sol], 1).cpu()
    assert set(mask.unique().tolist()) <= {0., 1.}
    assert torch.equal(mask.sum((-2, -1)), torch.full((1, 2), 500.))
    values = pair * mask
    fixed = dict(truth=pair, masks=mask, observations=values,
                 observed_fields=['coef'] if cfg.task == 'forward' else ['sol'] if cfg.task == 'inverse' else ['coef', 'sol'],
                 metadata={**masks.metadata, 'source': 'single_mask_pair_broadcast_to_every_draw'})
    fixed['hashes'] = {key: tensor_digest(fixed[key]) for key in ['truth', 'masks', 'observations']}
    return fixed

def expand_fixed(fixed, gt, batch_size, device):
    import torch
    from sampling.masks import PairMasks
    from sampling.losses import ObservationTargets
    mask = fixed['masks'].to(device).expand(batch_size, -1, -1, -1)
    values = fixed['observations'].to(device).expand(batch_size, -1, -1, -1)
    pair = torch.cat([gt.coef, gt.sol], 1)
    # Check the tensors actually handed to the loss, not merely metadata.
    assert torch.equal(pair, fixed['truth'].to(device).expand_as(pair))
    assert torch.equal(pair * mask, values)
    assert torch.equal(mask, mask[:1].expand_as(mask))
    assert torch.equal(values, values[:1].expand_as(values))
    proof = dict(hashes={key: tensor_digest(value[:1]) for key, value in
                        [('truth', pair), ('masks', mask), ('observations', values)]},
                 all_rows_identical=True, checked_rows=batch_size,
                 observed_fields=fixed['observed_fields'])
    assert proof['hashes'] == fixed['hashes']
    masks = PairMasks(mask[:, :1], mask[:, 1:], copy.deepcopy(fixed['metadata']))
    observations = ObservationTargets(values[:, :1], values[:, 1:], values[:, :1], values[:, 1:])
    return masks, observations, proof

def fast_sample(cfg, truth, bundle, indices, fixed, steps=100):
    import torch
    from sampling.batching import combine_truths
    import sampling.runner as r
    c = copy.deepcopy(cfg)
    c.batch_size = len(indices)
    c.initial_noise_source_indices = list(indices)
    gt = combine_truths({c.offset: truth}, [c.offset]*len(indices), c.device)
    masks, observations, observation_proof = expand_fixed(fixed, gt, len(indices), c.device)
    net, normalizer, _ = bundle
    assert c.obs_l2_reference_mse_zeta_a is None and c.obs_l2_reference_mse_zeta_u is None
    r._set_seed(c.sample_seed)
    grid = r.make_time_grid(c.time_grid, c.num_steps, device=c.device, eta=c.time_grid_eta)
    torch.cuda.synchronize(c.device)
    torch.cuda.reset_peak_memory_stats(c.device)
    calls = {'count':0}
    def count_call(*_): calls['count'] += 1
    hook = net.model.register_forward_hook(count_call)
    start = time.perf_counter()
    x = r._sample_initial_noise(c, gt, c.device)
    initial_noise_hashes = [tensor_digest(row) for row in x.detach().cpu()]
    for step in range(steps):
        x_cur = x.detach().clone().requires_grad_(True)
        t, t_next = grid[step], grid[step+1]
        out = r.sampler_step(net, x_cur, t, t_next, 'stochastic', c.step_method,
                             c.loss_state, device=c.device,
                             stochastic_noise_source_batch_size=1000,
                             stochastic_noise_source_indices=list(indices))
        phys = r._physical_from_model_state(out.x_loss_state, c, normalizer)
        losses = r.compute_guidance_losses(phys, gt, masks, c, observations=observations)
        assert losses.pde_residual_status != 'error'
        coeffs = r.scheduler_coefficients(t, scheduler='CondOT')
        affine = r.affine_coefficients(coeffs, training='velocity')
        schedule = r.make_zeta_schedule(c, t, t_next, affine.b_t, step=step)
        target = r._gradient_target_tensor(c, x_cur, out)
        if c.runtime_metadata.get('fused_guidance'):
            # Global clipping applies after the weighted gradient sum; linearity
            # therefore permits one reverse pass through the velocity network.
            from types import SimpleNamespace
            from sampling.guidance import _clip_per_sample
            assert c.clip_mode in {'global_norm', 'none'}
            total_loss = (schedule.zeta_obs_a_t * losses.guidance_L_obs_a
                          + schedule.zeta_obs_u_t * losses.guidance_L_obs_u
                          + schedule.zeta_pde_t * losses.guidance_L_pde)
            total = torch.autograd.grad(total_loss * len(indices), target)[0]
            total, _ = _clip_per_sample(total, c.clip_threshold, c.clip_mode == 'global_norm')
            gradient = SimpleNamespace(grad_total=total, metadata={})
        else:
            gradient = r.compute_guidance_gradient(losses, target, schedule, c)
        x = r.apply_guidance_update(out.x_raw_next, gradient, out, schedule, c).detach()
    with torch.no_grad():
        phys = r._physical_from_model_state(x, c, normalizer)
        # Averaging is included in the synchronized sampling time.
        mean = torch.cat([phys.coef, phys.sol], dim=1).double().mean(0).float()
    torch.cuda.synchronize(c.device)
    seconds = time.perf_counter()-start
    hook.remove()
    assert calls['count'] == steps
    peak = torch.cuda.max_memory_allocated(c.device)
    pred = torch.cat([phys.coef, phys.sol], dim=1).detach().cpu()
    assert torch.isfinite(pred).all()
    return pred, mean.cpu(), dict(seconds=seconds, peak_bytes=peak, batch_size=len(indices),
                                 seed_indices=list(indices), num_steps=steps, nfe=calls['count'],
                                 observations=observation_proof, initial_noise_hashes=initial_noise_hashes)
