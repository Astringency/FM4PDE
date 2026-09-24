"""Resident-model sampling, with diagnostics and file I/O outside timing."""
def fm_predict(config, bundle, gt, masks):
    """Original sampler update path, omitting scoring, progress and file I/O.

    Pilot validation compares this directly with run_single_ablation. No
    replacement denoiser, residual, derivative, schedule or update is used.
    """
    import torch
    import sampling.runner as r
    cfg=r.finalize_ground_truth_config(config);cfg.validate();r._disable_unreliable_pde_guidance(cfg)
    r._set_seed(cfg.sample_seed);device=torch.device(cfg.device)
    net,normalizer,payload=bundle
    a=r.add_observation_noise(gt.coef,masks.coef,0.,seed=cfg.noise_seed)
    u=r.add_observation_noise(gt.sol,masks.sol,0.,seed=cfg.noise_seed+1)
    obs=r.ObservationTargets(a.clean,u.clean,a.noisy,u.noisy)
    scalar,_=r._scalar_conditioning_for_sampling(checkpoint_payload=payload,gt=gt,config=cfg,device=device)
    classes,_=r._class_conditioning_for_sampling(checkpoint_payload=payload,pde=cfg.pde,batch_size=1,device=device,cfg_scale=cfg.cfg_scale)
    extra={**classes,**(scalar or {})} or None
    r._check_sampling_channels(gt,normalizer,payload)
    grid=r.make_time_grid(cfg.time_grid,cfg.num_steps,device=device,eta=cfg.time_grid_eta)
    x=r._sample_initial_noise(cfg,gt,device)
    for k in range(cfg.num_steps):
        phase=r.phase_for_step(cfg.sampler_phase,cfg.switch_ratio,k,cfg.num_steps)
        cur=x.detach().clone().requires_grad_(True)
        t,tn=grid[k],grid[k+1]
        out=r.sampler_step(net=net,x_cur=cur,t=t,t_next=tn,phase=phase,
                          step_method=cfg.step_method,loss_state=cfg.loss_state,device=device,model_extra=extra,
                          stochastic_noise_source_batch_size=cfg.initial_noise_source_batch_size,
                          stochastic_noise_source_indices=cfg.initial_noise_source_indices or None,
                          deterministic_endpoint_mode=cfg.deterministic_endpoint_mode,
                          deterministic_endpoint_time_grid=grid[k:],
                          deterministic_rollout_checkpoint=cfg.deterministic_rollout_checkpoint)
        physical=r._physical_from_model_state(out.x_loss_state,cfg,normalizer)
        losses=r.compute_guidance_losses(physical,gt,masks,cfg,obs)
        assert not r._calibrate_l2_observation_zeta(cfg,losses,step=k)
        affine=r.affine_coefficients(r.scheduler_coefficients(t,scheduler='CondOT'),training='velocity')
        schedule=r.make_zeta_schedule(cfg,t,tn,affine.b_t,step=k)
        gradient=r.compute_guidance_gradient(losses,r._gradient_target_tensor(cfg,cur,out),schedule,cfg)
        x=r.apply_guidance_update(out.x_raw_next,gradient,out,schedule,cfg).detach()
    final=r._physical_from_model_state(x,cfg,normalizer)
    return final.coef.detach(),final.sol.detach()
