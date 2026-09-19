"""Fixed equal-mass time-bin evaluation, separately for raw and EMA weights."""
import torch


@torch.no_grad()
def fixed_metrics(model, data, microbatch, seed=20260922, bins=10):
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    errors = []
    try:
        with torch.random.fork_rng(devices=[device.index or 0] if device.type == 'cuda' else []):
            for index in range(bins):
                gen = torch.Generator().manual_seed(seed + index)
                t = ((torch.rand(len(data), generator=gen) + index) / bins).clamp(1e-5, 1 - 1e-5)
                rows = []
                for start in range(0, len(data), microbatch):
                    sample = data[start:start + microbatch].to(device)
                    noise = torch.randn(sample.shape, generator=gen).to(device)
                    time = t[start:start + len(sample)].to(device)
                    xt = noise.lerp(sample, time[:, None, None, None])
                    error = (model(xt, time, extra={}).float() - (sample - noise)).square().mean((2, 3))
                    rows.append(error.cpu())
                errors.append(torch.cat(rows))
    finally:
        model.train(was_training)
    errors = torch.stack(errors)  # [time bin, input, channel]
    if not torch.isfinite(errors).all():
        raise FloatingPointError('Nonfinite fixed validation')
    return dict(mean=float(errors.mean()), by_time_bin=errors.mean((1, 2)).tolist(),
                by_channel=errors.mean((0, 1)).tolist(), by_time_channel=errors.mean(1).tolist(),
                per_input=errors.mean((0, 2)).tolist(),
                per_time_input=errors.mean(2).tolist(), seed=seed, bins=bins,
                input_count=len(data), distribution='uniform; equal mass in every time bin')


def evaluate_pair(ema, data, microbatch, seed=20260922, bins=10):
    was_training = ema.training
    try:
        ema.train(True)
        raw = fixed_metrics(ema.model, data, microbatch, seed, bins)
        ema.eval()
        averaged = fixed_metrics(ema.model, data, microbatch, seed, bins)
    finally:
        ema.train(was_training)
    return dict(raw=raw, ema=averaged, ema_updates=int(ema.num_updates), ema_decay=ema.decay)
