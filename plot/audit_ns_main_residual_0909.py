"""Recompute NS endpoint residuals independently and verify frozen inputs/settings."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from run_ns_main_revision_0909 import EVAL, SETTINGS, configuration, digest, write


def endpoint_mse(pred, nu=0.001, duration=1.0):
    """NumPy float64 midpoint secant on the unit torus, with 2/3 projection."""
    a, u = np.asarray(pred, dtype=np.float64)[:, 0], np.asarray(pred, dtype=np.float64)[:, 1]
    h, w = a.shape[-2:]
    midpoint = (a + u) / 2
    midpoint -= midpoint.mean(axis=(-2, -1), keepdims=True)
    mx, my = np.fft.fftfreq(h, d=1/h), np.fft.fftfreq(w, d=1/w)
    kx, ky = 2*np.pi*mx[:, None], 2*np.pi*my[None, :]
    k2 = kx*kx + ky*ky
    transformed = np.fft.fft2(midpoint)
    stream = transformed / np.where(k2 > 0, k2, 1)
    stream[:, 0, 0] = 0
    dx = np.fft.ifft2(1j*kx*transformed).real
    dy = np.fft.ifft2(1j*ky*transformed).real
    vx = np.fft.ifft2(1j*ky*stream).real
    vy = np.fft.ifft2(-1j*kx*stream).real
    lap = np.fft.ifft2(-k2*transformed).real
    retained = ((np.abs(mx[:, None]) <= (2/3)*(h//2)) &
                (np.abs(my[None, :]) <= (2/3)*(w//2)))
    advection = np.fft.ifft2(np.fft.fft2(vx*dx + vy*dy)*retained).real
    xy = np.arange(h)[:, None]/h + np.arange(w)[None, :]/w
    forcing = 0.1*(np.sin(2*np.pi*xy) + np.cos(2*np.pi*xy))
    forcing = np.fft.ifft2(np.fft.fft2(forcing)*retained).real
    residual = (u-a)/duration + advection - nu*lap - forcing
    return np.square(residual).mean(axis=(-2, -1))


def analytic_checks():
    n = 128
    constant = np.full((1, 2, n, n), 0.37)
    assert np.allclose(endpoint_mse(constant), 0.01, rtol=0, atol=1e-14)
    wave = np.broadcast_to(np.sin(2*np.pi*np.arange(n)[:, None]/n), (n, n))
    a, u = 0.7, 0.4
    pred = np.stack([a*wave, u*wave])[None]
    expected = 0.01 + 0.5*((u-a) + 0.001*4*np.pi**2*(a+u)/2)**2
    assert np.allclose(endpoint_mse(pred), expected, rtol=0, atol=1e-14)


def main(args):
    torch.set_num_threads(2)
    analytic_checks()
    protocol_path = args.audit/'inputs/protocol.json'
    selection_path = args.audit/'selection.json'
    protocol = json.loads(protocol_path.read_text())
    selection = json.loads(selection_path.read_text())
    protocol_hash, selection_hash = digest(protocol_path), digest(selection_path)
    assert selection['protocol_sha256'] == protocol_hash
    assert digest(args.audit/'weights.pth') == protocol['model']['weights_sha256']
    cache = {}
    for dist in ['id', 'smooth', 'rough']:
        path = args.audit/'inputs'/f'{dist}.npz'
        assert digest(path) == protocol['sources'][dist]['cache_sha256']
        data = np.load(path)
        cache[dist] = {int(i): row for i, row in zip(data['ids'], data['fields'])}
    files = sorted(args.results.rglob('offset*.pt'))
    seen = {(dist, setting): [] for dist in cache for setting in SETTINGS}
    maxima = dict(absolute=0.0, relative=0.0)
    all_masks = {}
    n = 0
    for path in files:
        receipt = json.loads(path.with_suffix('.json').read_text())
        assert digest(path) == receipt['result_sha256']
        assert receipt['protocol_sha256'] == protocol_hash
        assert receipt['selection_sha256'] == selection_hash
        data = torch.load(path, map_location='cpu', weights_only=False)
        dist, setting, ids = receipt['dist'], receipt['setting'], receipt['ids']
        cfg = configuration(protocol, setting, dist, data['config']['checkpoint_path'],
                            selection['scaled_inverse'].get(setting, False))
        cfg.offset = ids[0]
        assert data['config'] == cfg.asdict(), (path, 'frozen configuration mismatch')
        assert cfg.pde_residual_region == 'full' and not cfg.enforce_boundary_conditions
        assert cfg.ns_operator_mode == 'generator_dealiased'
        expected_truth = np.stack([cache[dist][i] for i in ids])
        assert np.array_equal(data['truths'].numpy(), expected_truth), (path, 'truth mismatch')
        masks = data['masks'].numpy()
        assert np.isin(masks, [0, 1]).all()
        counts = masks.sum(axis=(-2, -1))
        expected_counts = {'forward': [cfg.num_obs, 0], 'inverse': [0, cfg.num_obs],
                           'both': [cfg.num_obs, cfg.num_obs]}[cfg.task]
        assert np.array_equal(counts, np.broadcast_to(expected_counts, counts.shape))
        if setting not in all_masks:
            all_masks[setting] = masks[0].copy()
        assert np.array_equal(masks, np.broadcast_to(all_masks[setting], masks.shape))
        calculated = endpoint_mse(data['predictions'].numpy())
        reported = np.array([row['pde_mse'] for row in receipt['rows']])
        absolute = np.abs(calculated - reported)
        relative = absolute / np.maximum(np.abs(calculated), 1e-30)
        maxima['absolute'] = max(maxima['absolute'], float(absolute.max()))
        maxima['relative'] = max(maxima['relative'], float(relative.max()))
        # NumPy uses float64 FFTs; production residuals use CUDA float32 FFTs.
        assert np.allclose(calculated, reported, rtol=2e-5, atol=1e-9), (path, relative.max())
        seen[dist, setting].extend(ids)
        n += len(ids)
    complete = all(sorted(ids) == EVAL for ids in seen.values())
    if not args.allow_partial:
        assert complete and n == 15000
    result = dict(status='pass', complete=complete, examples=n, batches=len(files),
        independent_residual='NumPy float64 midpoint endpoint secant; generator 2/3 dealiased advection and forcing',
        analytic_constant_and_fourier_mode_checks=True, max_pde_mse_difference=maxima,
        tolerance=dict(rtol=2e-5, atol=1e-9), all_frozen_configurations_match=True,
        all_truths_equal_frozen_source_cache=True, all_protocol_and_selection_hashes_match=True,
        all_observation_counts_and_masks_match=True, model_weights_sha256=protocol['model']['weights_sha256'],
        protocol_sha256=protocol_hash, selection_sha256=selection_hash,
        cells=[dict(dist=d, setting=s, n=len(ids)) for (d, s), ids in seen.items()])
    write(args.output, result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-partial', action='store_true')
    main(parser.parse_args())
