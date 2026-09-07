"""Truth-only check of NS temporal-residual approximation; never sampler input.

Freeze the 11 archived frames for the already fixed 32 examples, then compare
endpoint and trajectory diagnostics on CPU. No checkpoint or guidance weight
is selected, trained, changed, or evaluated by this script.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_ns_loss_study import sha, write


def freeze(args):
    import h5py
    source = json.loads((args.inputs / 'source.json').read_text())
    assert not (args.output / 'trajectory_manifest.json').exists()
    assert sha(args.inputs / 'fields_masks.npz') == source['fields_sha256']
    endpoint = np.load(args.inputs / 'fields_masks.npz')
    path = Path(source['data_path'])
    assert path.stat().st_size == source['data_size'] and path.stat().st_mtime_ns == source['data_mtime_ns']
    ids = source['evaluation_ids']
    assert len(ids) == len(set(ids)) == 32
    fields = []
    with h5py.File(path, 'r') as f:
        attrs = {k: np.asarray(v).tolist() for k, v in f.attrs.items()}
        assert attrs == source['data_attrs']
        times = np.asarray(f['t']).reshape(-1)
        assert times.shape == (10,)
        for i in ids:
            initial, later = np.asarray(f['w0'][i]), np.asarray(f['w'][i])
            assert initial.shape == (128, 128) and later.shape == (128, 128, 10)
            assert np.array_equal(initial, endpoint[f'a_{i}'][0, 0])
            assert np.array_equal(later[:, :, -1], endpoint[f'u_{i}'][0, 0])
            fields.append(np.concatenate([initial[None], later.transpose(2, 0, 1)], axis=0)[:, None])
    fields = np.stack(fields)
    assert fields.shape == (32, 11, 1, 128, 128) and np.isfinite(fields).all()
    nominal_times = np.linspace(0, source['pde_params']['T'], 11)
    assert np.allclose(times, nominal_times[1:], rtol=0, atol=2e-7)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / 'true_trajectories.npz', fields=fields, sample_ids=ids,
                        archived_times=times, nominal_times=nominal_times)
    write(args.output / 'trajectory_manifest.json', dict(
        role='Truth-only temporal-discretization diagnostic; not used for sampling, calibration, or checkpoint selection.',
        source_sha256=sha(args.inputs / 'source.json'), endpoint_fields_sha256=source['fields_sha256'],
        data_path=str(path), data_size=source['data_size'], data_mtime_ns=source['data_mtime_ns'],
        sample_ids=ids, shape=list(fields.shape), dtype=str(fields.dtype),
        maximum_timestamp_roundoff=float(np.max(np.abs(times - nominal_times[1:]))),
        trajectory_sha256=sha(args.output / 'true_trajectories.npz'), script_sha256=sha(Path(__file__))))
    print('Frozen 32 true trajectories; both endpoints match original study bitwise.', flush=True)


def audit(args):
    import torch
    from sampling.config import AblationConfig
    from sampling.pde_residuals import _rhs_nsnonbounded, compute_pde_residual
    from ns_loss_exchange import fm_pde_loss
    torch.set_num_threads(2)
    manifest = json.loads((args.trajectories / 'trajectory_manifest.json').read_text())
    assert sha(args.inputs / 'source.json') == manifest['source_sha256']
    assert sha(args.trajectories / 'true_trajectories.npz') == manifest['trajectory_sha256']
    source = json.loads((args.inputs / 'source.json').read_text())
    assert sha(args.inputs / 'fields_masks.npz') == manifest['endpoint_fields_sha256'] == source['fields_sha256']
    frozen = np.load(args.inputs / 'fields_masks.npz')
    data = np.load(args.trajectories / 'true_trajectories.npz')
    assert data['sample_ids'].tolist() == source['evaluation_ids'] == manifest['sample_ids']
    assert data['fields'].shape == (32, 11, 1, 128, 128)
    T = float(source['pde_params']['T'])
    h = T / 10
    assert np.array_equal(data['nominal_times'], np.linspace(0, T, 11))
    params = {k: torch.tensor([v], dtype=torch.float64) for k, v in source['pde_params'].items()}
    cfg = AblationConfig(**source['fm_configs']['both'])
    rows = []
    for i, value in zip(data['sample_ids'], data['fields']):
        assert np.array_equal(value[0], frozen[f'a_{i}'][0]) and np.array_equal(value[-1], frozen[f'u_{i}'][0])
        w = torch.from_numpy(value).double()
        a, u = w[:1], w[-1:]
        with torch.no_grad():
            rhs = _rhs_nsnonbounded(w, params)
            secant = (u - a) / T - _rhs_nsnonbounded((a + u) / 2, params)
            assert np.isclose(float(secant.square().mean()), float(fm_pde_loss(a, u, cfg, params)), rtol=2e-12, atol=1e-18)
            centered = (w[2:] - w[:-2]) / (2 * h) - rhs[1:-1]
            temporal_params = dict(params, full_trajectory=w[None], trajectory_is_observed_ground_truth=True,
                                   boundary_condition_mode='none', enforce_boundary_conditions=False)
            native = compute_pde_residual('nsnonbounded', a, u, pde_params=temporal_params, residual_mode='full_trajectory_fd')
            assert native.metadata['guidance_compatible'] is False
            assert torch.allclose(native.residual.reshape_as(centered), centered, rtol=1e-12, atol=1e-14)
            centers = torch.tensor([2, 4, 6, 8])
            fd_h = (w[centers + 1] - w[centers - 1]) / (2 * h) - rhs[centers]
            fd_2h = (w[centers + 2] - w[centers - 2]) / (4 * h) - rhs[centers]
            trap = (rhs[0] / 2 + rhs[1:-1].sum(0) + rhs[-1] / 2) * h / T
            weights = torch.tensor([1, 4, 2, 4, 2, 4, 2, 4, 2, 4, 1], dtype=w.dtype).view(11, 1, 1, 1)
            simpson = (rhs * weights).sum(0) * h / (3 * T)
            swapped_rhs = _rhs_nsnonbounded(w[centers].transpose(-1, -2), params).transpose(-1, -2)
            swapped = (w[centers + 1] - w[centers - 1]) / (2 * h) - swapped_rhs
        rms = lambda x: float(x.square().mean().sqrt())
        row = dict(sample_id=int(i), endpoint_secant_rms=rms(secant),
                   endpoint_drift_true_midpoint_rms=rms((u - a) / T - rhs[5:6]),
                   centered_9_times_rms=rms(centered), centered_h_common_4_times_rms=rms(fd_h),
                   centered_2h_common_4_times_rms=rms(fd_2h), trapezoid_integral_rms=rms((u - a) / T - trap),
                   simpson_integral_rms=rms((u - a) / T - simpson),
                   swapped_axes_centered_h_rms=rms(swapped),
                   endpoint_average_midpoint_relative_error=float(((a + u) / 2 - w[5:6]).norm() / w[5:6].norm()))
        assert np.isfinite(list(row.values())).all()
        rows.append(row)
    summary = [dict(metric=k, n=32, mean=float(np.mean([r[k] for r in rows])),
                    sd=float(np.std([r[k] for r in rows], ddof=1)),
                    minimum=min(r[k] for r in rows), maximum=max(r[k] for r in rows)) for k in rows[0] if k != 'sample_id']
    args.output.mkdir(parents=True, exist_ok=True)
    for name, values in [('true_trajectory_per_input.csv', rows), ('true_trajectory_summary.csv', summary)]:
        with (args.output / name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    write(args.output / 'true_trajectory_audit.json', dict(
        status='complete', examples=32, archived_frames_per_input=11, precision='float64 diagnostics from archived float32 fields',
        source_sha256=manifest['source_sha256'], trajectory_sha256=manifest['trajectory_sha256'],
        frozen_trajectory_manifest_sha256=sha(args.trajectories / 'trajectory_manifest.json'),
        script_sha256=sha(Path(__file__)), residual_source_sha256=sha(ROOT / 'sampling/pde_residuals.py'),
        checks=dict(endpoint_identity=True, exact_main_secant_reduction=True, native_full_trajectory_fd_agreement=True,
                    true_trajectory_marked_incompatible_with_guidance=True),
        temporal_grid=dict(T=T, h=h, common_center_times=[2*h, 4*h, 6*h, 8*h],
                           maximum_archived_timestamp_roundoff=manifest['maximum_timestamp_roundoff']),
        interpretation='Oracle-trajectory diagnostics only. Different rows use different true temporal information. They are not sampling errors or evidence that a new inference method improves reconstruction. The axis-swapped row is an intentional convention sentinel, never a sampler candidate.',
        outputs={p.name: sha(p) for p in args.output.iterdir() if p.is_file() and p.name != 'true_trajectory_audit.json'}))
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['freeze', 'audit'])
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trajectories', type=Path)
    args = parser.parse_args()
    if args.mode == 'audit':
        assert args.trajectories is not None
    freeze(args) if args.mode == 'freeze' else audit(args)


if __name__ == '__main__':
    main()
