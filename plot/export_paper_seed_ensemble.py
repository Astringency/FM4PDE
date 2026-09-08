"""Analyze the registered 11-PDE, 32-input, three-seed experiment.

Exports fieldwise accuracy and spatial spectra from physical predictions.
No a/u maximum or average is used as a reported score. Partial PDE exports
are explicitly marked and are not a substitute for the full 11-PDE study.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.fft import dctn
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_paper_ablation_revision import PDES, digest

NONPERIODIC = {'poisson', 'helmholtz', 'darcy', 'steady_heat_conduction'}
NAMES = dict(poisson='Poisson', helmholtz='Helmholtz', darcy='Darcy',
             nsnonbounded='Navier--Stokes', burger='Burgers',
             reaction_diffusion='Reaction--Diff.', shallow_water='Shallow Water',
             heat='Heat', wave='Wave', advection_diffusion='Adv.--Diff.',
             steady_heat_conduction='Steady Heat')


def write_csv(path, rows):
    assert rows, path
    with path.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def tensor_hash(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def spectrum(field, pde):
    """Field layout C,H,W; Burgers uses C,T,X and only transforms X."""
    assert field.ndim == 3 and field.shape[-2:] == (128, 128)
    if pde == 'burger':
        coeff = np.fft.fft(field, axis=-1, norm='ortho')
        radius = np.abs(np.fft.fftfreq(field.shape[-1]) * field.shape[-1])
        radius = np.broadcast_to(radius, field.shape[-2:])
    elif pde in NONPERIODIC:
        coeff = dctn(field, type=2, axes=(-2, -1), norm='ortho')
        y, x = np.indices(field.shape[-2:])
        radius = np.hypot(x, y)
    else:
        coeff = np.fft.fft2(field, axes=(-2, -1), norm='ortho')
        y = np.fft.fftfreq(field.shape[-2]) * field.shape[-2]
        x = np.fft.fftfreq(field.shape[-1]) * field.shape[-1]
        radius = np.hypot(y[:, None], x[None, :])
    return coeff, radius


def analyze(args):
    torch.set_num_threads(2)
    target = args.output
    target.mkdir(parents=True, exist_ok=True)
    per_input, bands, summary, verified = [], [], [], []
    all_shells = {}
    # Register the same bootstrap input draws for every field and estimator.
    rng = np.random.default_rng(20260908)
    bootstrap_ids = rng.integers(0, 32, (20000, 32))
    estimators = ['seed0', 'single_seed_average', 'mean3']
    for pde in args.pdes:
        source = args.inputs / pde
        protocol = json.loads((source / 'protocol.json').read_text())
        protocol_hash = digest(source / 'protocol.json')
        run_root = args.results / pde
        assert (run_root / 'ensemble_complete.json').exists(), (pde, 'ensemble still running')
        assert protocol['evaluation_ids'] == list(range(1500, 1532))
        assert protocol['inference_seeds'] == [0, 1, 2]
        fields = ['u'] if pde == 'burger' else ['a', 'u']
        pde_rows = []
        shells = {f: {e: [] for e in estimators} for f in fields}
        shells_ref = {f: [] for f in fields}
        for sample_id in protocol['evaluation_ids']:
            predictions = {f: [] for f in fields}
            truths = {}
            mask_reference = None
            configs = []
            for seed in protocol['inference_seeds']:
                folder = run_root / 'ensemble' / f'seed{seed}' / f'sample{sample_id}'
                receipt = json.loads((folder / 'receipt.json').read_text())
                assert receipt['protocol_sha256'] == protocol_hash
                assert receipt['sample_ids'] == [sample_id] and receipt['finite']
                assert receipt['config']['num_steps'] == 100 and receipt['nfe'] == 100
                candidates = list(folder.rglob('result.pt'))
                assert len(candidates) == 1, (folder, candidates)
                payload_path = candidates[0]
                assert digest(payload_path) == receipt['result_sha256']
                payload = torch.load(payload_path, map_location='cpu', weights_only=False)
                masks = torch.load(payload_path.parent / 'masks.pt', map_location='cpu', weights_only=False)
                mask_hash = hashlib.sha256(masks['coef'].numpy().tobytes() + masks['sol'].numpy().tobytes()).hexdigest()
                assert mask_hash == receipt['mask_sha256']
                if mask_reference is None:
                    mask_reference = {key: masks[key].clone() for key in ['coef', 'sol']}
                else:
                    assert all(torch.equal(mask_reference[key], masks[key]) for key in mask_reference)
                config = receipt['config'].copy()
                for key in ['sample_seed', 'output_dir']:
                    config.pop(key, None)
                configs.append(config)
                for field in fields:
                    prefix = 'coef' if field == 'a' else 'sol'
                    truth = payload[prefix + '_ground_truth'][0]
                    pred = payload[prefix + '_final'][0].double().numpy()
                    assert np.isfinite(pred).all()
                    if field in truths:
                        assert torch.equal(truths[field], truth)
                    else:
                        truths[field] = truth.clone()
                    truth_np = truth.double().numpy()
                    norm = np.linalg.norm(truth_np)
                    assert norm > 0
                    error = np.linalg.norm(pred - truth_np) / norm
                    assert np.isclose(error, receipt['errors'][field][0], rtol=1e-11, atol=1e-12)
                    predictions[field].append(pred)
                verified.append(dict(pde=pde, sample_id=sample_id, seed=seed,
                                     result_sha256=receipt['result_sha256'], mask_sha256=mask_hash,
                                     receipt=str(folder / 'receipt.json')))
            assert configs[0] == configs[1] == configs[2], (pde, sample_id, 'settings changed across seeds')
            for field in fields:
                truth = truths[field].double().numpy()
                pred = np.stack(predictions[field])
                mean = pred.mean(axis=0)
                norm2 = np.square(truth).sum()
                axes = tuple(range(1, pred.ndim))
                single_squared = np.square(pred - truth).sum(axis=axes) / norm2
                mean_squared = np.square(mean - truth).sum() / norm2
                variance = np.square(pred - mean).sum(axis=axes).mean() / norm2
                assert np.isclose(single_squared.mean(), mean_squared + variance, rtol=1e-11, atol=1e-12)
                errors = dict(seed0=float(np.sqrt(single_squared[0])),
                              single_seed_average=float(np.sqrt(single_squared).mean()),
                              mean3=float(np.sqrt(mean_squared)))
                assert errors['mean3'] <= errors['single_seed_average'] + 1e-12
                row = dict(pde=pde, field=field, sample_id=sample_id,
                           **{key: value*100 for key, value in errors.items()},
                           paired_delta_pp=100*(errors['mean3']-errors['seed0']),
                           squared_variance=variance, truth_sha256=tensor_hash(truths[field]))
                pde_rows.append(row)
                coeff, radius = spectrum(truth, pde)
                predictions_coeff = [spectrum(x, pde)[0] for x in pred]
                mean_coeff = spectrum(mean, pde)[0]
                energy = np.abs(coeff)**2
                assert np.isclose(energy.sum(), norm2, rtol=1e-11, atol=1e-12)
                # Collapse channels only after computing channel-wise energy/error.
                group_coeff = {'seed0': predictions_coeff[:1],
                               'single_seed_average': predictions_coeff,
                               'mean3': [mean_coeff]}
                shell_index = np.floor(radius + 1e-10).astype(int)
                shell_count = shell_index.max() + 1
                ref_shell = np.bincount(shell_index.ravel(), weights=energy.sum(axis=0).ravel(), minlength=shell_count)/norm2
                shells_ref[field].append(ref_shell)
                for estimator, values in group_coeff.items():
                    pred_energy = np.mean([np.abs(x)**2 for x in values], axis=0)
                    error_energy = np.mean([np.abs(x-coeff)**2 for x in values], axis=0)
                    shell_pred = np.bincount(shell_index.ravel(), weights=pred_energy.sum(axis=0).ravel(), minlength=shell_count)/norm2
                    shell_error = np.bincount(shell_index.ravel(), weights=error_energy.sum(axis=0).ravel(), minlength=shell_count)/norm2
                    shells[field][estimator].append(np.stack([shell_pred, shell_error]))
                    for low, high in [(4,16), (8,32), (16,48)]:
                        for name, mask in [('dc', radius == 0), ('low', (radius>0)&(radius<=low)),
                                           ('mid', (radius>low)&(radius<=high)), ('high', radius>high)]:
                            reference = energy[:, mask].sum()
                            prediction = pred_energy[:, mask].sum()
                            error = error_energy[:, mask].sum()
                            supported = reference >= norm2*1e-14
                            band_errors = [np.sqrt(np.square(np.abs(x[:,mask]-coeff[:,mask])).sum()/reference)
                                           for x in values] if supported else []
                            bands.append(dict(pde=pde, field=field, sample_id=sample_id, estimator=estimator,
                                              low_cutoff=low, high_cutoff=high, band=name,
                                              reference_fraction=reference/norm2,
                                              prediction_fraction=prediction/norm2,
                                              error_fraction=error/norm2,
                                              band_error_percent=100*np.mean(band_errors) if supported else '',
                                              energy_ratio=prediction/reference if supported else '',
                                              reference_supported=supported))
        per_input.extend(pde_rows)
        for field in fields:
            field_rows = [x for x in pde_rows if x['field'] == field]
            assert len(field_rows) == 32
            delta = np.array([x['paired_delta_pp'] for x in field_rows])
            boot = delta[bootstrap_ids].mean(axis=1)
            alpha = .05/21
            ci = np.quantile(boot, [.025, .975, alpha/2, 1-alpha/2])
            row = dict(pde=pde, field=field, n=32)
            for estimator in estimators:
                errors = np.array([x[estimator] for x in field_rows])
                row[estimator+'_mean'] = errors.mean()
                row[estimator+'_sd'] = errors.std(ddof=1)
            row.update(paired_delta_pp=delta.mean(), ci95_low=ci[0], ci95_high=ci[1],
                       family21_low=ci[2], family21_high=ci[3], improved_adjusted=bool(ci[3]<0),
                       fraction_inputs_improved=float((delta<0).mean()))
            summary.append(row)
            all_shells[pde+'__'+field+'__reference'] = np.stack(shells_ref[field])
            for estimator in estimators:
                all_shells[pde+'__'+field+'__'+estimator] = np.stack(shells[field][estimator])
        print('ANALYZED', pde, len(pde_rows), 'input-field records', flush=True)
    write_csv(target/'per_input.csv', per_input)
    write_csv(target/'summary.csv', summary)
    write_csv(target/'frequency_per_input.csv', bands)
    write_csv(target/'verified_predictions.csv', verified)
    np.savez_compressed(target/'spectral_shells.npz', **all_shells)
    lines = [r'\begin{table}[!htbp]\centering\scriptsize\setlength{\tabcolsep}{3pt}',
             r'\caption{Averaging three 100-step FM4PDE predictions on 32 inputs per PDE. Errors are percentages, mean $\pm$ sample SD across inputs. Seed 0 is the prespecified single prediction; Single avg. averages the three individual errors; Mean of 3 scores the average prediction. Intervals compare Mean of 3 with Seed 0 and use a Bonferroni adjustment for 21 field comparisons.}',
             r'\label{tab:seed-ensemble-fields}',
             r'\begin{tabular}{@{}llrrrr@{}}\toprule',
             r'PDE & Field & Seed 0 & Single avg. & Mean of 3 & Paired change [adjusted CI] \\\midrule']
    for row in summary:
        stats = [f"${row[e+'_mean']:.2f}\\pm{row[e+'_sd']:.2f}$" for e in estimators]
        lines.append(f"{NAMES[row['pde']]} & ${row['field']}$ & "+' & '.join(stats)+
                     f" & ${row['paired_delta_pp']:.2f}\\ [{row['family21_low']:.2f},{row['family21_high']:.2f}]$ \\\\")
    lines += [r'\bottomrule\end{tabular}\end{table}']
    (target/'ensemble_fields.tex').write_text('\n'.join(lines)+'\n')
    (target/'manifest.json').write_text(json.dumps(dict(
        pdes=args.pdes, full_study=set(args.pdes)==set(PDES), predictions=len(verified),
        expected_full_predictions=1056, registered_family_fields=21,
        bootstrap_resamples=20000, bootstrap_seed=20260908,
        plan_sha256=digest(args.plan), exporter_sha256=digest(Path(__file__)),
        uncertainty='Approximate percentile-bootstrap intervals over 32 physical inputs; fixed checkpoint and three seeds.',
        spectra='Spatial FFT for periodic fields; DCT-II for nonperiodic fields; Burgers FFT only along space, energies summed across time and channels before full-field normalization.',
        outputs={p.name:digest(p) for p in target.iterdir() if p.is_file() and p.name!='manifest.json'}
    ),indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--pdes', nargs='+', choices=PDES, default=PDES)
    analyze(parser.parse_args())
