"""Frequency diagnostics of FM4PDE ID sampling results across reconstruction tasks.

Reads the archived 1000-input ID collections and the fixed-observation
conditional pools, and separates three quantities that a single energy ratio
conflates: how much band energy a reconstruction retains, how much of that
energy is aligned with the reference, and how much of it survives averaging
independent sampling draws.  Shuffled pairing supplies the alignment expected
from chance alone.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'plot'))
from run_ns_loss_study import sha, write
from spectral_diagnostics import paired_bootstrap, radii, self_check, spectral_record

MIN_REFERENCE_FRACTION = 1e-14
REPORT_BANDS = (('dc', 0.0, 0.0), ('low', 0.0, 8.0), ('mid1', 8.0, 16.0), ('mid2', 16.0, 32.0),
                ('hi1', 32.0, 64.0), ('hi2', 64.0, 96.0), ('hi3', 96.0, None))
LEGACY_BANDS = (('dc', 0.0, 0.0), ('low8', 0.0, 8.0), ('mid8', 8.0, 32.0), ('high8', 32.0, None))
ALL_BANDS = REPORT_BANDS + LEGACY_BANDS[1:]
TASKS = ('forward', 'inverse', 'both')
FIELDS = ('coef', 'sol')
SELECTION = {('poisson', 'forward'): 'MAIN1000_100_TEST_id',
             ('poisson', 'inverse'): 'MAIN1000_100_TEST_id_tuned1',
             ('poisson', 'both'): 'MAIN1000_100_TEST_id',
             ('darcy', 'forward'): 'MAIN1000_100_TEST_id_tuned1',
             ('darcy', 'inverse'): 'MAIN1000_100_TEST_id_tuned1',
             ('darcy', 'both'): 'MAIN1000_100_TEST_id'}
OBSERVED = {'forward': ('coef',), 'inverse': ('sol',), 'both': ('coef', 'sol')}
TARGET = {'forward': ('sol',), 'inverse': ('coef',), 'both': ('coef', 'sol')}
KS = (1, 3, 10, 100, 1000)
HIGH_RADIUS = 32.0
DISTANCE_BINS = (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0, 96.0, 1e4)


def band_masks(radius, bands=ALL_BANDS):
    out = {}
    for name, low, high in bands:
        if low == 0.0 and high == 0.0:
            mask = radius == 0
        else:
            mask = radius > low
            if high is not None:
                mask &= radius <= high
        out[name] = mask
    return out


def batch_coefficients(field):
    from scipy.fft import dctn
    return dctn(np.asarray(field, dtype=np.float64), type=2, norm='ortho', axes=(-2, -1))


def band_record(ref_power, pred_power, err_power, total, mask):
    """Energy ratio, alignment and band error under the appendix validity rules."""
    tr = float(ref_power[mask].sum())
    pr = float(pred_power[mask].sum())
    er = float(err_power[mask].sum())
    record = dict(reference_energy=tr, prediction_energy=pr, error_energy=er, mode_count=int(mask.sum()),
                  reference_fraction=tr / total if total > 0.0 else None)
    if not tr > total * MIN_REFERENCE_FRACTION:
        return record
    record['energy_ratio'] = pr / tr
    record['band_relative_error'] = float(np.sqrt(er / tr))
    if pr > 0.0:
        alignment = (pr + tr - er) / (2.0 * np.sqrt(pr * tr))
        record['alignment'] = alignment
        record['matched_fraction'] = alignment ** 2
    return record


def chance_alignment(prediction, reference, mask, rng, pairs_per_sample=10):
    """Alignment when predictions are paired with references from other inputs."""
    index = np.flatnonzero(mask.ravel())
    if index.size < 2 or len(prediction) < 2:
        return None
    x = prediction[:, index]
    y = reference[:, index]
    norms_x = np.linalg.norm(x, axis=1)
    norms_y = np.linalg.norm(y, axis=1)
    rows = np.repeat(np.arange(len(x)), pairs_per_sample)
    columns = (rows + 1 + rng.integers(0, len(x) - 1, size=rows.size)) % len(x)
    values = []
    for start in range(0, rows.size, 256):
        block_rows = rows[start:start + 256]
        block_columns = columns[start:start + 256]
        denominator = norms_x[block_rows] * norms_y[block_columns]
        keep = denominator > 0.0
        values.append(np.einsum('ij,ij->i', x[block_rows][keep], y[block_columns][keep]) / denominator[keep])
    values = np.concatenate(values)
    return dict(chance_alignment_mean=float(values.mean()),
                chance_alignment_sd=float(values.std(ddof=1)),
                chance_alignment_min=float(values.min()),
                chance_alignment_max=float(values.max()),
                chance_pairs=int(values.size),
                chance_mode_count=int(index.size),
                chance_reciprocal_root_modes=float(1.0 / np.sqrt(index.size)))


def write_rows(path, rows):
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class ShellAccumulator:
    """Per-radius spectra and per-sample alignment, accumulated with exact sums."""

    def __init__(self, shape):
        self.radius = radii(shape, 'cosine')
        self.shell = np.floor(self.radius + 1e-12).astype(int)
        self.n_shells = int(self.shell.max()) + 1
        self.reference = np.zeros(self.n_shells)
        self.prediction = np.zeros(self.n_shells)
        self.error = np.zeros(self.n_shells)
        self.alignment = np.zeros(self.n_shells)
        self.alignment_square = np.zeros(self.n_shells)
        self.valid = np.zeros(self.n_shells, dtype=int)
        self.samples = 0
        self.modes = np.bincount(self.shell.ravel(), minlength=self.n_shells)
        self.reference_total = 0.0

    def add(self, reference_power, prediction_power, error_power, reference_total):
        aggregate = lambda values: np.bincount(self.shell.ravel(),
                                               weights=np.asarray(values, dtype=np.float64).ravel(),
                                               minlength=self.n_shells)
        reference = aggregate(reference_power)
        prediction = aggregate(prediction_power)
        error = aggregate(error_power)
        self.reference += reference
        self.prediction += prediction
        self.error += error
        positive = (reference > 0.0) & (prediction > 0.0)
        with np.errstate(invalid='ignore', divide='ignore'):
            alignment = (prediction + reference - error) / (2.0 * np.sqrt(prediction * reference))
        self.alignment[positive] += alignment[positive]
        self.alignment_square[positive] += alignment[positive] ** 2
        self.valid[positive] += 1
        self.samples += 1
        self.reference_total += reference_total

    def rows(self):
        out = []
        for shell in range(self.n_shells):
            valid = int(self.valid[shell])
            mean = self.alignment[shell] / valid if valid else None
            out.append(dict(radius=shell, mode_count=int(self.modes[shell]),
                            reference_power=self.reference[shell] / self.samples,
                            prediction_power=self.prediction[shell] / self.samples,
                            error_power=self.error[shell] / self.samples,
                            reference_fraction=self.reference[shell] / self.reference_total,
                            pooled_alignment=(self.prediction[shell] + self.reference[shell] - self.error[shell])
                                             / (2.0 * np.sqrt(self.prediction[shell] * self.reference[shell]))
                                             if self.prediction[shell] > 0.0 and self.reference[shell] > 0.0 else None,
                            sample_alignment_mean=mean,
                            sample_alignment_sd=float(np.sqrt(max(self.alignment_square[shell] / valid - mean ** 2,
                                                                  0.0))) if valid > 1 else None,
                            sample_alignment_valid=valid))
        return out


def chunk_paths(root, limit=None):
    tags = sorted(p for p in root.iterdir() if p.is_dir())
    assert len(tags) == 1, f'expected one ablation directory under {root}, found {len(tags)}'
    chunks = sorted(p for p in tags[0].iterdir() if (p / 'result.pt').is_file())
    assert chunks, f'no result.pt under {tags[0]}'
    return tags[0], chunks[:limit] if limit else chunks


def analyze_cell(torch, root, pde, task, collection, limit=None):
    tag, chunks = chunk_paths(root, limit)
    rng = np.random.default_rng(20260921)
    masks = band_masks(radii((128, 128), 'cosine'))
    shells = {field: ShellAccumulator((128, 128)) for field in FIELDS}
    stored_coefficients = {field: ([], []) for field in FIELDS}
    per_sample = []
    seen = {field: set() for field in FIELDS}
    integrity = {'relative_error_max_abs_deviation': 0.0, 'chunks': len(chunks), 'tag': tag.name,
                 'mask_observation_counts': {}}

    for chunk in chunks:
        result = torch.load(chunk / 'result.pt', map_location='cpu', weights_only=False)
        masks_file = result['masks']
        for field in FIELDS:
            key = 'coef' if field == 'coef' else 'sol'
            integrity['mask_observation_counts'][key] = float(masks_file[key].sum())
        stored = {}
        metric_path = chunk / 'metrics_per_sample.csv'
        if metric_path.is_file():
            stored = {row['sample_id']: row for row in csv.DictReader(metric_path.open())}
        for field in FIELDS:
            prediction = result[f'{field}_final'].numpy().astype(np.float64)[:, 0]
            reference = result[f'{field}_ground_truth'].numpy().astype(np.float64)[:, 0]
            sample_ids = [str(s) for s in result['ground_truth_metadata']['sample_ids']]
            assert prediction.shape == reference.shape == (len(sample_ids), 128, 128)
            pred_coefficients = batch_coefficients(prediction)
            ref_coefficients = batch_coefficients(reference)
            stored_coefficients[field][0].append(pred_coefficients.reshape(len(sample_ids), -1).astype(np.float32))
            stored_coefficients[field][1].append(ref_coefficients.reshape(len(sample_ids), -1).astype(np.float32))
            for row, sample_id in enumerate(sample_ids):
                assert sample_id not in seen[field], f'duplicate sample {sample_id} for {field} in {root}'
                seen[field].add(sample_id)
                c_hat = pred_coefficients[row]
                c = ref_coefficients[row]
                ref_power = c ** 2
                pred_power = c_hat ** 2
                err_power = (c_hat - c) ** 2
                total = float(ref_power.sum())
                full_error = float(np.sqrt(err_power.sum() / total))
                physical = float(np.sqrt(np.sum((prediction[row] - reference[row]) ** 2)
                                         / np.sum(reference[row] ** 2)))
                integrity['relative_error_max_abs_deviation'] = max(integrity['relative_error_max_abs_deviation'],
                                                                    abs(physical - full_error))
                if sample_id in stored:
                    assert abs(float(stored[sample_id][f'rel_l2_{field}']) - full_error) < 1e-4
                shells[field].add(ref_power, pred_power, err_power, total)
                for name, mask in masks.items():
                    record = band_record(ref_power, pred_power, err_power, total, mask)
                    record.update(pde=pde, task=task, collection=collection, field=field,
                                  role='target' if field in TARGET[task] else 'observed',
                                  sample_id=sample_id, band=name, full_relative_error=full_error)
                    per_sample.append(record)

    summary = []
    for field in FIELDS:
        prediction = np.concatenate(stored_coefficients[field][0])
        reference = np.concatenate(stored_coefficients[field][1])
        rows_by_band = defaultdict(list)
        for row in per_sample:
            if row['field'] == field:
                rows_by_band[row['band']].append(row)
        for name, mask in masks.items():
            rows = rows_by_band[name]
            energy_ratio = [r['energy_ratio'] for r in rows if 'energy_ratio' in r]
            alignment = [r['alignment'] for r in rows if 'alignment' in r]
            band_error = [r['band_relative_error'] for r in rows if 'band_relative_error' in r]
            reference_fraction = [r['reference_fraction'] for r in rows if 'reference_fraction' in r]
            entry = dict(pde=pde, task=task, collection=collection, field=field,
                         role='target' if field in TARGET[task] else 'observed', band=name,
                         examples=len(rows), mode_count=int(mask.sum()),
                         reference_fraction_mean=float(np.mean(reference_fraction)) if reference_fraction else None,
                         energy_ratio_mean=float(np.mean(energy_ratio)) if energy_ratio else None,
                         energy_ratio_sd=float(np.std(energy_ratio, ddof=1)) if len(energy_ratio) > 1 else None,
                         alignment_mean=float(np.mean(alignment)) if alignment else None,
                         alignment_sd=float(np.std(alignment, ddof=1)) if len(alignment) > 1 else None,
                         alignment_median=float(np.median(alignment)) if alignment else None,
                         matched_fraction_mean=float(np.mean([a ** 2 for a in alignment])) if alignment else None,
                         mismatched_fraction_mean=float(np.mean([1.0 - a ** 2 for a in alignment])) if alignment else None,
                         band_relative_error_mean=float(np.mean(band_error)) if band_error else None,
                         full_relative_error_mean=float(np.mean([r['full_relative_error'] for r in rows])))
            entry.update(chance_alignment(prediction, reference, mask, rng) or {})
            summary.append(entry)
    return dict(per_sample=per_sample, summary=summary, shells=shells, integrity=integrity,
                samples=len(seen['coef']), collection=collection, tag=tag.name)


def analyze_spatial(torch, root, pde, task, limit=None):
    from scipy.fft import idctn
    from scipy.ndimage import distance_transform_edt
    _, chunks = chunk_paths(root, limit)
    bins = np.asarray(DISTANCE_BINS)
    high = radii((128, 128), 'cosine') > HIGH_RADIUS
    accum = {field: dict(error=np.zeros(len(bins) - 1), reference=np.zeros(len(bins) - 1),
                         count=np.zeros(len(bins) - 1)) for field in FIELDS}
    for chunk in chunks:
        result = torch.load(chunk / 'result.pt', map_location='cpu', weights_only=False)
        for field in FIELDS:
            prediction = result[f'{field}_final'].numpy().astype(np.float64)[:, 0]
            reference = result[f'{field}_ground_truth'].numpy().astype(np.float64)[:, 0]
            observed = result['masks'][field].numpy().astype(np.float64)[:, 0]
            pred_coefficients = batch_coefficients(prediction)
            ref_coefficients = batch_coefficients(reference)
            error = idctn((pred_coefficients - ref_coefficients) * high, type=2, norm='ortho', axes=(-2, -1))
            truth = idctn(ref_coefficients * high, type=2, norm='ortho', axes=(-2, -1))
            for row in range(prediction.shape[0]):
                distance = np.clip(np.digitize(distance_transform_edt(observed[row] < 0.5), bins) - 1, 0, len(bins) - 2)
                accum[field]['error'] += np.bincount(distance.ravel(), weights=(error[row] ** 2).ravel(),
                                                     minlength=len(bins) - 1)
                accum[field]['reference'] += np.bincount(distance.ravel(), weights=(truth[row] ** 2).ravel(),
                                                         minlength=len(bins) - 1)
                accum[field]['count'] += np.bincount(distance.ravel(), minlength=len(bins) - 1)
    rows = []
    for field in FIELDS:
        for index in range(len(bins) - 1):
            pixels = accum[field]['count'][index]
            if pixels <= 0.0:
                continue
            rows.append(dict(pde=pde, task=task, field=field,
                             role='target' if field in TARGET[task] else 'observed',
                             distance_low=bins[index], distance_high=bins[index + 1], pixels=float(pixels),
                             high_band_error_energy=accum[field]['error'][index] / pixels,
                             high_band_reference_energy=accum[field]['reference'][index] / pixels))
    return rows


def analyze_ensemble(torch, root, pde='poisson'):
    from scipy.fft import dctn
    radius = radii((128, 128), 'cosine')
    masks = band_masks(radius)
    shell = np.floor(radius + 1e-12).astype(int)
    n_shells = int(shell.max()) + 1
    per_input, shell_rows, summaries = [], [], []
    for task in TASKS:
        case_root = root / 'cases' / task
        offsets = sorted(p.name for p in case_root.iterdir() if (p / 'pool.pt').is_file())
        rng = np.random.default_rng(20260921)
        deterministic_rows = {field: ([], []) for field in FIELDS}
        shell_sums = {(field, k): dict(reference=np.zeros(n_shells), prediction=np.zeros(n_shells),
                                       error=np.zeros(n_shells)) for field in FIELDS for k in KS}
        for offset in offsets:
            pool = torch.load(case_root / offset / 'pool.pt', map_location='cpu', weights_only=False)
            draws = pool['predictions'].numpy()
            truth = pool['fixed']['truth'][0].numpy().astype(np.float64)
            assert tuple(draws.shape[1:]) == (2, 128, 128)
            assert pool['fixed']['observed_fields'] == list(OBSERVED[task]), (task, pool['fixed']['observed_fields'])
            for channel, field in enumerate(FIELDS):
                reference = dctn(truth[channel], type=2, norm='ortho')
                ref_power = reference ** 2
                total = float(ref_power.sum())
                coefficients = dctn(draws[:, channel].astype(np.float64), type=2, norm='ortho', axes=(-2, -1))
                prefix, running = {}, np.zeros_like(reference)
                for draw in range(coefficients.shape[0]):
                    running += coefficients[draw]
                    if draw + 1 in KS:
                        prefix[draw + 1] = (running / (draw + 1)).copy()
                assert np.allclose(prefix[KS[-1]], coefficients.mean(axis=0), rtol=0.0, atol=1e-9)
                deterministic_rows[field][0].append(prefix[KS[-1]].reshape(-1).astype(np.float32))
                deterministic_rows[field][1].append(reference.reshape(-1).astype(np.float32))
                squared = coefficients ** 2
                squared_error = (coefficients - reference) ** 2
                for k in KS:
                    for name, mask in masks.items():
                        record = band_record(ref_power, prefix[k] ** 2, (prefix[k] - reference) ** 2, total, mask)
                        per_input.append(dict(pde=pde, task=task, field=field, offset=offset, band=name, K=k,
                                              observed=field in OBSERVED[task], **record))
                    bucket = shell_sums[field, k]
                    bucket['reference'] += np.bincount(shell.ravel(), weights=ref_power.ravel(), minlength=n_shells)
                    bucket['prediction'] += np.bincount(shell.ravel(), weights=(prefix[k] ** 2).ravel(),
                                                        minlength=n_shells)
                    bucket['error'] += np.bincount(shell.ravel(), weights=((prefix[k] - reference) ** 2).ravel(),
                                                   minlength=n_shells)
                for name, mask in masks.items():
                    index = np.flatnonzero(mask.ravel())
                    draw_prediction = squared[:, index].sum(axis=1)
                    draw_error = squared_error[:, index].sum(axis=1)
                    valid = (draw_prediction > 0.0) & (total * MIN_REFERENCE_FRACTION < ref_power[index].sum())
                    alignment = (draw_prediction[valid] + float(ref_power[index].sum()) - draw_error[valid]) \
                        / (2.0 * np.sqrt(draw_prediction[valid] * float(ref_power[index].sum())))
                    deterministic = band_record(ref_power, prefix[KS[-1]] ** 2, (prefix[KS[-1]] - reference) ** 2,
                                                total, mask)
                    retained = deterministic.get('energy_ratio')
                    stochastic = max(float(draw_prediction.mean()) - deterministic['prediction_energy'], 0.0) / total
                    summaries.append(dict(
                        pde=pde, task=task, field=field, band=name, offset=offset, observed=field in OBSERVED[task],
                        reference_fraction=deterministic.get('reference_fraction'),
                        draw_energy_ratio_mean=float(draw_prediction.mean() / deterministic['reference_energy']),
                        draw_energy_ratio_sd=float(draw_prediction.std(ddof=1) / deterministic['reference_energy']),
                        draw_alignment_mean=float(alignment.mean()) if alignment.size else None,
                        draw_alignment_sd=float(alignment.std(ddof=1)) if alignment.size > 1 else None,
                        deterministic_energy_ratio=retained,
                        deterministic_alignment=deterministic.get('alignment'),
                        deterministic_matched_fraction=deterministic.get('matched_fraction'),
                        stochastic_energy_ratio=stochastic,
                        deterministic_share=retained / (retained + stochastic)
                        if retained is not None and retained + stochastic > 0.0 else None))
        for field in FIELDS:
            prediction = np.concatenate(deterministic_rows[field][0])
            reference = np.concatenate(deterministic_rows[field][1])
            for name, mask in masks.items():
                rows = [r for r in summaries if r['task'] == task and r['field'] == field and r['band'] == name]
                entry = dict(pde=pde, task=task, field=field, band=name,
                             observed=field in OBSERVED[task], inputs=len(rows), mode_count=int(mask.sum()))
                for key in ('reference_fraction', 'draw_energy_ratio_mean', 'draw_energy_ratio_sd',
                            'draw_alignment_mean', 'draw_alignment_sd', 'deterministic_energy_ratio',
                            'deterministic_alignment', 'deterministic_matched_fraction',
                            'stochastic_energy_ratio', 'deterministic_share'):
                    values = [r[key] for r in rows if r[key] is not None]
                    entry[key] = float(np.mean(values)) if values else None
                entry.update(chance_alignment(prediction, reference, mask, rng) or {})
                shell_rows.append(entry)
        for field in FIELDS:
            for k in KS:
                bucket = shell_sums[field, k]
                for index in range(n_shells):
                    shell_rows.append(dict(pde=pde, task=task, field=field, band=f'shell_K{k}', K=k, radius=index,
                                           mode_count=int((shell == index).sum()),
                                           reference_power=bucket['reference'][index] / len(offsets),
                                           prediction_power=bucket['prediction'][index] / len(offsets),
                                           error_power=bucket['error'][index] / len(offsets)))
    return per_input, shell_rows


def validate_matched(torch, study, pde):
    """Reproduce the published 8/32 tables: metrics averaged over seeds, then inputs."""
    protocol = json.loads((study / 'inputs_v2/protocol.json').read_text())
    ids, seeds = protocol['evaluation_ids'], protocol['seeds']
    records = defaultdict(list)
    for sample in ids:
        for seed in seeds:
            tensor = torch.load(study / 'results_v3' / pde / 'fm' / 'n100' / f'seed{seed}' / f'sample{sample}'
                                / 'prediction.pt', map_location='cpu', weights_only=False)
            record = spectral_record(tensor['prediction'][0, 0].double().numpy(),
                                     tensor['truth'][0, 0].double().numpy(), 'cosine', (8.0, 32.0))
            total = sum(record['bands'][b]['reference_energy'] for b in ['dc', 'low', 'mid', 'high'])
            for band, values in record['bands'].items():
                ratio = values['predicted_reference_energy_ratio']
                records[sample, band].append(dict(
                    reference_fraction=values['reference_energy'] / total,
                    energy_ratio=ratio,
                    alignment=(values['prediction_energy'] + values['reference_energy'] - values['error_energy'])
                              / (2.0 * np.sqrt(values['prediction_energy'] * values['reference_energy']))
                              if ratio is not None and values['prediction_energy'] > 0.0 else None))
    out = {}
    for band in ['dc', 'low', 'mid', 'high']:
        summary = {}
        for key in ('energy_ratio', 'alignment', 'reference_fraction'):
            per_input = []
            for sample in ids:
                values = [r[key] for r in records[sample, band]]
                if any(v is None for v in values):
                    per_input = None
                    break
                per_input.append(float(np.mean(values)))
            if not per_input:
                summary[key] = dict(examples=0)
                continue
            interval = paired_bootstrap(np.asarray(per_input))
            summary[key] = dict(examples=len(per_input), mean=float(np.mean(per_input)),
                                sd=float(np.std(per_input, ddof=1)),
                                ci_low=float(interval[0]), ci_high=float(interval[1]))
        out[band] = summary
    return out


def run_main(args):
    import torch
    torch.set_num_threads(args.threads)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    per_sample, summary, radial, spatial, integrity = [], [], [], [], []
    for pde in args.pdes:
        for task in TASKS:
            if args.cells and f'{pde}_{task}' not in args.cells:
                continue
            collections = [SELECTION[pde, task]]
            if args.all_collections and SELECTION[pde, task] != 'MAIN1000_100_TEST_id':
                collections.append('MAIN1000_100_TEST_id')
            for collection in collections:
                root = Path(args.main_root) / collection / pde / task
                print(f'[{pde}/{task}] {collection}', flush=True)
                result = analyze_cell(torch, root, pde, task, collection, args.limit_chunks)
                per_sample.extend(result['per_sample'])
                summary.extend(result['summary'])
                integrity.append(dict(pde=pde, task=task, collection=collection, samples=result['samples'],
                                      **result['integrity']))
                for field, accumulator in result['shells'].items():
                    for row in accumulator.rows():
                        radial.append(dict(pde=pde, task=task, collection=collection, field=field,
                                           role='target' if field in TARGET[task] else 'observed', **row))
                if args.spatial and not args.limit_chunks:
                    spatial.extend(analyze_spatial(torch, root, pde, task))
                print(f'  samples={result["samples"]} rows={len(result["per_sample"])}', flush=True)
    write_rows(output / 'main_per_sample.csv', per_sample)
    write_rows(output / 'main_band_summary.csv', summary)
    write_rows(output / 'main_radial.csv', radial)
    write_rows(output / 'main_spatial.csv', spatial)
    return integrity


def run_ensemble(args):
    import torch
    torch.set_num_threads(args.threads)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    per_input, shell_rows = analyze_ensemble(torch, Path(args.conditional_root))
    write_rows(output / 'ensemble_per_input.csv', per_input)
    write_rows(output / 'ensemble_band_summary.csv',
               [row for row in shell_rows if not row['band'].startswith('shell_K')])
    write_rows(output / 'ensemble_radial.csv', [row for row in shell_rows if row['band'].startswith('shell_K')])
    return per_input


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['main', 'ensemble', 'validate', 'self-check'], default='main')
    parser.add_argument('--main-root', type=Path)
    parser.add_argument('--conditional-root', type=Path)
    parser.add_argument('--study', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--pdes', nargs='+', default=['poisson', 'darcy'])
    parser.add_argument('--cells', nargs='+')
    parser.add_argument('--limit-chunks', type=int)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--spatial', action='store_true')
    parser.add_argument('--all-collections', action='store_true')
    args = parser.parse_args()
    if args.mode == 'self-check':
        print(json.dumps(self_check(), indent=2))
        return
    if args.mode == 'validate':
        import torch
        assert args.study and args.output
        args.output.mkdir(parents=True, exist_ok=True)
        report = {pde: validate_matched(torch, args.study, pde) for pde in args.pdes}
        report['script_sha256'] = sha(Path(__file__))
        write(args.output / 'matched_validation.json', report)
        print(json.dumps(report, indent=2))
        return
    if args.mode == 'main':
        assert args.main_root and args.output
        integrity = run_main(args)
        write(args.output / 'main_integrity.json', dict(script_sha256=sha(Path(__file__)), cells=integrity))
        return
    assert args.conditional_root and args.output
    per_input = run_ensemble(args)
    write(args.output / 'ensemble_manifest.json',
          dict(script_sha256=sha(Path(__file__)), rows=len(per_input)))


if __name__ == '__main__':
    main()
