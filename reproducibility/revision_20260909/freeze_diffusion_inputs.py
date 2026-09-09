#!/usr/bin/env python3
"""Freeze native-precision test slices used by the original Diffusion comparison.

Original MAT files and results are read only. The output is a derived NPZ with
the original field names/axes and exact dtypes, never a replacement MAT file.
No model is instantiated and CUDA is not initialized.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import re
import subprocess
import time
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def array_info(array):
    import numpy as np
    a = np.ascontiguousarray(array)
    return dict(shape=list(a.shape), dtype=str(a.dtype), sha256=hashlib.sha256(a.tobytes(order='C')).hexdigest())


def jsonable(value):
    import numpy as np
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='replace')
    return value


def read_slices(path, pde):
    import h5py
    import numpy as np
    import scipy.io
    arrays, desc, attrs = {}, {}, {}
    if pde in {'darcy', 'nsnonbounded'}:
        with h5py.File(path, 'r') as source:
            attrs = {str(k): jsonable(v) for k, v in source.attrs.items()}
            keys = ['thresh_a_data', 'thresh_p_data'] if pde == 'darcy' else ['w0', 'w']
            for key in keys:
                ds = source[key]
                if pde == 'darcy':
                    assert ds.shape == (128, 128, 10000)
                    arrays[key] = ds[:, :, :1000]
                    selection = ':,:,0:1000'
                elif key == 'w0':
                    assert ds.shape == (10000, 128, 128)
                    arrays[key] = ds[:1000]
                    selection = '0:1000,:,:'
                else:
                    assert ds.shape == (10000, 128, 128, 10)
                    arrays[key] = ds[:1000, :, :, -1:]
                    selection = '0:1000,:,:,-1: (final frame only)'
                desc[key] = dict(original_shape=list(ds.shape), original_dtype=str(ds.dtype), selection=selection, axes='unchanged')
            for key in source.keys():
                ds = source[key]
                if isinstance(ds, h5py.Dataset) and key not in keys and ds.size <= 10000:
                    value = ds[()]
                    arrays[key] = np.asarray(value).copy()
                    desc[key] = dict(original_shape=list(ds.shape), original_dtype=str(ds.dtype), selection='all (small metadata)', axes='unchanged')
    else:
        keys = {'poisson': ['f_data', 'phi_data'], 'helmholtz': ['f_data', 'psi_data'], 'burger': ['input', 'output']}[pde]
        variables = scipy.io.whosmat(path)
        small = [name for name, shape, _ in variables if np.prod(shape) <= 10000 and name not in keys]
        source = scipy.io.loadmat(path, variable_names=keys + small)
        for key in keys + small:
            value = source[key]
            if key in keys:
                assert value.shape[0] == 10000, (key, value.shape)
                arrays[key] = value[:1000].copy()
                selection = '0:1000,...'
            else:
                arrays[key] = value.copy()
                selection = 'all (small metadata)'
            desc[key] = dict(original_shape=list(value.shape), original_dtype=str(value.dtype), selection=selection, axes='unchanged')
    for key, value in arrays.items():
        assert value.dtype.kind != 'O', (key, value.dtype)
        if value.dtype.kind in 'fc':
            assert np.isfinite(value).all(), key
    return arrays, desc, attrs


class CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            import torch
            return lambda value: torch.load(io.BytesIO(value), map_location='cpu', weights_only=False)
        return super().find_class(module, name)


def numpy(value):
    import numpy as np
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def check_saved_results(entry, arrays, result_root):
    import numpy as np
    records = []
    for cell in entry['selected_cells']:
        root = result_root / cell['study']
        pde, task = entry['pde'], cell['task']
        folder = root / 'samples' / pde / task
        paths = list(folder.glob('*_results.pkl'))
        indexed = {}
        for path in paths:
            match = re.search(r'_(\d+)(?:_obs|_results)', path.name)
            assert match, path
            index = int(match.group(1))
            assert index not in indexed
            indexed[index] = path
        assert sorted(indexed) == list(range(1000))
        for index in [0, 17, 999]:
            path = indexed[index]
            with path.open('rb') as stream:
                payload = CPUUnpickler(stream).load()
            assert 'obs_index' in payload
            masks = {k: numpy(v) for k, v in payload['obs_index'].items()}
            for name, mask in masks.items():
                assert mask.shape == (128, 128), (path, name, mask.shape)
                assert np.isin(mask, [0, 1]).all(), (path, name)
            if pde == 'burger':
                targets = {'u': arrays['output'][index]}
                predictions = {'u': numpy(payload['x_final']).reshape(128, 128)}
            else:
                names = {'poisson': ('f_data', 'phi_data'), 'helmholtz': ('f_data', 'psi_data'),
                         'darcy': ('thresh_a_data', 'thresh_p_data'), 'nsnonbounded': ('w0', 'w')}[pde]
                targets = {}
                for field, name in zip(['a', 'u'], names):
                    targets[field] = arrays[name][:, :, index] if pde == 'darcy' else arrays[name][index]
                if pde == 'nsnonbounded':
                    targets['u'] = targets['u'][:, :, -1]
                predictions = {'a': numpy(payload['coef_final']).reshape(128, 128),
                               'u': numpy(payload['sol_final']).reshape(128, 128)}
            metric_path = root / 'metrics' / pde / task / (path.stem + '_metrics_final.json')
            metric = json.loads(metric_path.read_text())
            checks = {}
            for field, truth in targets.items():
                truth = np.asarray(truth, dtype=np.float64)
                pred = np.asarray(predictions[field], dtype=np.float64)
                value = float(np.linalg.norm(pred - truth) / np.linalg.norm(truth))
                recorded = float(metric['rel_l2_' + field])
                assert np.isclose(value, recorded, rtol=2e-7, atol=1e-10), (path, field, value, recorded)
                checks[field] = dict(recomputed=value, recorded=recorded, absolute_difference=abs(value - recorded))
                mask_name = 'known_sensor' if pde == 'burger' else 'known_index_' + field
                observed_key = 'obs_rel_l2_' + field
                if observed_key in metric and metric[observed_key] is not None:
                    mask = masks[mask_name].astype(np.float64)
                    observed = float(np.linalg.norm((pred - truth) * mask) / np.linalg.norm(truth * mask))
                    recorded_obs = float(metric[observed_key])
                    assert np.isclose(observed, recorded_obs, rtol=2e-7, atol=1e-10), (path, field, observed, recorded_obs)
                    checks[field]['observed_recomputed'] = observed
                    checks[field]['observed_recorded'] = recorded_obs
            records.append(dict(study=cell['study'], pde=pde, task=task, index=index,
                                result_path=str(path), result_sha256=sha(path), metrics_path=str(metric_path),
                                metrics_sha256=sha(metric_path), field_errors=checks,
                                observation_masks={key: array_info(value) for key, value in masks.items()},
                                observation_counts={key: int(value.sum()) for key, value in masks.items()}))
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--result-root', type=Path, required=True)
    ap.add_argument('--diffusion-code', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    import numpy as np
    import scipy
    import h5py
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    assert not torch.cuda.is_initialized()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    plan = json.loads(args.plan.read_text())
    report = dict(schema_version='diffusion-native-frozen-input-v1', status='running',
                  plan_sha256=sha(args.plan), freeze_script_sha256=sha(__file__),
                  freeze_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).parent, text=True).strip(),
                  current_diffusion_checkout_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.diffusion_code, text=True).strip(),
                  original_producer_commit='Not recorded in original pickle/config/metric records; current checkout is not asserted as original run HEAD.',
                  environment=dict(python=os.sys.version, numpy=np.__version__, scipy=scipy.__version__, h5py=h5py.__version__, torch=torch.__version__), entries=[])
    started = time.time()
    for entry in plan['entries']:
        pde = entry['pde']
        source = Path(entry['source_mat'])
        before = (source.stat().st_size, source.stat().st_mtime_ns)
        assert before[0] == entry['source_bytes']
        print('Hash/read', pde, source, flush=True)
        source_sha = sha(source)
        arrays, descriptions, attrs = read_slices(source, pde)
        path = args.output / (pde + '_native_0_999.npz')
        np.savez(path, **arrays)
        with np.load(path, allow_pickle=False) as loaded:
            assert set(loaded.files) == set(arrays)
            for key in loaded.files:
                assert loaded[key].dtype == arrays[key].dtype and np.array_equal(loaded[key], arrays[key])
        reread, descriptions2, attrs2 = read_slices(source, pde)
        assert descriptions == descriptions2 and attrs == attrs2
        for key, value in arrays.items():
            assert value.dtype == reread[key].dtype and np.array_equal(value, reread[key]), key
        del reread
        evidence = check_saved_results(entry, arrays, args.result_root)
        assert before == (source.stat().st_size, source.stat().st_mtime_ns)
        row = dict(pde=pde, cache_path=path.name, cache_bytes=path.stat().st_size, cache_sha256=sha(path),
                   source_mat_path=str(source), source_mat_sha256=source_sha, source_mat_bytes=before[0],
                   source_indices=list(range(1000)), arrays={key: array_info(value) for key, value in arrays.items()},
                   source_arrays=descriptions, source_attributes=attrs, source_slice_reread_exact=True,
                   save_reload_exact=True, selected_cells=entry['selected_cells'], saved_result_checks=evidence)
        report['entries'].append(row)
        (args.output / 'manifest.in_progress.json').write_text(json.dumps(report, indent=2) + '\n')
        print('PASS', pde, row['cache_bytes'], 'bytes;', len(evidence), 'saved-result checks', flush=True)
        del arrays
    assert not torch.cuda.is_initialized()
    report.update(status='pass', complete=True, elapsed_seconds=time.time() - started,
                  cache_bytes=sum(x['cache_bytes'] for x in report['entries']),
                  saved_results_checked=sum(len(x['saved_result_checks']) for x in report['entries']), cuda_initialized=False)
    (args.output / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ['status', 'cache_bytes', 'saved_results_checked', 'elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
