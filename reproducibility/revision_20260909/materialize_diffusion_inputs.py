#!/usr/bin/env python3
"""Restore compact native input files and runnable configs from verified NPZs.

The generated MAT/HDF5 files are derived evaluation subsets, with their own
hashes. They are not the original MAT files. No sampler or model is executed.
The original generator's data-loading statements validate the derived files.
"""
import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

MODULES = dict(poisson='poisson', darcy='darcy', helmholtz='helmholtz',
               nsnonbounded='ns_nonbounded', burger='burgers')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def array_record(array):
    import numpy as np
    return dict(shape=list(array.shape), dtype=str(array.dtype),
                sha256=hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest())


def load_cache(path, entry):
    import numpy as np
    assert entry['source_indices'] == list(range(1000))
    assert entry['save_reload_exact'] and entry['source_slice_reread_exact']
    assert path.stat().st_size == entry['cache_bytes']
    assert sha(path) == entry['cache_sha256'], path
    with np.load(path, allow_pickle=False) as source:
        assert set(source.files) == set(entry['arrays'])
        arrays = {key: source[key] for key in source.files}
    for key, array in arrays.items():
        assert array_record(array) == entry['arrays'][key], key
    return arrays


def write_native(path, pde, arrays):
    import h5py
    import numpy as np
    import scipy.io
    if path.exists():
        raise FileExistsError(path)
    if pde in ('darcy', 'nsnonbounded'):
        with h5py.File(path, 'x') as target:
            for key, value in arrays.items():
                target.create_dataset(key, data=value)
        with h5py.File(path, 'r') as source:
            restored = {key: source[key][()] for key in source}
    else:
        with path.open('xb') as stream:
            scipy.io.savemat(stream, arrays, do_compression=False)
        restored = scipy.io.loadmat(path, variable_names=list(arrays))
    for key, value in arrays.items():
        assert restored[key].dtype == value.dtype and np.array_equal(restored[key], value), key
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size,
                format='HDF5 datasets' if pde in ('darcy', 'nsnonbounded') else 'MAT v5',
                all_arrays_exact=True)


def native_loader(source_path, pde):
    """Use the original statements up to the network-loading boundary."""
    import h5py
    import numpy as np
    import scipy.io
    import torch
    tree = ast.parse(source_path.read_text())
    original = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'generate_' + MODULES[pde])
    stop = next(i for i, n in enumerate(original.body) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'batch_size' for t in n.targets))
    function = copy.deepcopy(original)
    function.name = 'load_native_fields'
    result = 'return init_state, ground_truth' if pde == 'burger' else 'return a_GT, u_GT'
    function.body = copy.deepcopy(original.body[:stop]) + ast.parse(result).body
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    scope = dict(h5py=h5py, np=np, scipy=__import__('scipy'), torch=torch)
    exec(compile(module, str(source_path) + '[data-loading-only]', 'exec'), scope)
    return scope['load_native_fields'], ast.unparse(module) + '\n'


def expected_fields(arrays, pde, index):
    import numpy as np
    if pde == 'burger':
        values = arrays['input'], arrays['output'][index]
    elif pde == 'darcy':
        values = arrays['thresh_a_data'][:, :, index], arrays['thresh_p_data'][:, :, index]
    elif pde == 'nsnonbounded':
        values = arrays['w0'][index], arrays['w'][index, :, :, -1]
    else:
        values = arrays['f_data'][index], arrays['phi_data' if pde == 'poisson' else 'psi_data'][index]
    return tuple(np.asarray(x, dtype=np.float64) for x in values)


def source_identity(root, expected):
    actual = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    assert actual == expected, (actual, expected)
    tracked = subprocess.check_output(['git', '-C', str(root), 'ls-tree', '-rz', 'HEAD'])
    for row in tracked.split(b'\0'):
        if not row:
            continue
        description, filename = row.split(b'\t', 1)
        mode, kind, digest = description.split()
        assert kind == b'blob' and mode in (b'100644', b'100755')
        path = root / os.fsdecode(filename)
        data = path.read_bytes()
        observed = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        assert observed == digest.decode(), path
    return actual


def main(args):
    import numpy as np
    import scipy
    import h5py
    import torch
    import yaml
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    assert not torch.cuda.is_initialized()
    assert sha(args.manifest) == args.manifest_sha256
    manifest = json.loads(args.manifest.read_text())
    assert manifest['status'] == 'pass' and manifest['complete']
    assert {e['pde'] for e in manifest['entries']} == set(MODULES)
    assert len(manifest['entries']) == 5
    source_commit = source_identity(args.diffusion_code, manifest['current_diffusion_checkout_commit'])
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    (args.output / 'inputs').mkdir()
    (args.output / 'configs').mkdir()
    (args.output / 'native_data_loaders').mkdir()
    report = dict(status='running', original_mat_files_recreated=False,
                  manifest_sha256=args.manifest_sha256, source_commit=source_commit,
                  original_producer_commit=manifest['original_producer_commit'],
                  script_sha256=sha(__file__), source_files_unchanged=True,
                  environment=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                                   h5py=h5py.__version__, torch=torch.__version__, cpu_threads=2),
                  entries=[], commands=[], models_executed=0)
    before = {args.manifest: sha(args.manifest)}
    checked_weights = {}
    for entry in manifest['entries']:
        pde = entry['pde']
        cache = args.inputs / entry['cache_path']
        arrays = load_cache(cache, entry)
        before[cache] = entry['cache_sha256']
        generated = write_native(args.output / 'inputs' / (pde + '_evaluation_0_999.mat'), pde, arrays)
        source = args.diffusion_code / 'scripts' / ('generate_' + MODULES[pde] + '.py')
        loader, loader_code = native_loader(source, pde)
        (args.output / 'native_data_loaders' / (pde + '.py')).write_text(loader_code)
        validations = []
        for cell_index, cell in enumerate(entry['selected_cells']):
            original = args.original_results / cell['study'] / '.sample_sweep/configs' / Path(cell['config']).name
            before[original] = sha(original)
            config = yaml.safe_load(original.read_text())
            assert config['data']['datapath'] == entry['source_mat_path']
            assert config['generate']['batch_size'] == 1
            assert config['test']['iterations'] == int(cell['study'].rsplit('_', 1)[1])
            weight = args.weights / Path(config['test']['pre-trained']).name
            if weight not in checked_weights:
                checked_weights[weight] = sha(weight)
            before[weight] = checked_weights[weight]
            updated = copy.deepcopy(config)
            updated['data']['datapath'] = generated['path']
            updated['test']['pre-trained'] = str(weight)
            out = args.output / 'new_results' / cell['study'] / pde
            (out / cell['task']).mkdir(parents=True)
            updated['output']['file_path'] = str(out)
            generated_config = args.output / 'configs' / (cell['study'] + '_' + Path(cell['config']).name)
            generated_config.write_text(yaml.safe_dump(updated, sort_keys=False))
            # One load per PDE and representative index; every stored array was
            # already reread in full above. Numerical settings are copied unchanged.
            if cell_index == 0:
                for index in (0, 17, 999):
                    validation_config = copy.deepcopy(updated)
                    validation_config['generate']['device'] = 'cpu'
                    validation_config['data']['offset'] = index
                    observed = loader(validation_config)
                    expected = expected_fields(arrays, pde, index)
                    for a, b in zip(observed, expected):
                        assert a.device.type == 'cpu' and a.dtype == torch.float64
                        assert np.array_equal(a.numpy(), b), (pde, index)
                    validations.append(dict(index=index, original_native_loader_exact=True))
            report['commands'].append(dict(study=cell['study'], pde=pde, task=cell['task'],
                cwd=str(args.diffusion_code), config=str(generated_config), config_sha256=sha(generated_config),
                original_config=str(original), original_config_sha256=before[original],
                selected_checkpoint=str(weight), checkpoint_sha256=checked_weights[weight],
                argv=[sys.executable, 'generate_pde.py', '--config', str(generated_config),
                      '--problem', cell['task'], '--batch', '1000', '--start_offset', '0',
                      '--step_size', str(config['test']['iterations'])]))
        report['entries'].append(dict(pde=pde, source_mat_path=entry['source_mat_path'],
            source_mat_sha256=entry['source_mat_sha256'], cache_sha256=entry['cache_sha256'],
            derived_native_file=generated, original_generator_sha256=sha(source),
            native_loader_checks=validations, arrays=entry['arrays']))
        print('PASS', pde, 'native input materialization and data loader', flush=True)
    assert len(report['commands']) == 26
    assert not torch.cuda.is_initialized()
    for path, expected in before.items():
        assert sha(path) == expected, path
    source_identity(args.diffusion_code, source_commit)
    report.update(status='pass', input_files=5, configurations=26, native_loader_checks=15,
                  cuda_initialized=False)
    (args.output / 'materialization_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PASS: five derived native input files; 26 runnable configurations; no model inference.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'inputs', 'original-results', 'weights', 'diffusion-code', 'output'):
        parser.add_argument('--' + name, type=lambda value: Path(value).resolve(), required=True)
    parser.add_argument('--manifest-sha256', required=True)
    main(parser.parse_args())
