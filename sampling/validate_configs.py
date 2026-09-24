"""Validate all published sampling profiles and ablation combinations on CPU."""
import argparse
from collections import Counter
from pathlib import Path
from argparse import Namespace

from sampling.config import VALID_PDES, load_config, load_yaml_file
from experiments.paper.run import resolved_jobs


def validate(root, check_assets=False):
    expected = {(pde, task) for pde in VALID_PDES
                for task in (['both'] if pde == 'burger' else ['forward','inverse','both'])}
    missing_assets = set()
    counts = {}
    for folder in ['configs/main']:
        found = set()
        checkpoints = {}
        for path in sorted((root/folder).rglob('*.yaml')):
            cfg = load_config(path)
            assert path.stem == cfg.pde and path.parent.name == cfg.task, path
            found.add((cfg.pde, cfg.task))
            assert {'id', 'smooth', 'rough'} <= set(cfg.data_paths), path
            assert cfg.pde_guidance_clock == 'step_fraction', path
            assert cfg.hermite_include_integral_residual is False, path
            assert cfg.checkpoint_path, path
            checkpoints.setdefault(cfg.pde, set()).add(cfg.checkpoint_path)
            assert cfg.allow_synthetic_data is False, path
            if cfg.pde == 'nsnonbounded':
                assert cfg.model_profile == 'light' and cfg.residual_mode == 'endpoint_secant', path
            if check_assets:
                missing_assets.update(p for p in [cfg.checkpoint_path, *cfg.data_paths.values()]
                                      if not Path(p).is_file())
        assert found == expected, (folder, 'missing', expected-found, 'extra', found-expected)
        assert all(len(paths) == 1 for paths in checkpoints.values()), checkpoints
        counts[folder] = len(found)
    grid_counts = Counter()
    args = Namespace(pdes=None, limit=None, override=[], device='cpu', output=Path('/tmp/fm4pde-config-validation'))
    for path in (root/'configs/experiments').rglob('*.yaml'):
        spec = load_yaml_file(path)
        assert (root/'scripts/sample'/path.parent.name/(path.stem+'.sh')).is_file(), path
        if spec['engine'] == 'sampling':
            for job, cfg in resolved_jobs(spec, args):
                grid_counts[path.stem] += 1
                if check_assets:
                    missing_assets.update(p for p in [cfg.checkpoint_path,cfg.data_path] if not Path(p).is_file())
        else:
            assert spec['engine'] in {'averaging','architecture','timing','traces'}, path
    training = load_yaml_file(root/'configs/training_data.yaml')['train_files']
    assert set(training) == VALID_PDES
    assert all(len(paths) == 5 and len(set(paths)) == 5 for paths in training.values())
    return counts, grid_counts, sorted(missing_assets)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-assets', action='store_true', help='Also require local data and weights.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    counts, groups, missing = validate(root, args.check_assets)
    print('Sampling profiles:', dict(counts))
    print('Ablation jobs:', sum(groups.values()), dict(groups))
    print('Training manifests: 11 PDEs, five shards each.')
    if missing:
        for path in missing: print('Missing asset:', path)
        raise SystemExit(1)
    print('Configuration validation passed.')


if __name__ == '__main__':
    main()
