"""Frozen physical inputs and observation masks for the 2026-09-15 rerun.

This module never samples a model. ``load_cell`` uses saved masks only; its
only mask transformations are selecting the observed field and full coverage.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

GRID = 128
COUNT = 1000
TASKS = {'sparse_forward': 'forward', 'sparse_inverse': 'inverse',
         'sparse_joint': 'both', 'full_forward': 'forward', 'full_inverse': 'inverse'}


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def tensor_sha256(value: Any) -> str:
    """Historical audit convention: float32, shape prefix, row-major bytes."""
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    a = np.asarray(value, dtype='<f4')
    return hashlib.sha256(str(a.shape).encode() + a.tobytes(order='C')).hexdigest()


def supervised_mask(sample_id: str) -> torch.Tensor:
    """Archived sensor contract v3, test split, sensor seed 1, epoch 0."""
    payload = f'mask|1|test|{sample_id}|0'.encode('utf-8')
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big') % (2**63 - 1)
    gen = torch.Generator(device='cpu').manual_seed(seed)
    mask = torch.zeros(GRID * GRID, dtype=torch.uint8)
    mask[torch.randperm(GRID * GRID, generator=gen)[:500]] = 1
    return mask.reshape(1, GRID, GRID)


def diffusion_mask(seed: int) -> torch.Tensor:
    """Same legacy RandomState sequence as np.random.seed + np.random.choice."""
    rng = np.random.RandomState(seed)
    mask = np.zeros(GRID * GRID, dtype=np.uint8)
    mask[rng.choice(GRID * GRID, 500, replace=False)] = 1
    return torch.from_numpy(mask.reshape(1, 1, GRID, GRID))


@lru_cache(maxsize=128)
def _verify_file(path: str, expected: str, size: int, mtime_ns: int) -> None:
    del size, mtime_ns
    if file_sha256(path) != expected:
        raise ValueError(f'Frozen input hash mismatch: {path}')


@lru_cache(maxsize=2)
def _read_tensor_file(path: str, expected: str, size: int, mtime_ns: int) -> Any:
    del expected, size, mtime_ns
    return torch.load(path, map_location='cpu', weights_only=False)


def _read(root: Path, name: str, expected: str, verify: bool) -> Any:
    path = Path(name)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    stat = path.stat()
    args = (str(path), expected, stat.st_size, stat.st_mtime_ns)
    if verify:
        _verify_file(*args)
    return _read_tensor_file(*args)


def load_cell(input_root: str | Path, cell: dict[str, Any], *,
              verify_hashes: bool = True) -> dict[str, Any]:
    """Return CPU truth, exact final masks, IDs and metadata for one cell.

    ``truth_file`` and ``masks_file`` are relative to ``input_root`` unless
    absolute. Saved masks have singleton or 1,000-row batch dimensions.
    Full-field masks and disabled fields are deterministic task semantics.
    No random-number function is called here. Returned truth tensors should
    be treated as immutable; callers send only truth * mask to guidance.
    """
    root = Path(input_root)
    truth = _read(root, cell['truth_file'], cell['truth_sha256'], verify_hashes)
    saved = _read(root, cell['masks_file'], cell['masks_sha256'], verify_hashes)
    n = int(cell.get('count', COUNT))
    a, u = truth['coef_ground_truth'], truth['sol_ground_truth']
    for value in [a, u]:
        if value.shape != (n, 1, GRID, GRID) or value.dtype != torch.float32:
            raise ValueError(f'Invalid physical tensor in {cell["cell_id"]}')
    for key in ['pde', 'distribution', 'cohort']:
        if truth[key] != cell[key] or saved[key] != cell[key]:
            raise ValueError(f'Input identity mismatch for {key}: {cell["cell_id"]}')
    if saved['sample_ids'] != truth['sample_ids'] or len(set(truth['sample_ids'])) != n:
        raise ValueError('Mask and physical-input IDs differ or are duplicated')
    if not torch.equal(truth['source_indices'], torch.arange(n, dtype=torch.int64)):
        raise ValueError('Expected complete physical indices 0..999')
    setting = cell['setting']
    task = cell['task']
    if setting not in TASKS or TASKS[setting] != task:
        raise ValueError(f'Inconsistent task/setting: {task}/{setting}')
    expected_observed = (['sol'] if cell['pde'] == 'burger' else
                         ['coef'] if task == 'forward' else ['sol'] if task == 'inverse' else ['coef', 'sol'])
    if cell.get('observed_fields', expected_observed) != expected_observed:
        raise ValueError('Observed-field declaration disagrees with task')
    full = setting.startswith('full_')
    if int(cell.get('num_obs', 16384 if full else 500)) != (16384 if full else 500):
        raise ValueError('Observation budget disagrees with setting')
    masks = {}
    for key in ['coef', 'sol']:
        mask = saved[key]
        if mask.shape not in [(1, 1, GRID, GRID), (n, 1, GRID, GRID)]:
            raise ValueError(f'Unexpected stored mask shape: {mask.shape}')
        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError('Observation mask is not binary')
        if not torch.all(mask.flatten(1).sum(1) == 500):
            raise ValueError('Stored sparse mask must contain exactly 500 points per field')
        if key not in expected_observed:
            mask = torch.zeros((1, 1, GRID, GRID), dtype=torch.uint8)
        elif full:
            mask = torch.ones((1, 1, GRID, GRID), dtype=torch.uint8)
        masks[key] = mask.expand(n, -1, -1, -1)
    masks['metadata'] = dict(saved.get('metadata', {}), setting=setting,
                             observed_fields=expected_observed,
                             final_count_per_observed_field=16384 if full else 500,
                             observed_scalar_values=(16384 if full else 500)*len(expected_observed),
                             source_file=cell['masks_file'], source_sha256=cell['masks_sha256'])
    return dict(coef_ground_truth=a, sol_ground_truth=u, masks=masks,
                sample_ids=list(truth['sample_ids']), source_indices=truth['source_indices'],
                pde_params=dict(truth['pde_params']), metadata=dict(truth['metadata']),
                pde=truth['pde'], distribution=truth['distribution'], cohort=truth['cohort'],
                cell_id=cell['cell_id'])
