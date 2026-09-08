"""Compare independent batched Burgers inference with the frozen native sampler.

This is an implementation/timing pilot only. It writes no formal evaluation
receipts. Each row retains its original initial-noise seed, observation mask,
Heun steps, float64 state and residuals, and per-input guidance norm.
"""
from __future__ import annotations
import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import pickle
import socket
import sys
import time

from diffusion_timing_adapter import build
from run_paper_ablation_revision import digest, write


def build_batch(root):
    import torch
    import torch.nn.functional as F
    import numpy as np
    _, source = build(root, 'burger', native_precision=True)
    tree = ast.parse(source)
    changes = dict(batch_size=0, latents=0, shapes=0, squeeze=0, norms=0)

    class IndependentRows(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            if node.name == 'predict':
                node.args.args.append(ast.arg(arg='seeds'))
            return self.generic_visit(node)

        def visit_Assign(self, node):
            names = {n.id for n in node.targets if isinstance(n, ast.Name)}
            if names == {'batch_size'}:
                node.value = ast.parse('len(seeds)', mode='eval').body
                changes['batch_size'] += 1
            if names == {'latents'}:
                node.value = ast.parse('draw_latents(seeds, net, device)', mode='eval').body
                changes['latents'] += 1
            if names in [{'L_pde'}, {'L_obs'}]:
                field, denominator = ('pde_loss', '128*128') if names == {'L_pde'} else ('observation_loss', '128*5')
                node.value = ast.parse(
                    f'torch.linalg.vector_norm({field}.flatten(1), dim=1).sum()/({denominator})',
                    mode='eval').body
                changes['norms'] += 1
            return self.generic_visit(node)

        def visit_Call(self, node):
            self.generic_visit(node)
            if isinstance(node.func, ast.Attribute):
                if ast.unparse(node) in ['u.view(1, 1, 128, 128)', 'u_GT.view(1, 1, 128, 128)']:
                    node.args[0] = ast.Constant(value=-1)
                    changes['shapes'] += 1
                if ast.unparse(node) in ['pde_loss.squeeze()', 'observation_loss.squeeze()']:
                    node.args = [ast.Constant(value=1)]
                    changes['squeeze'] += 1
            return node

    tree = IndependentRows().visit(tree)
    assert changes == dict(batch_size=1, latents=1, shapes=2, squeeze=2, norms=2), changes
    ast.fix_missing_locations(tree)

    def draw_latents(seeds, net, device):
        assert net.label_dim == 0, 'The pilot only covers unconditional pretrained Burgers weights'
        draws = []
        for seed in seeds:
            torch.manual_seed(int(seed))
            draws.append(torch.randn([1, net.img_channels, net.img_resolution, net.img_resolution], device=device))
        return torch.cat(draws, dim=0)

    scope = dict(torch=torch, F=F, np=np, draw_latents=draw_latents)
    exec(compile(tree, '<independent-burgers-batch>', 'exec'), scope)
    return scope['predict'], ast.unparse(tree)+'\n'


def main(args):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    protocol_path = args.inputs/'sampling_protocol_v3.json'
    protocol = json.loads(protocol_path.read_text())
    assert digest(args.inputs/'protocol.json') == protocol['input_protocol_sha256']
    assert digest(args.weights) == protocol['dm_weights_sha256']
    assert digest(args.diffusion_root/'scripts/generate_burgers.py') == protocol['diffusion_source_sha256']
    assert digest(Path(__file__).with_name('diffusion_timing_adapter.py')) == protocol['adapter_sha256']
    assert torch.cuda.mem_get_info()[0] >= 24*1024**3, 'At least 24 GiB free GPU memory is required'
    input_protocol = json.loads((args.inputs/'protocol.json').read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.append(str(args.diffusion_root))
    with args.weights.open('rb') as handle:
        net = pickle.load(handle)['ema'].to('cuda:0').eval()
    net.requires_grad_(False)
    native, _ = build(args.diffusion_root, 'burger', native_precision=True)
    batch, code = build_batch(args.diffusion_root)
    (args.output/'batched_native.py').write_text(code)
    report = dict(protocol_sha256=digest(protocol_path), pilot_sha256=digest(Path(__file__)),
                  host=socket.gethostname(), gpu=torch.cuda.get_device_name(),
                  maximum_field_difference_percent=0.001, maximum_error_difference_pp=0.001,
                  cases=[], formal_predictions_written=False)
    # Fixed before viewing pilot outcomes; indices 0,1,2,5 have native random
    # receipts on server216. The independent structured references are new.
    for mode in ['random', 'structured']:
        assert digest(args.inputs/f'smooth_{mode}.pt') == input_protocol['artifacts'][f'smooth_{mode}.pt']
        cell = torch.load(args.inputs/f'smooth_{mode}.pt', map_location='cpu', weights_only=False)
        for steps in [100, 1000]:
            ids = [0, 1, 2, 5] if mode == 'random' or steps == 100 else [0, 1]
            truth = cell['truth'][ids].to('cuda:0')
            masks = cell['mask'][ids].to('cuda:0', dtype=torch.float32)
            observed = truth*masks
            config = copy.deepcopy(protocol['diffusion_config'])
            config['generate'].update(device='cuda:0', seed=protocol['seed'])
            config['test']['iterations'] = steps
            seeds = [protocol['seed']+i for i in ids]
            references = []
            reference_seconds = []
            for j, i in enumerate(ids):
                path = args.native_results/'results'/f'smooth_{mode}'/f'DiffusionPDE_{steps}'/f'sample{i}.json'
                if path.is_file():
                    receipt = json.loads(path.read_text())
                    assert receipt['protocol_sha256'] == digest(protocol_path)
                    assert digest(path.with_suffix('.pt')) == receipt['tensor_sha256']
                    payload = torch.load(path.with_suffix('.pt'), map_location='cpu', weights_only=False)
                    assert torch.equal(payload['truth'], truth[j:j+1].cpu())
                    assert torch.equal(payload['mask'], masks[j:j+1].cpu())
                    references.append(payload['prediction'].to('cuda:0'))
                    reference_seconds.append(None)  # Different GPU contention; not a timing control.
                else:
                    config['generate']['seed'] = seeds[j]
                    torch.cuda.synchronize(); started = time.perf_counter()
                    references.append(native(config, net, observed[j:j+1], observed[j:j+1], masks[j,0], masks[j,0])[1])
                    torch.cuda.synchronize(); reference_seconds.append(time.perf_counter()-started)
            reference = torch.cat(references)
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
            prediction = batch(config, net, observed, observed, masks[:,0], masks[:,0], seeds)[1]
            torch.cuda.synchronize(); elapsed = time.perf_counter()-started
            assert torch.isfinite(prediction).all()
            norm = torch.linalg.vector_norm(truth.double().flatten(1), dim=1)
            difference = 100*torch.linalg.vector_norm((prediction-reference).double().flatten(1), dim=1)/norm
            e0 = 100*torch.linalg.vector_norm((reference-truth).double().flatten(1), dim=1)/norm
            e1 = 100*torch.linalg.vector_norm((prediction-truth).double().flatten(1), dim=1)/norm
            row = dict(mode=mode, steps=steps, sample_ids=ids,
                       field_difference_percent=difference.tolist(), error_difference_pp=(e1-e0).tolist(),
                       max_abs=float((prediction-reference).abs().max()), batch_seconds=elapsed,
                       same_gpu_single_seconds=reference_seconds, peak_bytes=torch.cuda.max_memory_allocated())
            row['passed'] = bool((difference <= .001).all() and ((e1-e0).abs() <= .001).all())
            report['cases'].append(row)
            torch.save(dict(prediction=prediction.cpu(), reference=reference.cpu(), truth=truth.cpu(),
                            mask=masks.cpu(), seeds=seeds, comparison=row), args.output/f'{mode}_{steps}.pt')
            write(args.output/'report.json', report)
            print(json.dumps(row), flush=True)
    report['all_passed'] = all(r['passed'] for r in report['cases'])
    report['completed_unix'] = time.time()
    write(args.output/'report.json', report)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--diffusion-root', type=Path, required=True)
    p.add_argument('--native-results', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
