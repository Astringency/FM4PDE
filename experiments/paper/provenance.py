"""Execution details that must agree when resuming a numerical experiment."""
import importlib.metadata
import os
import platform


def runtime_identity(device):
    import torch

    versions = {}
    for name in ('numpy', 'scipy', 'h5py', 'torch'):
        versions[name] = importlib.metadata.version(name)
    target = torch.device(device)
    if target.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('A CUDA experiment cannot be resumed or run on CPU implicitly')
    return dict(
        python=platform.python_version(), packages=versions, cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(), device_type=target.type,
        gpu=torch.cuda.get_device_name(target) if target.type == 'cuda' else None,
        threads=torch.get_num_threads(),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,
        cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
    )
