# Shared inverse guidance settings

The main comparison, physics-based comparison, conditional averaging, and
Poisson sampling trajectories use the same task-specific guidance settings:

| Task | Solution observation weight | PDE weight | Gradient norm threshold |
| --- | ---: | ---: | ---: |
| Inverse Poisson | 9.216e11 | 0.3 | 600 |
| Inverse Helmholtz | 1.2288e12 | 0.03 | 600 |

Both use 100 stochastic steps on a uniform flow-time grid, with `c_zeta=0.1`
and PDE guidance starting at step 80. The defaults live in
`configs/main/inverse/poisson.yaml` and `configs/main/inverse/helmholtz.yaml`.
There is no additional parameter search for the following experiments.

From the repository root, after setting `PYTHON_BIN`, `DATA_ROOT`, and the
checkpoint paths as described in the README:

```bash
bash scripts/sample/comparison/physics_based.sh --device cuda:0 --output outputs/physics_shared_guidance
bash scripts/sample/ablations/conditional_averaging.sh --tasks inverse --batch-size 64 --device cuda:0 --output outputs/averaging_shared_guidance
bash scripts/sample/comparison/accuracy_during_sampling.sh --pdes poisson --tasks inverse --methods FM4PDE --device cuda:0 --output outputs/traces_shared_guidance
```

The physics comparison contains 100 Smooth inputs for each of three PDEs
and two reconstruction tasks. Its manifest preserves the original batches
and the fixed observation positions used by the physics baselines. The
`physics_smooth` observation protocol uses the baseline release filenames
in its sample identifiers, so renaming a dataset file with a `_smooth`
suffix does not change its sensors.

Conditional averaging uses ID input indices 1500–1531. Each input fixes its
observations while generating 1,000 predictions; reported means use the
first 1, 3, 10, 100, and 1,000 predictions. Its observation and sampling seed
is `20260912 + input_index`. The initial and per-step random draws retain
their indices in a canonical 1,000-draw batch. The sampler uses the same
weighted gradient and global clipping as ordinary sampling, evaluating
the weighted sum with one reverse pass. The input and observation hashes
are recorded for every draw batch. Omitting `--tasks inverse` runs all three
tasks; their unchanged forward and joint results can be reused when only
the inverse guidance settings change.

The averaging script supports `--tasks`, `--num-shards`, `--shard-index`,
and `--plan-only`. Sharding partitions the fixed input list without changing
input indices or random seeds. Use a separate output directory per shard.
Its `--override key=value` arguments follow the common experiment runner.
An 80 GB GPU can use the recorded batch size of 64; the RTX 4090 runs use 16.
Each batch and its size are retained in the output receipts.

For prediction-by-prediction reproduction, retain the recorded CUDA random-number
environment as well as the seed. The A800/A100 runs used PyTorch 2.8.0 with CUDA
12.8; the RTX 4090 runs used PyTorch 2.5.1 with CUDA 12.1. CUDA random streams
can differ across GPU/software environments, so equal seeds alone do not imply
identical draws across those environments. Independent reruns of a saved
100-step prediction on each of the three recorded environments were bitwise
identical. The output receipts record each input's actual command and batch size.

The displayed inverse averaging results use this input partition:

| GPU and software environment | Input indices | Batch size |
| --- | --- | ---: |
| A800, PyTorch 2.8.0 / CUDA 12.8 | 1500–1506, 1511–1517, 1522–1528, 1531 | 64 |
| A100, PyTorch 2.8.0 / CUDA 12.8 | 1507–1508, 1518–1519, 1529–1530 | 64 |
| RTX 4090, PyTorch 2.5.1 / CUDA 12.1 | 1509–1510, 1520–1521 | 16 |

An individual input can be reproduced through the same public script. For
example, input 1509 uses shard index `1509 - 1500 = 9`:

```bash
bash scripts/sample/ablations/conditional_averaging.sh --tasks inverse --num-shards 32 --shard-index 9 --batch-size 16 --device cuda:0 --output outputs/averaging_1509
```

Use a separate output directory for each selected input. No guidance overrides
are required. The saved `prefix_errors.json` reports the relative errors of the
averaged fields for every requested prefix count.

The inverse trajectory experiment uses the original 100 ID inputs and one
separate warmup input on an RTX 4090. Timings exclude diagnostic evaluation;
its warmup checks the timed path against ordinary sampling. DiffusionPDE's
previous trajectories can be reused because this update changes FM4PDE only.
