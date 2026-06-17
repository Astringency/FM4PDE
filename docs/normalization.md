# Normalization and Physical-Space Sampling

New FM4PDE training uses channel-wise training-set standardization. For a training tensor
`data` with shape `[N,C,H,W]`, the normalizer computes:

```python
mean = data.mean(dim=(0, 2, 3), keepdim=True)
std = data.std(dim=(0, 2, 3), keepdim=True, unbiased=False)
```

The saved tensors have shape `[1,C,1,1]`. Channels with near-zero standard deviation are
protected against divide-by-zero. Training samples are standardized directly and are not
mapped to `[0,1]` or `[-1,1]`.

Each new training run saves the same normalizer in three places:

* `payload["normalizer"]` inside every checkpoint.
* `output_dir/normalizer.pt`.
* `output_dir/normalization.json`.

Resume and sampling must use the checkpoint normalizer when it exists. Test samples must
never estimate their own mean/std or min/max, because that leaks test-sample information
into inference and changes the model state scale.

The sampler treats the neural state as standardized. Before observation losses or PDE
residuals are evaluated, the full pair state is inverse-transformed with the saved
normalizer and then split into physical coefficient/source and solution fields. PDE
residual guidance is therefore computed in physical space.

Future HDF5 PDE scalar constants such as `alpha`, `c`, `b_x`, `b_y`, `kappa`, and `u_D`
are metadata for residuals, not Flow Matching channels. The model channels are
`input_data + output_data` unless a legacy materialized-constant mode is explicitly used.

Training example:

```bash
python train.py --dataset heat --data_path /large_storage/zhangxf/PDEdata/ --epochs 500 --batch_size 32
```

Sampling example:

```bash
python -m fm4pde_ablation.runner --config configs/heat.yaml --override checkpoint_path=.../fm4heat.pth
```
