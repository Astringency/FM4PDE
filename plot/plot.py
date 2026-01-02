import numpy as np
import matplotlib.pyplot as plt
    

def plot_eval_pde(true, obs, sample, cmap="plasma"):

    def to_numpy(x):
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return x

    true = to_numpy(true)
    obs = to_numpy(obs)
    sample = to_numpy(sample)

    C = true.shape[0]

    fig, axes = plt.subplots(
        nrows=C,
        ncols=3,
        figsize=(10, 3 * C),
        constrained_layout=True
    )

    if C == 1:
        axes = axes[np.newaxis, :]

    for i in range(C):
        row_min = min(
            true[i].min(),
            obs[i].min(),
            true[i].min()
        )
        row_max = max(
            true[i].max(),
            obs[i].max(),
            true[i].max()
        )

        # True
        im0 = axes[i, 0].imshow(
            true[i],
            cmap=cmap,
            vmin=row_min,
            vmax=row_max
        )
        axes[i, 0].set_title(f"True[{i}]")
        axes[i, 0].axis("off")
        fig.colorbar(im0, ax=axes[i, 0], fraction=0.046, pad=0.04)

        # Observation
        im1 = axes[i, 1].imshow(
            obs[i],
            cmap=cmap,
            vmin=row_min,
            vmax=row_max
        )
        axes[i, 1].set_title(f"Obs[{i}]")
        axes[i, 1].axis("off")
        fig.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)

        # Samples
        im2 = axes[i, 2].imshow(
            sample[i],
            cmap=cmap,
            vmin=row_min,
            vmax=row_max
        )
        axes[i, 2].set_title(f"Generate[{i}]")
        axes[i, 2].axis("off")
        fig.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04, extend="both")

