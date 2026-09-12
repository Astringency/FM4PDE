# September 12 complete manuscript controls

The executed runner is commit `01c7d92`; the plan is `revision_0912_full_plan.json`.
It uses the final September 8 publication snapshot, matching checkpoint/truth hashes and current joint-task guidance weights.

- Layouts: Helmholtz input 0, four layouts, five paired seeds, shared and separate locations (40 runs). Fixed is restricted to the left half; Grid uses a (3,3) periodic translation; Columns uses five full columns.
- Temporal: six PDEs, three residual modes, input/seed 0 (18 runs). Near-endpoint differences require additional sparse temporal observations. Both endpoint errors include every channel.
- Both tasks use 100 stochastic Euler steps, batch size one, TF32 disabled.

Raw results and logs are under `outputs/ablations/revision_0912_full/{layouts,temporal,execution}` in the local FM4PDE repository and the server197 main FM4PDE repository. Both execution exit codes are zero. Failed initial import attempts remain in the logs.

`plot_revision_0912_full.py` checks all raw receipt hashes, masks and field errors, archives CPU arrays and metadata into the paper source_data directory, and renders the two paired-layout figures and three result tables. `--render-only` rebuilds from the archived arrays.

`validate_revision_0912_full.py` independently recomputes squared-sum relative errors, means and sample SDs, checks the exact ordered table cells and figure hashes, and verifies the compiled main PDF page cap and LaTeX references. Paths at the top of these scripts are specific to the documented local workspace.

The result archive `sources/code.bundle` includes the complete committed code history with no Git prerequisites. It contains the executed runner and the final renderer/validator. The source-data and audit files are archived separately beside it.
