# Poisson proposal endpoint guidance

This experiment addresses the gradient transport defect in the paper's
`Endpoint-Transport Defects` subsection (`fm4pde_jmlr_revision_0906.tex:1580`).
It changes only the point where the endpoint loss and its gradient are evaluated.

Original:

\[
\widetilde x_{k+1}=\mathscr A_k(x_k,\epsilon_k),\qquad
g_k^-=\nabla_x\mathcal L_h(E_{t_k}(x_k)),\qquad
x_{k+1}=\widetilde x_{k+1}-\gamma_k\operatorname{clip}(g_k^-).
\]

Proposal variant:

\[
z=\operatorname{stopgrad}(\widetilde x_{k+1}),\qquad
g_k^+=\nabla_z\mathcal L_h(E_{t_{k+1}}(z)),\qquad
x_{k+1}=z-\gamma_k\operatorname{clip}(g_k^+).
\]

The endpoint is the CondOT predictor `z + (1 - t_next) * v(z, t_next)` followed
by physical-unit decoding. Its input Jacobian is included. The generating
proposal's Jacobian is excluded. At `t_next = 1`, the endpoint is exactly `z`.
The noise draws, guidance multiplier, PDE gate, loss weights and clipping rule
retain their original values. In particular, the schedule still uses the same
`t_k` as the original implementation; changing it would be a second intervention.

Under L-smoothness and a sufficiently small effective step
`a_k = gamma_k * clip_scale`, the new correction satisfies

\[
F_{k+1}(x_{k+1})\le F_{k+1}(\widetilde x_{k+1})
-a_k(1-La_k/2)\|g_k^+\|^2.
\]

The stale-gradient error term is absent from this correction bound. This does
not remove objective transport `p_k`, endpoint approximation error, incomplete
observation ambiguity, or the step-size restriction. It does not imply smaller
reconstruction error. Diagnostics report objective changes, not `p_k` itself,
because the moving local minima in its definition are unavailable.

## Running

The default sampler is unchanged. Enable the new method with:

```bash
python sample.py --config configs/main/forward/poisson.yaml \
  --override gradient_target=proposal_state_chain_rule \
  --override checkpoint_path=/path/to/poisson.pth \
  --override device=cuda:0 --override output_dir=/absolute/results/path
```

The experimental option requires endpoint loss, Euler steps and single-step
endpoint prediction. Both stochastic and deterministic proposals are supported.
`next_state_direct` is a different existing option: it applies loss directly to
the raw proposal without the endpoint prediction used here.

`study.py setup` freezes a randomly chosen subset of the archived ID inputs,
its masks and checkpoint identity. `study.py pilot` measures memory at batch
sizes 1 and 4 and checks full-trajectory reproducibility. `study.py run` performs
the paired experiment at 100 steps using 32 cases and seeds 0, 1 and 2.
Only masked measurements enter inference; full fields enter scoring afterward.
All initial and bridge noise tensors are compared by SHA256. The three tasks
are sparse forward, sparse inverse and sparse joint reconstruction.

`diagnose.py` evaluates both gradients and both counterfactual corrections at
selected steps on the original trajectory, holding the scheduled loss fixed.
`budget_control.py` gives the original forward sampler a step count chosen only
from measured runtime, rounded to a multiple of five. The same input cases and
seeds are used. `report.py` audits saved predictions and noise pairing and
computes paired bootstrap intervals over cases after averaging seeds.

The production runs use code commit `3b4adb7`; subsequent commits add analysis,
tests and distinct output naming without changing the numerical algorithm.
Artifacts retain the frozen protocol, exact code bundle, input tensors, per-case
predictions and metrics, timings, memory measurements and process exit receipts.
