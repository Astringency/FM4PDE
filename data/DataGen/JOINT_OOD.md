# Joint input / solution OOD test distribution

Run from the FM4PDE repository root. The dedicated entry point keeps all existing
GRF generation profiles unchanged. It generates 1000 samples by default and uses
sample-wise deterministic seeds, independent of generation batch size.

```bash
python -m data.DataGen.generate_joint_ood calibrate --pde all \
  --baseline-root /path/to/FM4PDEbaseline --output-root /absolute/study/path
python -m data.DataGen.generate_joint_ood generate --pde poisson \
  --output-root /absolute/study/path
```

Supported PDE names: `poisson`, `helmholtz`, `darcy`, `nsnonbounded`, `burger`.
The default data root is `/large_storage/zhangxf/PDEdata`. Outputs sit beside
existing tests, with names `<pde>_test_1000-128-128_joint_ood.mat` (NS additionally
includes `-10`; Burgers uses the existing `burgers/` folder). JSON receipts record
source revision, seed, reference data checksum, physical parameters and shift checks.
Existing files are verified and reused, never overwritten. Interrupted generation
resumes from saved batches in the study's `generation/` directory.

- Poisson / Helmholtz: resolved Dirichlet sine modes 4–9 construct the solution;
  the original discrete operator constructs its forcing. Solution RMS remains
  near ID. Forcing amplitude increases with frequency as required by the PDE.
- Darcy: original 4/12 phases, source and boundary conditions; connected low
  permeability barriers inside a dominant high permeability background. This
  changes phase fraction and solution amplitude, not just small-scale texture.
- Navier–Stokes: zero-mean intermediate Fourier modes with stronger amplitude;
  the original forced solver advances them with viscosity 0.001 to time 1.
- Burgers: intermediate-frequency initial conditions with a nonzero conserved
  mean; original periodic SPIN solver, viscosity 0.01, time 1. Here `a` denotes
  the initial condition and `u` the full trajectory; terminal shift is also checked.

Calibration uses ID test cases 100:1100, disjoint from the comparison's first 100.
For each field, at least 95% of new cases must be outside the empirical ID central
99% interval in a declared feature: normalized gradient (Poisson/Helmholtz),
coefficient mean and solution RMS (Darcy), RMS (NS), signed mean (Burgers).
This establishes an empirical distribution shift, not disjoint mathematical
support. No model outputs are used to construct or accept these tests. The
interventions differ between equations; they are not a common roughness scale.
