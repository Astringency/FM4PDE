# Original DiffusionPDE evaluation inputs

The five frozen NPZ files contain the exact arrays used for the selected
indices 0–999. Four retain float64 fields; NS retains the original float32
arrays, which its native sampler converts to float64. Darcy keeps its
`H,W,N` axes. NS retains `w0` and the final `w` frame with a singleton final
axis, as its generator reads `w[..., -1]`. The historical Burgers files use
the ID test distribution; they are separate from the later Burgers extension.

`materialize_diffusion_inputs.py` creates compact files readable by the
unchanged native generators: MAT v5 for Poisson, Helmholtz and Burgers, and
HDF5 datasets for Darcy and NS. The HDF5 files serve the original `h5py.File`
reader; they are not claimed to be complete MATLAB v7.3 files. All derived
files retain the original keys, selected field values, axes and dtypes.
Their file hashes differ from the original full MAT files and the NPZs;
the report preserves all three identities separately.

After verifying the three relevant result/input/model archive receipts,
run the following from the **server197 FM4PDE root**. Restore the selected
DiffusionPDE source first if that dependency checkout does not yet exist.

```bash
REPRO="$PWD/reproducibility/revision_20260909"
OLD_DM="$PWD/outputs/main/revision_20260909/diffusion_original_main"
FROZEN_DM="$OLD_DM/native_frozen_inputs/frozen_inputs_v3"
NATIVE_DM="$REPRO/dependencies/DiffusionPDE"
FM_PY=/research_data/users/zhangxifeng/.conda/envs/fm4pde/bin/python

"$FM_PY" "$REPRO/source_snapshots.py" restore \
  --archive "$REPRO/source_snapshots/DiffusionPDE-151e721b9991.tar.gz" \
  --destination "$NATIVE_DM"

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  "$FM_PY" "$REPRO/materialize_diffusion_inputs.py" \
  --manifest "$FROZEN_DM/manifest.json" \
  --manifest-sha256 818bdbfc3c7e4888110f92eda0fd1a2d9bf5e5fbe98874c749ee80f3eff04f15 \
  --inputs "$FROZEN_DM" \
  --original-results "$OLD_DM/server216/outputs" \
  --weights "$OLD_DM/selected_models" \
  --diffusion-code "$NATIVE_DM" \
  --output "$REPRO/verification_runs/diffusion_native_inputs"
```

Both restore and materialization require new destinations. They preserve the
archived inputs and sources. The derived files occupy approximately 1.05 GB
and can be generated again; an additional permanent copy is unnecessary.

The output contains five derived input files, 26 YAML configurations, and
`materialization_report.json`. Each configuration changes only the data-file,
selected-model and new output paths. The original configuration SHA, model SHA,
source commit, original MAT SHA and derived file SHA are recorded. The report
also contains the complete command and working directory for every cell.
No command is launched by the materialization tool.

Use the generated **`generate_pde.py` commands**, including `--problem`,
`--batch 1000`, `--start_offset 0` and the recorded `--step_size`. This CLI
sets the unobserved field's guidance coefficient to zero for forward/inverse
tasks; calling an individual `generate_*` function with the unprocessed YAML
would omit that step. The CLI also replaces the last saved YAML offset with
each selected test index. The native batch size remains one; the CLI's
`--batch` argument counts consecutive test examples.

For example, the generated Poisson forward command has this form:

```bash
cd "$NATIVE_DM"
"$FM_PY" generate_pde.py \
  --config "$REPRO/verification_runs/diffusion_native_inputs/configs/MAIN1000_100_poisson_forward.yaml" \
  --problem forward --batch 1000 --start_offset 0 --step_size 100
```

Sampling uses the recorded device, seeds, observation construction, numerical
precision, guidance settings and model. Check the selected device's resources
and use a separate tmux session before starting a remote run, following the
workspace SSH instructions. An explicit change of device or environment must
be recorded. In particular, the Burgers sensor generator uses device-dependent
PyTorch random permutations; an identical integer seed in a CPU environment
does not establish that the original CUDA observation mask was reproduced.
Compare newly generated `obs_index` arrays with the archived predictions before
using a replay as the same-observation experiment.

## Evidence and limits

The complete stored-result audit recomputed all 26,000 predictions from these
NPZ truths and compared the available original metric JSON or final stored
loss. Its records are in `native_frozen_inputs/full_saved_prediction_audit`.
That numerical audit is distinct from rerunning a trained generative model.

The materialization utility has also been executed on server216 with two CPU
threads and CUDA hidden. All five derived files were reread and compared in
full, including dtype and shape. The original generators' data-loading
statements, stopped before model loading, were checked at indices 0, 17 and
999 for every PDE. All 15 checks passed. All 26 configurations were generated,
and source files, NPZ files, configurations and weights had unchanged hashes
afterwards. No model was instantiated or sampled during this check.
The report and extracted original loading statements are retained under
`verification_runs/diffusion_native_materialization/materialized_evidence`.

The original run records did not save a producer Git HEAD. The source version
`151e721b9991404154ad4b430ab85cdaedbaa399` is the preserved, verified native
implementation used for this interface; it is not presented as a recovered
historical run identity. Earlier generator changes are retained separately
in the source catalog. These materials support recomputing the historical
statistics and rerunning the recorded protocol with the captured implementation;
they do not prove bitwise recovery of every historical random trajectory.
