# Revision result archive: inventory and controlled transfer

This document prepares the full archive. The 2026-09-09 census used read-only
SSH and local metadata generation. A subsequent authorized pilot archived
exactly one completed 4,195-byte NS timing log; its evidence is described below.
After that pilot, the authorized completed-study batch began on 2026-09-09;
its live completion record is described below. Existing GPU workers and their
checkouts were not changed.

## Scope and capacity

The target is server197, under
`/research_data/users/zhangxifeng/C01Python/FM4PDE/outputs`:

- `main/revision_20260909/<study>/<origin-or-component>`: main evaluations,
  controlled sampling time, and Burgers.
- `ablations/revision_20260909/<study>/<origin-or-component>`: guidance,
  three-draw averaging, new conditional sample scaling, time approximations,
  and NS supplementary studies/model selection.

`archive_inventory.summary.json` is the small review index;
`archive_inventory.json` is the executable index with per-file configuration
and protocol hashes. Its source observations include recorded commits and
script hashes where present. These are recorded evidence, not claims that the
current server checkout produced an old result. Exact code versions and
external repositories are documented separately in `source_repositories.json`.

The initial census found 70 entries, with no missing source or mismatch against
known hashes. Ready primary/dependency entries total approximately 62.66 GiB.
Pending sources are growing; the expected final additional archive is about
85–90 GiB, including intentionally separate canonical reading views and frozen
input copies. This estimate is not a transfer measurement. The K study alone
has 96,000 canonical draws and 106,944 actual timed draws across all K values;
float32 two-field prediction payloads require 14,017,363,968 bytes (13.05 GiB),
plus tensor containers, averaged fields, receipts, and validation records.

The server197 `/research_data` filesystem was checked at preparation time:
7,619,219,456,000 total bytes, 4,339,177,496,576 available bytes (41% used).
Recheck immediately before executing. The existing FM `outputs/main` (~62.34
GiB) and `outputs/ablations` (~4.10 GiB) already meet the requested location and
are registered in place. They must not be recursively copied into themselves.
`in_place_reference_checks.json` additionally verifies that all 316 unchanged
ablation tensors and all 150 Burgers original random-sensor result files are
present on server197. It records every SHA256 and compares the available
local-copy/frozen-manifest expected hashes. Refresh that bounded read-only
check with `python reproducibility/revision_20260909/verify_in_place_references.py`.

The 3.06 TB baseline evaluation parent and the 248.5 GB historical checkpoint
library are references only. The inventory selects exactly 156 published
baseline raw metric files (about 150 MiB) through table provenance, and retains
necessary frozen inference inputs/models. Training datasets remain external
dependencies with their original protocol paths/hashes and source generators.
No full training dataset is needed by the frozen-input replay recipes.

Shared paper `source_data/` and `figures/` mix main and ablation studies. They
are assigned to the root-owned
`FM4PDE/reproducibility/revision_20260909/paper_assets`, not to main results.
The root also archives the ignored source snapshot tar files and Git bundles;
restore dependency checkouts from these bundles instead of transferring
duplicate `.git` directories. Result studies retain their own numerical exports.

## Meaning of inventory states

`ready` means the owner considered the artifact complete and its source was
present at the census. It is not a substitute for the final scientific audit.
Only `ready` entries with role `primary` or `dependency` can be executed.
`alternate` sources preserve provenance and recovery options; they are not
copied by `--all-ready` because their complete outcomes already occur in a
primary collected study. `reference_only` and `already_at_target` are never
copy sources. All original directories remain untouched.

The K production and collector entries stay `pending` until the four shard
completion receipts, 480 aggregate rows, 96 complete K=1000 pools, and final
scientific checks pass. The NS local study stays pending until all 15,000
examples, the collection gate, and the independent residual audit pass. Logs
of active producers are also pending. A directory's existence or stable size
does not establish completion. Legitimate recorded attempts within each
production study are retained.

`build_archive_inventory.py` refreshes the census but intentionally never
promotes the K entries automatically. It consumes the NS owner's
`audit/ns_main_revision_0909/ARCHIVE_INVENTORY.json` for NS completion states.
After the relevant owner supplies the final evidence, the root may create a
reviewed inventory copy, promote only explicitly reviewed entry IDs, and bind
their current `observed.metadata_sha256` as `evidence_sha256`. Retain the review
record and the original inventory. Do not merely change every pending status
to ready. A refresh while production is running remains an observation only.

## Read-only planning and local safety tests

Run from the local FM4PDE repository:

```bash
python reproducibility/revision_20260909/build_archive_inventory.py
python reproducibility/revision_20260909/archive_results.py --all-ready
python -m unittest discover -s reproducibility/revision_20260909 -p test_archive_results.py -v
```

The default archive invocation prints a plan and performs no SSH or writes.
The builder performs read-only SSH, reads file metadata/configurations, and
writes only the local inventories. The eleven local tests use disposable tiny
trees; they check repeated execution, refusal of different data, pending
states, frozen hashes, corrupted transfers, link materialization, selected
files, source mutation, archive verification, and persisted task recovery
without launching duplicate tmux sessions. Tmux/child execution is mocked in
the two persistence unit tests. The separate real pilot below exercised
remote tmux promotion, fresh verification, and completed-job recovery.

## Completed small real-data pilot

`archive_pilot_validation.json` records a successful one-file pilot using the
exact committed script at `6f5bfd51de24fceeba5e39e362a4de513176c842`. That code
was transferred by a verified 16,397,311-byte Git bundle and restored to the
new independent checkout
`/research_data/users/zhangxifeng/C01Python/FM4PDEArchive0909_6f5bfd5`.
The canonical FM4PDE Git checkout remained clean at its original commit.

The source was the completed server193
`/home/zhangxf/C01Python/NSMainRevision0909/timing.log`, copied to the formal
`main/revision_20260909/ns_main_revision_0909/server193/timing_log/timing.log`.
Source and destination both contain 4,195 bytes and have SHA256
`8895e943187125c5d8d9a36d643cde67f4d4db2a0972062473a1dd7349c12f8a`.
The archive receipt SHA256 is
`2ef6e8d88a16c9a8d75b19f40b1130d8129c5740345c4128b26afccb821718fc`.

Actual `_promote` and fresh `_verify` helpers ran in independent tmux sessions;
both completed with empty stderr and their sessions automatically exited.
`--resume-job` returned the recorded promotion result and did not create a
third job. The two local orchestrator sessions also exited. Original source
bytes were checked again after the pilot and were unchanged. This validates
the real remote lifecycle on a small relayed file; it does not claim a
deliberately injected mid-transfer network failure or a completed large-copy
stress test. No `--all-ready` transfer or GPU task was started.

## Completed-study batch

`review_completed_archive.py` binds the final study gates, rechecks all
configuration metadata against the original census, and produces a separate
reviewed inventory. The 07:13 UTC review approved 51 primary/dependency entries
totalling 74,427,442,592 bytes (69.32 GiB), with no changed metadata or missing
source. It includes the explicitly frozen NS 15,000-example study and its
80-call timing. The K production directories and collector remain pending.
`archive_completion_review.json` records the exact gates and per-entry reasons.

The frozen execution inventory is
`/home/tat512/C01Python/audit/revision_archive_execution_20260909/archive_inventory.reviewed.json`,
SHA256 `9a10fcc11367f8a74198837cece6da672068dce647f5743064a324d63e6b5dd4`.
`run_reviewed_archive.py` runs each approved entry through `--execute` and then
an independent `--verify`, strictly serially, from the local owned session
`revision_archive_0909`. It retains separate per-entry command logs and atomically
updates `execution_status.json` in that same local audit directory. The record
contains verified bytes and receipt hashes. Failed entries are held with their
logs preserved; no different target is replaced. The runner skips verified
entries if explicitly resumed with the same reviewed inventory. It refuses
changed inventories or unreviewed existing attempt logs.

The remote archive script remains the tested, independent `6f5bfd5` checkout;
new local review/orchestration code does not alter any remote producer. The
completion record, rather than this preparation document, determines which
entries have actually completed. Final paper assets and source bundles are
managed separately by the root coordinator.

At 07:32 UTC, all nine original ablation entries were verified. Reproduction
review identified one further required dependency: the frozen 1060-row
`archived_ablation_summary.csv` (3,837,690 bytes). Its SHA256 is exactly the
publication manifest's `source_archive_sha256`. The root explicitly authorized
this one small dependency transfer alongside the bulk to unblock the archive
recomputation; it is the only concurrent-entry exception. The independent
copy and verification completed, and its owned tmux session exited. It is
stored at
`ablations/revision_20260909/paper_revision_20260908/local_exports/archived_ablation_summary/archived_ablation_summary.csv`.
`archive_inventory.addendum_summary.json` records the reviewed dependency;
`ablation_archive_receipts.json` includes the complete study handoff and its
receipt SHA256
`5550c18e385300f589f3c558f7dcdfe59da8c704e9092c1e5572d42736ccc215`.
The active 51-entry inventory was not rewritten to insert this addition.

## Install the exact script through Git, outside production

The first archive run can use a **new independent server197 checkout**. The
script's `--remote-script` accepts that checkout's absolute script path; the
output target remains the canonical FM4PDE `outputs` directory.

Before preparing the checkout, commit the reviewed local code and inventory.
The following commands are an execution recipe, not actions already performed:

```bash
ARCHIVE_COMMIT=$(git rev-parse HEAD)
ARCHIVE_CODE_DIR=$(mktemp -d /tmp/fm4pde-archive-code.XXXXXX)
git bundle create "$ARCHIVE_CODE_DIR/archive.bundle" HEAD
```

Choose unused remote names for the bundle and checkout. For example, after
setting `REMOTE_BUNDLE` and `REMOTE_CHECKOUT` to new absolute paths:

```bash
ssh server197 "test ! -e '$REMOTE_BUNDLE' && test ! -e '$REMOTE_CHECKOUT'"
scp "$ARCHIVE_CODE_DIR/archive.bundle" "server197:$REMOTE_BUNDLE"
ssh server197 "git clone --no-checkout '$REMOTE_BUNDLE' '$REMOTE_CHECKOUT'"
ssh server197 "git -C '$REMOTE_CHECKOUT' checkout --detach '$ARCHIVE_COMMIT'"
ssh server197 "git -C '$REMOTE_CHECKOUT' rev-parse HEAD"
REMOTE_SCRIPT="$REMOTE_CHECKOUT/reproducibility/revision_20260909/archive_results.py"
```

Use trusted literal path values without single quotes in these shell examples.
The script checks that its local and remote SHA256 match before any copy.
No producer checkout, tmux session, configuration, or existing archive changes
as a result of installing the independent Git checkout.

A preparation-time read-only check of server197's canonical FM4PDE showed a
clean `main` at `bea62781585c897af1b9cded623e6f4105b4d879`, which was an ancestor
of local `main` (`62b07060b47294f95c09a3ba560edca4de7e53cb`). Thus a fast-forward
was possible at that observation. Recheck before any later update; the archive
does not require updating this checkout. Once the root chooses to consolidate
the code there, normal reviewed Git synchronization can do so independently.

## Transfer and verify after the completion review

Start with one small entry and preserve the command output containing its
receipt SHA. Create a new local session with `tmux new-session -s
revision_archive_0909` and run the long orchestrator there; detach with Ctrl-b
then d and later attach with `tmux attach-session -t revision_archive_0909`.
The following examples use the reviewed
inventory copy and the independent checkout installed above:

```bash
python reproducibility/revision_20260909/archive_results.py \
  --inventory /absolute/path/archive_inventory.reviewed.json \
  --ids burgers_random_fm --remote-script "$REMOTE_SCRIPT" --execute

python reproducibility/revision_20260909/archive_results.py \
  --inventory /absolute/path/archive_inventory.reviewed.json \
  --all-ready --remote-script "$REMOTE_SCRIPT" --execute \
  > /absolute/new/path/archive_execution.jsonl 2>&1

python reproducibility/revision_20260909/archive_results.py \
  --inventory /absolute/path/archive_inventory.reviewed.json \
  --all-ready --remote-script "$REMOTE_SCRIPT" --verify \
  > /absolute/new/path/archive_verification.jsonl 2>&1
```

`--verify` only reads the archived payload and receipt, recomputes every file
SHA256, checks identity and the inventory's frozen metadata, and prints the
receipt hash. It does not read or modify the producer or create copy staging.
It does save its own job input/status/log under `.archive_jobs`, as described
below. A new invocation always performs a new verification; only an explicit
`--resume-job` returns the result of the identified previous invocation.
Compare receipt hashes with the saved execution log as an external record.
Numerical re-export checks are separate and still required where specified in
`STUDY_RECIPES.md` and `REPRODUCIBILITY_REVIEW.md`.

The long server197 local-copy, promotion hash, and verification operations each
run in a dedicated `archive0909_<24-hex-request-id>` tmux session. Its input,
start time/PID, stdout, stderr, and atomic final result are stored under
`outputs/<category>/revision_20260909/.archive_jobs/<request-id>/`. Short SSH
polls merely observe this task; an SSH disconnect does not terminate it. The
script prints this exact job path and the request nonce to the execution log.
After a client/connection interruption, use:

```bash
python reproducibility/revision_20260909/archive_results.py \
  --remote-script "$REMOTE_SCRIPT" \
  --resume-job /absolute/path/from/persistent_archive_job
```

This observes the same task and never starts another helper. Once a result is
available, the invocation prints it and exits. If the server itself stopped
or the task was terminated, the status is `interrupted`; inspect that task's
logs and private staging before choosing a new invocation. A failed helper
also keeps its logs. Each task changes `remain-on-exit` only in its own new
window, so it exits automatically after writing its result, regardless of
the user's default. No existing tmux session or configuration is changed.
After the local orchestrator and verification finish, exit the shell in the
owned `revision_archive_0909` session to remove that session; retain archive
job records and execution/verification logs as provenance.

For server197 sources, copying occurs locally on server197. Other sources are
relayed through a private local directory because rsync cannot accept remote
hosts at both ends. The default relay root is
`/home/tat512/C01Python/audit/revision_archive_relay`; override it with
`--relay-root` if necessary. Only one entry is relayed at a time. Allow at least
the largest selected source plus 10% and 1 GiB locally (about 13 GiB for the
current largest ready entry), and free capacity for the complete destination.
Cross-host data traverses the network twice. At a hypothetical sustained
20–100 MiB/s, 90 GiB would take 15–77 minutes for one copy pass; hashing,
small-file overhead, the relay hop, and verification increase elapsed time.
No bandwidth benchmark was run, so this is a planning range, not an ETA.

Each entry is copied to a unique `.incoming/<id>-<uuid>/payload` directory on
the destination filesystem. SHA256 hashes are checked before publication.
For server197 sources the source tree is hashed before and after copying;
for relayed sources rsync rereads source checksums after the pull. Each final
tree receives `.revision_archive_receipt.json` with per-file SHA256, byte
counts, original host/path, inventory metadata, and script SHA. Only then is
the entire new directory renamed into place. Existing identical archives are
reported as already verified; different content is refused without changing
the old target. Final destination files are never updated in place. Failed
staging/relay directories are kept for inspection; the operator decides later
whether to remove their own failed staging directories. Producers and old
archives are never deleted.

## Required reading views and model bindings

The ablation `canonical_results` entry dereferences all local PDE links and
creates one complete 11-PDE root. Use this for both ablation and original
three-draw exporters. Do not pass the separate host trees and the canonical
copy together to a tool that forbids duplicate results. Rebuild a **new**
ablation snapshot against this archived root before plotting; historical
`records.json` contains absolute producer paths and remains immutable.
For unchanged controls, `--original-root` is the canonical server197 FM4PDE
root, where the existing `outputs/ablations` remains in place.

The Burgers FM checkpoint is explicitly archived to
`main/revision_20260909/burgers_revision_20260908/selected_models/fm/weights.pth`:
385,948,627 bytes, SHA256
`b76ea10874c37b04061d41e909a6e05bfb954f3e6797fd70895bf27f49abf956`.
The matching DiffusionPDE checkpoint is in `local/burgers_weights/`, SHA256
`a24eddebaff43e477e015e8a0f4869e27bc0aeaae224f24f319c16e3dad50edf`.
The six frozen input tensor cells suffice; old MAT training/data parents are
not copied. The `burgers_fm_random` entry is a metric export plus a manifest
binding original FM predictions already present under `outputs/main`.

The complete NS local study is copied with links dereferenced. This retains
both the root weight and a usable main `inputs/weights.pth`, and materializes
the timing source/masks/protocol/two weights. The main `inputs` and
`timing_inputs` are different layouts and must not be interchanged. Preserve
frozen original path strings in protocols; use the explicit relocation flags
in the recipes instead of rewriting protocol bytes.

K raw roots retain each original host separately. Run
`export_conditional_sample_scaling.py --results <host>/results/production
--inputs <frozen_inputs>/poisson --selection <original-frozen-selection.json>
--output <new-host-export>` for each host, then aggregate with the plotter.
The exporter `--inputs` points to `poisson/` itself; the production/validation
runner takes its parent. Use the recorded Poisson selection bytes from the
frozen producer source snapshot. The original collector remains a deployment
monitor and must not be invoked for offline archive review. Root's
`k_relocation_validation.json` records the completed one-pool equivalence
check; it is not a claim of full-study numerical migration validation.
