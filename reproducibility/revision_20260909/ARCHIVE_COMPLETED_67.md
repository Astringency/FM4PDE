# Final result archive: 67 verified entries

All 67 approved result and dependency entries completed copying and a separate
destination-file SHA256 verification on server197 by **2026-09-09 11:38:49 UTC
(19:38:49 Beijing)**. The reconciled inventory contains **194,202 files and
94,486,625,572 logical bytes**. Every approved entry is verified; no entry is
held, missing, or inconsistent. Original source files remain in place.

| Frozen batch | Verified entries | Logical bytes |
| --- | ---: | ---: |
| Initial reviewed batch | 51 | 74,427,442,592 |
| Original ablation summary CSV | 1 | 3,837,690 |
| Selected baseline checkpoints and metadata | 3 | 485,715,856 |
| Burgers dependencies, native inputs and consumer evidence | 6 | 4,128,870,079 |
| Native DiffusionPDE inputs and numerical audit | 3 | 1,054,818,359 |
| Completed conditional-sample averaging study | 3 | 14,385,940,996 |
| **Total** | **67** | **94,486,625,572** |

The destination root is
`/research_data/users/zhangxifeng/C01Python/FM4PDE/outputs/` on server197.
Each study is under its recorded `main/revision_20260909/` or
`ablations/revision_20260909/` destination. The authoritative final ledger is
`archive_completed_67_entries.json`, SHA256
`8f8e252898ce9bd7d9207136bf2225bc1412ee5bf36485821f6399adc9baabc0`.
It records every source, destination, file count, byte count, receipt hash,
and completion time. The reconciliation checks the approved inventory hashes,
matching execute/verify/state receipt hashes, total bytes, unique identifiers,
and nonoverlapping destination subtrees. The entry counts refer to copied
inventory entries; file counts refer to the source files listed in those entries.

## Completed averaging study

The three final entries are below
`ablations/revision_20260909/conditional_scaling_20260909/`:

| Relative destination | Files | Logical bytes | Receipt SHA256 |
| --- | ---: | ---: | --- |
| `server197/results` | 284 | 3,561,553,404 | `5905709acb8764ad8917603b4c7215aeaa0e7621eb36e06aa2d95458caee4409` |
| `server216/results` | 1,079 | 10,730,368,835 | `452a6ae73858014b58d4d2a1b8a342d2a773159555ab0e104a53df68e265ca94` |
| `local_audit` | 18 | 94,018,757 | `12facf9c6d18a543e20d718e8eeaa44e0e6418d6e43bbe4e5a2e06ab29da6d6c` |

The frozen release inventory is
`archive_inventory.conditional_scaling_final.json`, SHA256
`c473fa1979647f40144b515cbae49550b8b17edcdda6f628bbc29b98452f9ad9`.
Its release required four completed producers, the completed collector, all
480 tensor proofs and timing rows, and the frozen scientific review. The
review binds the final manifest, per-input CSV, and READY hashes. The release
check binds current tensor receipts, timing values, and environments to that
reviewed evidence; the executor then verifies the actual source and destination
file bytes. The original production directories were not changed or restarted.

The study retains 96 complete pools for 32 Poisson inputs and three tasks,
with K = 1, 3, 10, 100, and 1000 and 100 sampling steps. It contains 96,000
canonical trajectories and records 106,944 timed draws. Its selected inputs
and weights were already included in the preceding 64 entries. Scientific
and figure review passed before this final transfer; CPU replay of archived
results is documented separately by the verification owners.

The initial inventory's pending K observations remain unchanged as historical
records. The final released inventory and this 67-entry ledger supersede those
pending observations. A first release-builder attempt refused historical
duplicate identifiers before any transfer; the corrected check applies the
same ready-entry filter as the executor. The refusal and corrected command
are preserved in the evidence package; no scientific gate was weakened.

## Self-contained verification evidence

`completed_archive_evidence_67.tar.gz` contains **314 exact audit files**, with
38,783,528 logical bytes compressed to **6,784,056 bytes**. Its SHA256 is
`5438624687a0a2c926e6aabb3cef4b7df539866b5be223aeacd9f598dc453070`.
The companion `completed_archive_evidence_67.manifest.json` lists each member's
original source path, size, and SHA256; its SHA256 is
`acba7da43b418b8d09c94506ed10a8f0d51343943665e71e114213b04dce0c55`.

The package includes all exact frozen execution inventories, execute/verify
outputs, final execution states, queue dependencies, independent addendum
reviews, the final 67-entry ledger, and the K release/review evidence. All 279
members of the previous 64-entry package were verified before inclusion. Every
member of the new package was reopened and checked by size and SHA256, and
all newly included source files were checked again after packaging. The prior
64-entry ledger and package remain available as historical completion records.

The included `final_ledger_command.json` records the successful invocation of
`summarize_archive_execution.py --require-complete`. Its input inventories and
states are all included in the package; after extraction their paths can be
replaced by the corresponding extracted paths to repeat the accounting check.
These audit records complement each actual destination's
`.revision_archive_receipt.json`, which contains the per-file archive hashes.
They do not replace reading the destination files when running a new full
archive verification.

At 11:40 UTC, the local archive tmux session and all server197 `archive0909_*`
helper sessions had exited. The independent helper checkout remained at
`6f5bfd51de24fceeba5e39e362a4de513176c842`; existing user sessions and the
separate CPU replay session were left intact. Server197 had
4,225,107,681,280 bytes free. The exact observations are included as
`final_process_check.json` in the package.

The old 316 ablation controls and 150 original Burgers prediction files were
separately verified in place under the requested output roots; their bytes
are not counted again here. Shared final paper assets, source/environment
packages, and CPU replay evidence have separate delivery manifests under
`reproducibility/revision_20260909/`. The final paper asset directory is
`paper_assets_final_20260909`; the earlier paper-asset snapshot is preserved.
