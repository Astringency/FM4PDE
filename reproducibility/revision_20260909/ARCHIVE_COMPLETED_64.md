# Completed result archival before the K-study release

All 64 approved entries completed copying and a separate destination SHA256
verification on server197 by 2026-09-09 09:53:41 UTC. The reconciled total is
80,100,684,576 logical bytes across 192,821 files listed in the inventories.
No entry was held, overwritten with different bytes, or omitted. Original
sources remain in place.

| Frozen batch | Verified entries | Logical bytes |
| --- | ---: | ---: |
| Initial reviewed batch | 51 | 74,427,442,592 |
| Original ablation summary CSV | 1 | 3,837,690 |
| Selected baseline checkpoints and metadata | 3 | 485,715,856 |
| Burgers dependencies, native inputs and consumer evidence | 6 | 4,128,870,079 |
| Native DiffusionPDE inputs and numerical audit | 3 | 1,054,818,359 |
| Total | 64 | 80,100,684,576 |

The target root is
`/research_data/users/zhangxifeng/C01Python/FM4PDE/outputs/` on server197.
Each entry is under its recorded `main/revision_20260909/` or
`ablations/revision_20260909/` destination. The exact per-entry destination,
source, byte count, receipt SHA256, and completion time are in
`archive_completed_64_entries.json`, SHA256
`7f79577f93b869ef01648795c50ed8a0a80b2d17588f543d516839d9a0597e3e`.
For every entry, the independent execute, verify, and state receipt hashes
agree. The reconciler also checked total bytes, identifiers, and destination
subtree separation across all five frozen inventories.

`completed_archive_evidence_64.tar.gz` preserves 279 exact audit files,
including all frozen execution inventories, execute/verify outputs, final
execution states, queue dependencies, and independent addendum reviews.
Its SHA256 is
`05a14fbdab5555c1df03b2cc37e50c011f657ba5727e3480af4a8cfc84d97863`;
its size is 6,456,007 bytes. Every member was reopened from the archive and
compared by size and SHA256. `completed_archive_evidence_64.manifest.json`
lists their original source paths and hashes. These records complement the
per-file `.revision_archive_receipt.json` in each actual result destination.

All four local archive orchestration sessions and server197's corresponding
`archive0909_*` helper sessions had exited by 09:55 UTC. The independent
archive helper checkout remained at `6f5bfd5`; no active producer checkout
was synchronized. Server197 still had 4,242,704,715,776 bytes free at this
check. Other existing user sessions were left untouched.

The three K-study result entries remain outside this completed batch while
their producer, exporter, and scientific-review gates finish. Their selected
Poisson input/weight snapshot is already included among these 64 entries.
The old 316 ablation controls and 150 original Burgers prediction files are
separately verified in place under the requested output roots; their bytes
are not counted again here. Final shared paper assets, CPU replay evidence,
and source/environment packages are recorded by their respective final
publication and reproducibility manifests.
