"""Synthetic failure-boundary checks; no SSH, inference, or real result writes."""
import copy
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import collect_remote
from wait_finalize import Finalizer, remote_action, validate_audit, validate_comparison, validate_sensitivity


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class FinalizationBoundaries(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / collect_remote.TASK
        self.root.mkdir()
        cells = [dict(cell_id=f"synthetic/{i}", count=1000) for i in range(57)]
        write(self.root / "protocol.json", dict(status="frozen", cells=cells, expected_predictions=57000))
        self.protocol_sha = hashlib.sha256((self.root / "protocol.json").read_bytes()).hexdigest()
        workers = [f"{s}-gpu{g}" for s, n in (("server197", 2), ("server216", 8)) for g in range(n)]
        jobs = [dict(worker=workers[i % 10]) for i in range(114)]
        queue = self.root / "provenance/production_queue_v1.json"
        write(queue, dict(jobs=jobs))
        queue_sha = hashlib.sha256(queue.read_bytes()).hexdigest()
        self.worker_files = []
        for worker in workers:
            prefix = self.root if worker.startswith("server197") else self.root / "incoming/server216"
            path = prefix / "queue_state/workers" / f"{worker}.json"
            write(path, dict(status="complete", worker=worker, protocol_sha256=self.protocol_sha,
                             queue_sha256=queue_sha, completed_jobs=sum(j["worker"] == worker for j in jobs),
                             total_jobs=sum(j["worker"] == worker for j in jobs), active_jobs=[], failed_jobs=[]))
            self.worker_files.append(path)
        self.receipts = []
        for cell in cells:
            folder = self.root / "jobs/prod_test" / cell["cell_id"] / "batch_test"
            receipt = folder / "receipt.json"
            write(receipt, dict(cell_id=cell["cell_id"], indices=list(range(1000)), protocol_sha256=self.protocol_sha))
            (folder / "prediction.pt").write_bytes(b"synthetic-only")
            self.receipts.append(receipt)

    def snapshot(self):
        return remote_action("snapshot", str(self.root), {})

    def test_full_coverage_and_ten_correct_workers_are_ready(self):
        result = self.snapshot()
        self.assertTrue(result["ready"])
        self.assertEqual(result["unique_predictions"], 57000)
        self.assertEqual(len(result["workers"]), 10)

    def test_missing_prediction_never_counts_as_completed(self):
        (self.receipts[0].parent / "prediction.pt").unlink()
        result = self.snapshot()
        self.assertFalse(result["ready"])
        self.assertEqual(result["unique_predictions"], 56000)

    def test_duplicate_receipt_cannot_hide_behind_57000_deduplicated_count(self):
        folder = self.receipts[0].parent.with_name("batch_duplicate")
        write(folder / "receipt.json", json.loads(self.receipts[0].read_text()))
        (folder / "prediction.pt").write_bytes(b"synthetic-only")
        result = self.snapshot()
        self.assertEqual(result["unique_predictions"], 57000)
        self.assertFalse(result["ready"])
        self.assertTrue(result["errors"])

    def test_partial_staging_does_not_count_as_a_duplicate(self):
        folder = self.root / "jobs/prod_test/.partial_upload/batch_duplicate"
        write(folder / "receipt.json", json.loads(self.receipts[0].read_text()))
        (folder / "prediction.pt").write_bytes(b"synthetic-only")
        self.assertTrue(self.snapshot()["ready"])

    def test_worker_failure_blocks_even_with_all_predictions(self):
        path = self.worker_files[-1]
        state = json.loads(path.read_text())
        state["status"] = "failed"
        write(path, state)
        result = self.snapshot()
        self.assertFalse(result["ready"])
        self.assertTrue(result["errors"])

    def test_stale_queue_and_self_reported_wrong_job_count_block(self):
        path = self.worker_files[-1]
        original = json.loads(path.read_text())
        for change in (dict(queue_sha256="stale"), dict(completed_jobs=1, total_jobs=1)):
            write(path, dict(original, **change))
            self.assertFalse(self.snapshot()["ready"])

    def test_ssh_helper_is_self_contained(self):
        source = inspect.getsource(remote_action) + "\nimport json,sys\nprint(json.dumps(remote_action('snapshot',sys.argv[1],{})))"
        result = subprocess.run([sys.executable, "-c", source, str(self.root)], capture_output=True, text=True, check=True)
        self.assertTrue(json.loads(result.stdout)["ready"])

    def test_marker_is_immutable_and_restarts_keep_original_timestamp(self):
        document = dict(status="computation_verified", verified_utc="first", audit_sha256="abc")
        original = remote_action("publish", self.root, dict(name="proof.json", document=document))
        repeated = remote_action("publish", self.root, dict(name="proof.json", document=dict(document, verified_utc="second")))
        self.assertEqual(original, repeated)
        with self.assertRaises(ValueError):
            remote_action("publish", self.root, dict(name="proof.json", document=dict(document, audit_sha256="changed")))

    def stage_payload(self):
        code = self.root / "code"
        (code / "revision_pairing_0915").mkdir(parents=True)
        for name in ("run_stage.py", "audit_outputs.py"):
            (code / "revision_pairing_0915" / name).write_text("# synthetic stage\n")
        return dict(name="final_audit", session="test_session", code_root=str(code),
                    output=str(self.root / "audits/final"),
                    command=[sys.executable, str(code / "revision_pairing_0915/audit_outputs.py")])

    def test_stage_launches_once_and_resumes_recorded_success(self):
        payload = self.stage_payload()
        with patch("subprocess.run", side_effect=[subprocess.CompletedProcess([], 1),
                                                   subprocess.CompletedProcess([], 0, "", "")]) as run:
            self.assertTrue(remote_action("stage", self.root, payload)["alive"])
            self.assertEqual(run.call_count, 2)
            self.assertIn("--status", run.call_args.args[0][-1])
        write(self.root / "completion/final_audit.exit.json", dict(exit_code=0))
        with patch("subprocess.run", return_value=subprocess.CompletedProcess([], 1)) as run:
            self.assertEqual(remote_action("stage", self.root, payload)["exit"]["exit_code"], 0)
            self.assertEqual(run.call_count, 1)

    def test_stage_refuses_unowned_output_and_unowned_tmux(self):
        payload = self.stage_payload()
        Path(payload["output"]).mkdir(parents=True)
        with patch("subprocess.run", return_value=subprocess.CompletedProcess([], 1)) as run:
            with self.assertRaisesRegex(ValueError, "Unowned existing output"):
                remote_action("stage", self.root, payload)
            self.assertEqual(run.call_count, 1)
        payload["name"] = "different_stage"
        with patch("subprocess.run", return_value=subprocess.CompletedProcess([], 0)):
            with self.assertRaisesRegex(ValueError, "not owned"):
                remote_action("stage", self.root, payload)

    def test_strict_audit_gate_rejects_partial_bad_hash_and_cohort_counts(self):
        valid = dict(status="pass", complete=True, expected_predictions=57000, verified_predictions=57000,
                     protocol_sha256="protocol", failed_batches=[],
                     predictions_by_cohort=dict(supervised=45000, diffusion=12000),
                     coverage={str(i): dict(expected=1000, found=1000, missing=[]) for i in range(57)})
        validate_audit(dict(document=valid), "protocol")
        for change in (dict(status="partial"), dict(verified_predictions=56999), dict(protocol_sha256="wrong"),
                       dict(failed_batches=["bad"]), dict(predictions_by_cohort=dict(supervised=44000, diffusion=13000))):
            with self.assertRaises(ValueError):
                validate_audit(dict(document=dict(valid, **change)), "protocol")
        broken = copy.deepcopy(valid)
        broken["coverage"]["0"]["missing"] = [0]
        with self.assertRaises(ValueError):
            validate_audit(dict(document=broken), "protocol")

    def test_comparison_requires_full_formal_counts_and_bound_audit(self):
        valid = dict(status="complete", allow_partial=False, new_predictions=57000, expected_predictions=57000,
                     complete_cells=57, expected_cells=57, historical_comparison_rows=73, formal_historical_rows=73,
                     baseline_comparison_rows=203, formal_baseline_rows=203,
                     sources=dict(protocol=dict(sha256="protocol"), audit=dict(sha256="audit")))
        validate_comparison(dict(document=valid), "protocol", "audit")
        for change in (dict(allow_partial=True), dict(formal_baseline_rows=202), dict(complete_cells=56)):
            with self.assertRaises(ValueError):
                validate_comparison(dict(document=dict(valid, **change)), "protocol", "audit")
        with self.assertRaises(ValueError):
            validate_comparison(dict(document=valid), "protocol", "different_audit")

    def test_missing_or_partial_996_sensitivity_blocks_finalization(self):
        csv = dict(path="ns_smooth_996_summary.csv", sha256="csv")
        valid = dict(status="complete", complete=True, primary_audit_status="pass", primary_audit_complete=True,
                     protocol_sha256="protocol", expected_cells=8, verified_primary_predictions=8000,
                     verified_sensitivity_predictions=7968, exclude_source_indices=[197, 251, 364, 878],
                     frozen_physical_overlap_checks=[{}] * 32, csv=csv,
                     cells=[dict(cell_id=str(i), status="complete", primary_n=1000, n=996, excluded_verified_n=4)
                            for i in range(8)])
        binding = dict(status="complete", complete=True, json=dict(sha256="json"), csv=csv)
        audit = dict(document=dict(ns_smooth_selection_overlap_sensitivity=binding))
        validate_sensitivity(dict(document=valid, sha256="json"), audit, "protocol")
        for change in (dict(verified_sensitivity_predictions=7967), dict(status="partial_snapshot"),
                       dict(exclude_source_indices=[0, 1, 2, 3])):
            with self.assertRaises(ValueError):
                validate_sensitivity(dict(document=dict(valid, **change), sha256="json"), audit, "protocol")
        with self.assertRaises(ValueError):
            validate_sensitivity(dict(document=valid, sha256="changed"), audit, "protocol")

    def test_collector_stop_waits_for_successful_cycle_and_once_ignores_it(self):
        folder = self.root / "collector"
        folder.mkdir()
        (folder / "STOP_AFTER_CYCLE").touch()
        with patch.object(collect_remote.Collector, "cycle", side_effect=[1, 0]) as cycle, patch.object(collect_remote.time, "sleep") as sleep:
            result = collect_remote.main(["--local-root", str(self.root), "--mode", "watch"])
        self.assertEqual(result, 0)
        self.assertEqual(cycle.call_count, 2)
        self.assertEqual(sleep.call_count, 1)
        self.assertEqual(json.loads((folder / "status.json").read_text())["state"], "stopped")
        with patch.object(collect_remote.Collector, "cycle", return_value=1) as cycle:
            self.assertEqual(collect_remote.main(["--local-root", str(self.root), "--mode", "once"]), 1)
            self.assertEqual(cycle.call_count, 1)

    def test_metadata_relay_excludes_predictions_partials_and_preserves_existing(self):
        source, target = self.root / "metadata_source", self.root / "metadata_target"
        files = ["job/invocation_1.json", "job/cell/pilot_complete.json", "job/cell/batch_1/prediction.pt",
                 "job/cell/.partial_test/invocation_bad.json", "job/other.json"]
        for name in files:
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name)
        finalizer = Finalizer(SimpleNamespace(local_root=self.root, timeout_hours=1))
        finalizer.sync("synthetic_metadata", source, target,
                       patterns=["invocation_*.json", "pilot_complete.json"], immutable=True)
        copied = {str(p.relative_to(target)) for p in target.rglob("*") if p.is_file()}
        self.assertEqual(copied, set(files[:2]))
        preserved = target / files[0]
        preserved.write_text("existing artifact stays unchanged")
        finalizer.sync("synthetic_metadata_resume", source, target,
                       patterns=["invocation_*.json", "pilot_complete.json"], immutable=True)
        self.assertEqual(preserved.read_text(), "existing artifact stays unchanged")


if __name__ == "__main__":
    unittest.main()
