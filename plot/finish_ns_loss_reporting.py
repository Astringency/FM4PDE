"""Generate formal NS reports only after the collector's complete audit exists."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    args.study.mkdir(parents=True, exist_ok=True)
    audit = args.study / 'ns_complete_audit'
    manifest_path = audit / 'ns_audit_manifest.json'
    progress = args.study / 'ns_reporting_progress.json'

    def record(status, **details):
        value = dict(status=status, checked_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
                     **details)
        temp = progress.with_suffix('.json.tmp')
        temp.write_text(json.dumps(value, indent=2) + '\n')
        temp.replace(progress)
        print(json.dumps(value), flush=True)

    while True:
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        ready = bool(manifest and manifest['status'] == 'complete' and manifest['calls_verified'] == 1728)
        if not ready:
            record('waiting_for_complete_audit', calls_verified=manifest.get('calls_verified', 0) if manifest else 0)
            if not args.watch:
                return
            time.sleep(600)  # Own background tmux; no interactive tool is blocked.
            continue
        try:
            # Failed outcomes remain in the audit and require an explicit report
            # before publication. Do not export conditional successes silently.
            assert manifest['outcome_counts'] == {'complete': 1728}
            record('reporting', calls_verified=1728, audit_manifest_sha256=sha(manifest_path))
            root = Path(__file__).parent
            report = args.study / 'ns_complete_report'
            guidance = args.study / 'ns_guidance_report'
            subprocess.run([sys.executable, str(root / 'export_ns_loss_spectra.py'),
                            '--audit', str(audit), '--inputs', str(args.study / 'inputs_v2'),
                            '--output', str(report)], check=True)
            subprocess.run([sys.executable, str(root / 'plot_ns_guidance_traces.py'),
                            '--audit', str(audit), '--output', str(guidance)], check=True)
            report_manifest = report / 'ns_report_manifest.json'
            guidance_manifest = guidance / 'ns_guidance_plot_manifest.json'
            for path in [report_manifest, guidance_manifest]:
                m = json.loads(path.read_text())
                assert m['status'] == 'complete' and m['calls_verified'] == 1728
                for name, expected in m['outputs'].items():
                    assert sha(path.parent / name) == expected, name
            record('reports_complete', calls_verified=1728, audit_manifest_sha256=sha(manifest_path),
                   report_manifest_sha256=sha(report_manifest), guidance_manifest_sha256=sha(guidance_manifest),
                   remaining='Visual review, source-backed narrative, active paper/response integration, and full builds.')
            return
        except Exception as exc:
            record('reporting_failed', error=repr(exc), source_jobs_unchanged=True)
            raise


if __name__ == '__main__':
    main()
