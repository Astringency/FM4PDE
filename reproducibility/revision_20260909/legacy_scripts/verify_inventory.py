"""Verify preserved source bytes and syntax without executing study code."""
from pathlib import Path
import ast
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / 'reproducibility/revision_20260909/external_script_inventory.json'


def main():
    manifest = json.loads(MANIFEST.read_text())
    checked, source_changes, unavailable = 0, [], []
    for row in manifest['copied']:
        path = ROOT / row['destination']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['copy_sha256'], path
        assert row['copy_sha256'] == row['source_sha256'], path
        if path.suffix == '.py':
            ast.parse(path.read_text(), filename=str(path))
        elif path.suffix == '.sh':
            subprocess.run(['bash', '-n', str(path)], check=True)
        source = Path(row['source_path'])
        if source.is_file():
            if hashlib.sha256(source.read_bytes()).hexdigest() != row['source_sha256']:
                source_changes.append(str(source))
        else:
            unavailable.append(str(source))
        checked += 1
    print(json.dumps(dict(status='pass',preserved_files=checked,
                         source_paths_changed_since_snapshot=source_changes,
                         original_source_paths_unavailable=unavailable,
                         study_scripts_executed=False, gpu_or_remote_actions=False), indent=2))


if __name__ == '__main__':
    main()
