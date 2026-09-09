#!/usr/bin/env python3
"""Read-only SHA census of original FM tensors already under target outputs."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
AUDIT = Path('/home/tat512/C01Python/audit/paper_revision_20260908')
ROOT = '/research_data/users/zhangxifeng/C01Python/FM4PDE/outputs/'
REMOTE = r'''
import hashlib,json,sys
from pathlib import Path
rows=json.load(sys.stdin)
for row in rows:
 p=Path(row['source_path']);row['exists']=p.is_file()
 if not row['exists']:continue
 before=p.stat();h=hashlib.sha256()
 with p.open('rb') as f:
  for data in iter(lambda:f.read(8<<20),b''):h.update(data)
 after=p.stat()
 if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise RuntimeError('Source changed: '+str(p))
 row.update(bytes=after.st_size,sha256=h.hexdigest())
 row['matches_expected']=row.get('expected_sha256',row['sha256'])==row['sha256']
print(json.dumps(rows))
'''


def main():
    rows = []
    snapshot = AUDIT/'ablation_publication_snapshot/records.json'
    manifest = AUDIT/'burgers_fm_random/fm_random_manifest.json'
    for record in json.loads(snapshot.read_text()):
        if record['source'] == 'revised':
            continue
        row = dict(study='unchanged_ablations',id=record['id'],pde=record['pde'],source_path=record['source_result'])
        if record['result_path'] and Path(record['result_path']).is_file():
            row['expected_sha256'] = hashlib.sha256(Path(record['result_path']).read_bytes()).hexdigest()
            row['expected_from'] = record['result_path']
        rows.append(row)
    for path, digest in json.loads(manifest.read_text())['runs'].items():
        rows.append(dict(study='burgers_original_random',source_path=path,expected_sha256=digest,expected_from=str(manifest)))
    if any(not r['source_path'].startswith(ROOT) or '/..' in r['source_path'] for r in rows):
        raise ValueError('Reference escaped the expected existing FM output root')
    result = subprocess.run(['ssh','server197',shlex.join(['/usr/bin/python3','-c',REMOTE])],
                            input=json.dumps(rows),capture_output=True,text=True,check=True)
    observed = json.loads(result.stdout)
    report = dict(observed_utc=dt.datetime.now(dt.timezone.utc).isoformat(), source_host='server197',
                  status='passed' if all(r.get('exists') and r.get('matches_expected') for r in observed) else 'failed',
                  snapshot_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                  burgers_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                  note='Original files remain in place. SHA256 is observed for all referenced tensors; where a local copy or frozen manifest supplies an expected hash, it is compared explicitly.',
                  counts={s:sum(r['study']==s for r in observed) for s in sorted({r['study'] for r in observed})},
                  rows=observed)
    (HERE/'in_place_reference_checks.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
    if report['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':main()
