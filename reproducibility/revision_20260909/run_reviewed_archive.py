#!/usr/bin/env python3
"""Serial completed-entry archive and verification; no pending-state promotion."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
OUT = Path('/home/tat512/C01Python/audit/revision_archive_execution_20260909')
INVENTORY = OUT/'archive_inventory.reviewed.json'
REMOTE = '/research_data/users/zhangxifeng/C01Python/FM4PDEArchive0909_6f5bfd5/reproducibility/revision_20260909/archive_results.py'


def save(path, data):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,indent=2)+'\n')
    temporary.replace(path)


def main():
    inventory=json.loads(INVENTORY.read_text())
    fingerprint=hashlib.sha256(INVENTORY.read_bytes()).hexdigest()
    entries=[e for e in inventory['entries'] if e['status']=='ready' and e['role'] in ['primary','dependency']]
    target=OUT/'execution_status.json'
    state=json.loads(target.read_text()) if target.exists() else dict(inventory_sha256=fingerprint,entries={},status='running')
    assert state['inventory_sha256']==fingerprint, 'Reviewed inventory changed during the run'
    for entry in entries:
        name=entry['id']
        if state['entries'].get(name,{}).get('status')=='verified':
            continue
        record=dict(status='running',started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),source_host=entry['source_host'],destination_relative=entry['destination_relative'])
        state.update(status='running',current_entry=name);state['entries'][name]=record;save(target,state)
        for operation in ['execute','verify']:
            stdout=OUT/(name+'.'+operation+'.jsonl');stderr=OUT/(name+'.'+operation+'.stderr.log')
            if stdout.exists() or stderr.exists():
                record.update(status='held',reason='Existing attempt logs retained; inspect and explicitly resolve before retrying.');break
            command=[sys.executable,str(HERE/'archive_results.py'),'--inventory',str(INVENTORY),'--ids',name,'--remote-script',REMOTE,'--'+operation]
            with stdout.open('x') as output, stderr.open('x') as error:
                completed=subprocess.run(command,stdout=output,stderr=error)
            if completed.returncode:
                record.update(status='held',failed_operation=operation,returncode=completed.returncode,stderr=str(stderr));break
            lines=[json.loads(line) for line in stdout.read_text().splitlines() if line.startswith('{')]
            outcome=lines[-1]
            assert outcome['status'] in (['archived','already_verified'] if operation=='execute' else ['verified'])
            record[operation]=outcome
        else:
            assert record['execute']['receipt_sha256']==record['verify']['receipt_sha256']
            record.update(status='verified',bytes=record['verify']['bytes'],receipt_sha256=record['verify']['receipt_sha256'])
        record['finished_utc']=dt.datetime.now(dt.timezone.utc).isoformat()
        state['verified_entries']=sum(r['status']=='verified' for r in state['entries'].values())
        state['verified_bytes']=sum(r.get('bytes',0) for r in state['entries'].values() if r['status']=='verified')
        save(target,state)
        print(json.dumps({'entry':name,'status':record['status'],'verified_entries':state['verified_entries'],'verified_bytes':state['verified_bytes']}),flush=True)
    state.update(status='complete' if all(r['status']=='verified' for r in state['entries'].values()) and len(state['entries'])==len(entries) else 'complete_with_held_entries',current_entry=None,finished_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    save(target,state)


if __name__=='__main__':main()
