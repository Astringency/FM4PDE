#!/usr/bin/env python3
"""Organize finished, explicitly selected results without changing their sources."""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile

PROJECTS = {'FM4PDE', 'FM4PDEbaseline', 'DiffusionPDE'}
RECEIPT = '.organization_receipt.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def snapshot(path):
    s = path.stat()
    return dict(bytes=s.st_size, mtime_ns=s.st_mtime_ns, inode=s.st_ino,
                device=s.st_dev, mode=s.st_mode & 0o777)


def safe_path(base, relative):
    rel = Path(relative)
    if rel.is_absolute() or '..' in rel.parts or str(rel) != relative:
        raise ValueError(f'Unsafe relative path: {relative}')
    p = base / rel
    for parent in (p, *p.parents):
        if parent == base:
            break
        if parent.is_symlink():
            raise ValueError(f'Symlink in selected path: {parent}')
    return p


def selected_files(base, selected):
    found = {}
    for spec in selected:
        p = safe_path(base, spec['source'])
        if not p.exists():
            raise FileNotFoundError(p)
        paths = [p] if p.is_file() else sorted(p.rglob('*'))
        for f in paths:
            relative = f.relative_to(base).as_posix()
            if any(part in ('.git', '__pycache__', '.pytest_cache', '.ipynb_checkpoints')
                   for part in f.relative_to(p.parent).parts):
                continue
            if f.is_symlink():
                raise ValueError(f'Symlink requires explicit review: {f}')
            if f.is_dir():
                continue
            if not f.is_file():
                raise ValueError(f'Unsupported source node: {f}')
            if relative in found:
                raise ValueError(f'Overlapping selections: {relative}')
            found[relative] = (spec, snapshot(f))
    return found


def route(relative, spec):
    """Method names in result paths override a study's shared-file assignment."""
    projects = spec['projects']
    if len(projects) == 1:
        return projects
    source = Path(spec['source'])
    rel = Path(relative).relative_to(source) if relative != str(source) else Path(source.name)
    parts = [p.lower() for p in rel.parts]
    if 'DiffusionPDE' in projects and any(
            p.startswith(('diffusionpde', 'diffusion_', 'effective_sampler', 'pilot_reference'))
            for p in parts):
        return ['DiffusionPDE']
    if 'FM4PDE' in projects and any(p.startswith('fm4pde') or p == 'fm' for p in parts):
        return ['FM4PDE']
    baseline_names = ('voronoicnn', 'senseiver', 'recfno', 'pde_opt', 'vivid',
                      'deeponet', 'ifno', 'fno')
    if 'FM4PDEbaseline' in projects and any(
            p == name or p.startswith(name + '_') or p.startswith('pilot_' + name)
            for p in parts for name in baseline_names):
        return ['FM4PDEbaseline']
    if 'FM4PDE' in projects and any(p == 'reference' for p in parts):
        return ['FM4PDE']
    return projects


def make_plan(specs):
    base = Path(specs['base']).resolve(strict=True)
    records, groups = [], collections.Counter()
    for relative, (spec, stat) in selected_files(base, specs['selected']).items():
        for project in route(relative, spec):
            if project not in PROJECTS:
                raise ValueError(project)
            category = spec.get('category', 'main') if project == 'FM4PDE' else ''
            reuse = spec.get('reuse', {}).get(project)
            if reuse:
                remainder = Path(relative).relative_to(spec['source'])
                destination = (Path(reuse) / remainder).as_posix()
                archive_root = None
            else:
                archive_root = (Path(project) / 'outputs' / category /
                                'organized_20260910').as_posix()
                destination = archive_root + '/' + relative
            safe_path(base, destination)
            record = dict(source=relative, destination=destination, project=project,
                          category=category, archive_root=archive_root, reused=bool(reuse),
                          source_stat=stat)
            records.append(record)
            groups[(project, category, bool(reuse))] += stat['bytes']
    dests = [r['destination'] for r in records]
    if len(set(dests)) != len(dests):
        raise ValueError('Duplicate destinations')
    return dict(schema_version=1, host=specs['host'], base=str(base),
                created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                script_sha256=sha(__file__), scope=specs['scope'],
                selected=specs['selected'], retained=specs['retained'], files=records,
                unique_source_files=len({r['source'] for r in records}),
                destination_files=len(records), logical_bytes=sum(r['source_stat']['bytes'] for r in records),
                groups=[dict(project=k[0], category=k[1], reused=k[2], bytes=v)
                        for k, v in sorted(groups.items())])


def apply_plan(plan, plan_path, report_dir):
    base = Path(plan['base']).resolve(strict=True)
    if sha(__file__) != plan['script_sha256']:
        raise ValueError('Organizer differs from the reviewed plan')
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / 'organizer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        expected = {r['source']: r['source_stat'] for r in plan['files']}
        current = {k: v[1] for k, v in selected_files(base, plan['selected']).items()}
        if current != expected:
            raise RuntimeError('Source files changed after planning')
        groups = collections.defaultdict(list)
        for r in plan['files']:
            groups[r['archive_root']].append(r)
        required = sum(r['source_stat']['bytes'] for r in plan['files'] if not r['reused'])
        if shutil.disk_usage(base).free < required + (5 << 30):
            raise RuntimeError('Insufficient disk space for independent copies and reserve')
        write_new(report_dir / 'resources.json', dict(
            hostname=socket.gethostname(), cpu_load=os.getloadavg(),
            meminfo=Path('/proc/meminfo').read_text(), free_bytes=shutil.disk_usage(base).free,
            planned_copy_bytes=required, gpu_work=False))
        source_hashes, receipts = {}, []
        for root_relative, records in groups.items():
            final = safe_path(base, root_relative) if root_relative else None
            stage = None
            existing = final is not None and final.exists()
            retained_hashes = {}
            if existing:
                receipt_path = final / RECEIPT
                if not receipt_path.is_file():
                    raise FileExistsError(f'Existing result folder left unchanged: {final}')
                receipt = json.loads(receipt_path.read_text())
                # A corrected source-location mapping may leave an already copied
                # group unchanged. Preserve its receipt and verify its exact file
                # mapping and contents before accepting that completed group.
                expected_mapping = {(r['source'], r['destination'], r['source_stat']['bytes'])
                                    for r in records}
                retained_mapping = {(r['source'], r['destination'], r['bytes'])
                                    for r in receipt['files']}
                if (receipt.get('status') != 'pass' or
                        receipt.get('project_root') != root_relative or
                        retained_mapping != expected_mapping or
                        len(receipt['files']) != len(records)):
                    raise ValueError(f'Existing result folder differs from this plan: {final}')
                retained_hashes = {r['destination']: r['sha256'] for r in receipt['files']}
            elif final is not None:
                final.parent.mkdir(parents=True, exist_ok=True)
                stage = Path(tempfile.mkdtemp(prefix='.organized_20260910-', dir=final.parent))
            verified = []
            for number, r in enumerate(records, 1):
                src = safe_path(base, r['source'])
                if snapshot(src) != r['source_stat']:
                    raise RuntimeError(f'Source changed: {src}')
                before_hash = source_hashes.setdefault(r['source'], sha(src))
                if existing and retained_hashes[r['destination']] != before_hash:
                    raise RuntimeError(f'Existing receipt differs from source contents: {src}')
                if stage is not None:
                    relative = Path(r['destination']).relative_to(root_relative)
                    dest = stage / relative
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with src.open('rb') as source, dest.open('xb') as target:
                        shutil.copyfileobj(source, target, 8 << 20)
                    shutil.copystat(src, dest, follow_symlinks=False)
                else:
                    dest = safe_path(base, r['destination'])
                if not dest.is_file() or sha(dest) != before_hash:
                    raise RuntimeError(f'Copy mismatch or missing existing result: {dest}')
                if snapshot(src) != r['source_stat']:
                    raise RuntimeError(f'Source changed during copying: {src}')
                if (dest.stat().st_mode & 0o777) != r['source_stat']['mode']:
                    raise RuntimeError(f'File permissions differ: {dest}')
                verified.append(dict(source=r['source'], destination=r['destination'],
                                     bytes=r['source_stat']['bytes'], sha256=before_hash))
                if number % 2000 == 0:
                    print(json.dumps(dict(stage='copy_and_verify', root=root_relative,
                                          files=number, total=len(records))), flush=True)
            result = dict(status='pass', project_root=root_relative,
                          plan_sha256=sha(plan_path), script_sha256=sha(__file__),
                          source_preserved=True, files=verified)
            if stage is not None:
                write_new(stage / RECEIPT, result)
                if final.exists():
                    raise FileExistsError(f'Concurrent destination creation: {final}')
                stage.rename(final)
            if final is not None:
                receipts.append(dict(path=str(final / RECEIPT), sha256=sha(final / RECEIPT),
                                     files=len(records), bytes=sum(x['bytes'] for x in verified)))
            else:
                write_new(report_dir / 'existing_archive_verification.json', result)
            print(json.dumps(dict(stage='group_verified', root=root_relative,
                                  files=len(records))), flush=True)
        after = {k: v[1] for k, v in selected_files(base, plan['selected']).items()}
        if after != expected:
            raise RuntimeError('Source file list or metadata changed during organization')
        for number, (relative, expected_sha) in enumerate(source_hashes.items(), 1):
            if sha(safe_path(base, relative)) != expected_sha:
                raise RuntimeError(f'Source content changed: {relative}')
            if number % 10000 == 0:
                print(json.dumps(dict(stage='source_recheck', files=number,
                                      total=len(source_hashes))), flush=True)
        write_new(report_dir / 'source_hashes.json', source_hashes)
        report = dict(status='complete', host=plan['host'], base=str(base),
                      completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                      plan_sha256=sha(plan_path), script_sha256=sha(__file__),
                      unique_source_files=len(expected), destination_files=len(plan['files']),
                      copied_files=sum(not r['reused'] for r in plan['files']),
                      reused_files=sum(r['reused'] for r in plan['files']),
                      copied_bytes=required, logical_destination_bytes=plan['logical_bytes'],
                      source_hashes_unchanged=True, original_sources_preserved=True,
                      all_destination_hashes_match=True, receipts=receipts,
                      source_hash_manifest_sha256=sha(report_dir / 'source_hashes.json'))
        write_new(report_dir / 'verification_complete.json', report)
        print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['plan', 'apply'])
    parser.add_argument('--spec', type=Path)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--report-dir', type=Path)
    args = parser.parse_args()
    if args.mode == 'plan':
        plan = make_plan(json.loads(args.spec.read_text()))
        write_new(args.plan, plan)
        print(json.dumps({k: v for k, v in plan.items()
                          if k not in ('files', 'selected', 'retained')}))
    else:
        apply_plan(json.loads(args.plan.read_text()), args.plan, args.report_dir)


if __name__ == '__main__':
    main()
