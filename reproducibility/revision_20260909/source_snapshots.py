#!/usr/bin/env python3
"""Preserve and verify source trees and Git history used by the revision.

Only committed files enter the archives. Existing archives are verified before
reuse; differing files are never replaced. No source checkout is modified.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()


def write_new(path, data):
    payload = (json.dumps(data, indent=2, sort_keys=True) + '\n').encode()
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f'Refusing to replace different manifest: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        f.write(payload)


def atomic_install(source, target):
    """Exclusive creation by hard link on the destination filesystem."""
    try:
        os.link(source, target)
    except FileExistsError:
        if sha256(source) != sha256(target):
            raise RuntimeError(f'Refusing to replace different archive: {target}')


def verify_record(root, record):
    path = root / record['file']
    if not path.is_file() or path.stat().st_size != record['bytes']:
        raise RuntimeError(f'Missing or size mismatch: {path}')
    if sha256(path) != record['sha256']:
        raise RuntimeError(f'Checksum mismatch: {path}')


def verify_catalog_coverage(catalog, records):
    """Require every named source version, even if other archives are intact."""
    trees = {(r['repository'], r['commit']): r for r in records if r.get('tree')}
    histories = {r['repository'] for r in records
                 if r.get('format') == 'self-contained Git bundle'}
    required = 0
    for repository in catalog['repositories']:
        name = repository['name']
        for snapshot in repository['snapshots']:
            revision = snapshot['revision']
            matches = [r for (repo, commit), r in trees.items()
                       if repo == name and commit.startswith(revision)]
            if len(matches) != 1:
                raise RuntimeError(f'Missing or ambiguous required source: {name}@{revision}')
            required += 1
        if repository.get('retain_history') and name not in histories:
            raise RuntimeError(f'Missing required Git history: {name}')
    return required


def verify_history_coverage(archives, catalog, validation, restore_record):
    """Recheck the frozen bundle selection without relying on original repos."""
    prior = json.loads(restore_record.read_text())
    selected = prior['bundles']
    names = [bundle['repository'] for bundle in selected]
    required = [repo['name'] for repo in json.loads(catalog.read_text())['repositories']]
    if len(set(names)) != len(names) or set(names) != set(required):
        raise RuntimeError('Frozen restore record does not cover each source repository')
    arguments = []
    for bundle in selected:
        verify_record(archives, bundle)
        sidecar = archives / Path(bundle['file']).with_suffix('.json')
        record = json.loads(sidecar.read_text())
        if any(record[key] != bundle[key]
               for key in ('repository', 'file', 'bytes', 'sha256')):
            raise RuntimeError(f'Frozen history selection differs from sidecar: {sidecar}')
        arguments.extend(['--bundle', bundle['repository'] + '=' + sidecar.name])
    # The prior report selects immutable bundles; all restore checks run afresh.
    # Private clones are discarded when verification finishes, never source data.
    with tempfile.TemporaryDirectory(prefix='fm4pde-source-verify-') as directory:
        temporary = Path(directory)
        output = temporary / 'coverage.json'
        subprocess.run([sys.executable, str(HERE / 'verify_source_restore.py'),
                        '--archives', str(archives), '--catalog', str(catalog.resolve()),
                        '--validation', str(validation.resolve()),
                        '--work', str(temporary / 'clones'), '--output', str(output),
                        *arguments], check=True)
        result = json.loads(output.read_text())
        return result['source_trees']


def capture_tree(repo, spec, dest):
    commit = git(repo, 'rev-parse', '--verify', spec['revision'] + '^{commit}')
    name = spec['name'] + '-' + commit[:12]
    target = dest / (name + '.tar.gz')
    sidecar = dest / (name + '.json')
    if sidecar.exists():
        record = json.loads(sidecar.read_text())
        if record['commit'] != commit:
            raise RuntimeError(f'Commit mismatch: {sidecar}')
        verify_record(dest, record)
        return record
    # Reject absent submodule content: a commit pointer is not runnable source.
    tree = git(repo, 'ls-tree', '-r', commit)
    if any(line.startswith('160000 ') for line in tree.splitlines()):
        raise RuntimeError(f'Unresolved Git submodules in {name}')
    with tempfile.TemporaryDirectory(dir=dest, prefix='.source-') as staging:
        raw = Path(staging) / 'tree.tar'
        subprocess.run(['git', '-C', str(repo), 'archive', '--format=tar',
                        '--prefix=' + name + '/', '--output=' + str(raw), commit], check=True)
        packed = Path(staging) / 'tree.tar.gz'
        with raw.open('rb') as src, packed.open('wb') as out:
            with gzip.GzipFile(filename='', mode='wb', fileobj=out, mtime=0) as zipped:
                for block in iter(lambda: src.read(2**20), b''):
                    zipped.write(block)
        atomic_install(packed, target)
    record = dict(file=target.name, sha256=sha256(target), bytes=target.stat().st_size,
                  repository=spec['name'], commit=commit,
                  tree=git(repo, 'rev-parse', commit + '^{tree}'),
                  purpose=spec['purpose'], source_location=str(repo.resolve()),
                  format='git archive tar, deterministic gzip; tracked tree only')
    write_new(sidecar, record)
    return record


def capture_bundle(repo, spec, dest):
    # A dated bundle preserves historical producer versions independently of
    # future branch changes. It intentionally does not include the worktree.
    head = git(repo, 'rev-parse', 'HEAD')
    name = spec['name'] + '-history-through-' + head[:12]
    target = dest / (name + '.bundle')
    sidecar = dest / (name + '.json')
    if sidecar.exists():
        record = json.loads(sidecar.read_text())
        verify_record(dest, record)
        return record
    with tempfile.TemporaryDirectory(dir=dest, prefix='.history-') as staging:
        tmp = Path(staging) / 'source.bundle'
        subprocess.run(['git', '-C', str(repo), 'bundle', 'create', str(tmp), '--all'],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        heads = git(repo, 'bundle', 'list-heads', str(tmp))
        if head not in heads:
            raise RuntimeError('Source HEAD changed while recording Git history; retry.')
        subprocess.run(['git', '-C', str(repo), 'bundle', 'verify', str(tmp)],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        atomic_install(tmp, target)
    record = dict(file=target.name, sha256=sha256(target), bytes=target.stat().st_size,
                  repository=spec['name'], observed_head=head, refs=heads.splitlines(),
                  purpose='Complete reachable Git history, including earlier experiment versions',
                  source_location=str(repo.resolve()), format='self-contained Git bundle')
    write_new(sidecar, record)
    return record


def restore_tree(archive, destination):
    record = json.loads(archive.with_suffix('').with_suffix('.json').read_text())
    verify_record(archive.parent, record)
    if not record.get('tree') or archive.name != record['file']:
        raise RuntimeError('Select a source-tree .tar.gz archive with its adjacent manifest')
    if destination.exists():
        raise FileExistsError(f'Restore requires a new destination: {destination}')
    bundles = []
    for path in archive.parent.glob('*.json'):
        candidate = json.loads(path.read_text())
        if candidate.get('repository') == record['repository'] and candidate.get('format') == 'self-contained Git bundle':
            bundles.append((path.stat().st_mtime, candidate))
    if not bundles:
        raise RuntimeError('A matching Git history bundle is required to preserve the producer Git identity')
    destination.parent.mkdir(parents=True, exist_ok=True)
    for _, bundle in sorted(bundles, key=lambda pair: pair[0], reverse=True):
        verify_record(archive.parent, bundle)
        with tempfile.TemporaryDirectory(dir=destination.parent, prefix='.restore-') as staging:
            checkout = Path(staging) / 'checkout'
            subprocess.run(['git', 'clone', '--quiet', '--no-checkout',
                            str(archive.parent / bundle['file']), str(checkout)], check=True)
            exists = subprocess.run(['git', '-C', str(checkout), 'cat-file', '-e',
                                     record['commit'] + '^{commit}'], capture_output=True)
            if exists.returncode:
                continue
            subprocess.run(['git', '-C', str(checkout), 'checkout', '--quiet', '--detach',
                            record['commit']], check=True)
            if git(checkout, 'rev-parse', 'HEAD') != record['commit'] or git(checkout, 'rev-parse', 'HEAD^{tree}') != record['tree']:
                raise RuntimeError('Restored Git identity or tree does not match the source archive')
            files = len(git(checkout, 'ls-files').splitlines())
            write_new(checkout / '.FM4PDE_SOURCE.json', dict(record, history_bundle=bundle['file']))
            if destination.exists():
                raise FileExistsError(f'Destination was created during restore: {destination}')
            os.rename(checkout, destination)
            return dict(status='pass', repository=record['repository'], commit=record['commit'],
                        destination=str(destination), files=files, git_identity_preserved=True)
    raise RuntimeError('No verified Git bundle contains the required producer commit')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['capture', 'verify', 'restore'])
    p.add_argument('--catalog', type=Path, default=HERE / 'source_repositories.json')
    p.add_argument('--output', type=Path, default=HERE / 'source_snapshots')
    p.add_argument('--repository', action='append', default=[], metavar='NAME=PATH',
                   help='Override a source repository location without changing the catalog')
    p.add_argument('--archive', type=Path, help='Source-tree .tar.gz to restore')
    p.add_argument('--destination', type=Path, help='New source checkout directory for restore')
    p.add_argument('--validation', type=Path, default=HERE / 'source_validation.json',
                   help='Recorded complete comparison of tar contents with Git objects')
    p.add_argument('--restore-record', type=Path, default=HERE / 'source_restore_coverage.json',
                   help='Frozen bundle selection; verify repeats the independent restore checks')
    args = p.parse_args()
    if args.mode == 'restore':
        if args.archive is None or args.destination is None:
            p.error('restore requires --archive and --destination')
        print(json.dumps(restore_tree(args.archive.resolve(), args.destination.resolve())))
        return
    overrides = dict(item.split('=', 1) for item in args.repository)
    dest = args.output.resolve()
    if args.mode == 'verify':
        manifests = sorted(dest.glob('*.json'))
        if not manifests:
            raise RuntimeError('No source archives to verify')
        records = [json.loads(path.read_text()) for path in manifests]
        for record in records:
            verify_record(dest, record)
        required = verify_catalog_coverage(json.loads(args.catalog.read_text()), records)
        restored = verify_history_coverage(dest, args.catalog, args.validation,
                                           args.restore_record)
        if restored != required:
            raise RuntimeError('Restored version count differs from required source catalog')
        print(json.dumps(dict(status='pass', archives=len(manifests),
                              required_source_versions=required, catalog_coverage=True,
                              independent_bundle_restore=True)))
        return
    dest.mkdir(parents=True, exist_ok=True)
    catalog = json.loads(args.catalog.read_text())
    records = []
    for spec in catalog['repositories']:
        repo = Path(overrides.get(spec['name'], ROOT / spec['location_relative_to_fm4pde']))
        for revision in spec['snapshots']:
            records.append(capture_tree(repo, dict(name=spec['name'], **revision), dest))
        if spec.get('retain_history'):
            records.append(capture_bundle(repo, spec, dest))
    print(json.dumps(dict(status='pass', archives=len(records),
                          total_bytes=sum(x['bytes'] for x in records))))


if __name__ == '__main__':
    main()
