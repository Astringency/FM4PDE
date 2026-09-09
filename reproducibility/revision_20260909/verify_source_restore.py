#!/usr/bin/env python3
"""Check catalogued source identities in independent clones of saved bundles."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def main(args):
    assert not args.work.exists() and not args.output.exists()
    args.work.mkdir(parents=True)
    catalog_sha = sha(args.catalog)
    validation_sha = sha(args.validation)
    catalog = json.loads(args.catalog.read_text())
    validation = json.loads(args.validation.read_text())
    assert validation['status'] == 'pass'
    validated = {(r['repository'], r['commit']): r for r in validation['records']}
    required = sum(len(repo['snapshots']) for repo in catalog['repositories'])
    assert required == validation['source_trees'] == len(validated)
    selected = dict(item.split('=', 1) for item in args.bundle)
    assert set(selected) == {r['name'] for r in catalog['repositories']}
    environment = dict(os.environ)
    # No local object store, alternates or worktree may satisfy missing objects.
    for key in list(environment):
        if key.startswith('GIT_'):
            del environment[key]
    records, bundles = [], []
    input_hashes = {str(args.catalog): catalog_sha, str(args.validation): validation_sha}
    for repo in catalog['repositories']:
        name = repo['name']
        sidecar = args.archives / selected[name]
        bundle = json.loads(sidecar.read_text())
        assert bundle['repository'] == name
        assert bundle['format'] == 'self-contained Git bundle'
        file = args.archives / bundle['file']
        assert file.stat().st_size == bundle['bytes'] and sha(file) == bundle['sha256']
        input_hashes[str(sidecar)] = sha(sidecar)
        input_hashes[str(file)] = bundle['sha256']
        clone = args.work / name
        command = ['git', 'clone', '--no-checkout', '--no-hardlinks', str(file.resolve()), str(clone.resolve())]
        with (args.work / (name + '_clone.log')).open('w') as log:
            subprocess.run(command, env=environment, check=True, stdout=log, stderr=subprocess.STDOUT)
        assert not (clone / '.git/objects/info/alternates').exists()
        assert sorted(p.name for p in clone.iterdir()) == ['.git']
        with (args.work / (name + '_fsck.log')).open('w') as log:
            subprocess.run(['git', '-C', str(clone), 'fsck', '--full', '--strict', '--no-reflogs'],
                           env=environment, check=True, stdout=log, stderr=subprocess.STDOUT)
        def git(*arguments):
            return subprocess.check_output(['git', '-C', str(clone), *arguments],
                                           env=environment, text=True).strip()
        observed_head = git('rev-parse', 'HEAD')
        assert observed_head == bundle['observed_head']
        for spec in repo['snapshots']:
            revision = spec['revision']
            commit = git('rev-parse', '--verify', revision + '^{commit}')
            meta = args.archives / (name + '-' + commit[:12] + '.json')
            tree = json.loads(meta.read_text())
            assert tree['repository'] == name and tree['commit'] == commit
            assert tree['sha256'] == validated[name, commit]['archive_sha256']
            assert validated[name, commit]['all_git_blob_hashes_match']
            assert validated[name, commit]['executable_modes_match']
            tar = args.archives / tree['file']
            assert tar.stat().st_size == tree['bytes'] and sha(tar) == tree['sha256']
            restored_tree = git('rev-parse', '--verify', commit + '^{tree}')
            assert restored_tree == tree['tree']
            input_hashes[str(meta)] = sha(meta)
            input_hashes[str(tar)] = tree['sha256']
            records.append(dict(repository=name, commit=commit, purpose=spec['purpose'],
                bundle=bundle['file'], bundle_sha256=bundle['sha256'],
                bundle_sidecar_sha256=input_hashes[str(sidecar)], clone=str(clone.resolve()),
                commit_exists=True, tree=restored_tree, tree_matches_tar_sidecar=True,
                archive=tree['file'], archive_sha256=tree['sha256'],
                tar_sidecar_sha256=input_hashes[str(meta)],
                prior_full_tar_blob_and_mode_validation=True, status='pass'))
        bundles.append(dict(repository=name, file=bundle['file'], sha256=bundle['sha256'],
            bytes=bundle['bytes'], observed_head=observed_head, command=command,
            self_contained_clone_pass=True, no_alternates=True, no_worktree_materialized=True,
            full_strict_git_fsck_pass=True,
            clone_log_sha256=sha(args.work/(name+'_clone.log')),
            fsck_log_sha256=sha(args.work/(name+'_fsck.log'))))
        print('RESTORED', name, len(repo['snapshots']), flush=True)
    assert len(records) == required
    for path, expected in input_hashes.items():
        assert sha(path) == expected, ('Input changed during restore check', path)
    output = dict(status='pass', complete=True,
        completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_trees=required, repositories=len(bundles), independent_clones=len(bundles),
        worktrees_materialized=0, models_executed=0, original_sources_and_archives_unchanged=True,
        catalog_sha256=catalog_sha, source_validation_sha256=validation_sha,
        script_sha256=sha(__file__), git_version=subprocess.check_output(['git','--version'],text=True).strip(),
        bundles=bundles, records=records,
        interpretation='All catalogued commits and tree identities were recovered from independent self-contained Git-bundle clones. Matching tar archives retain the separately verified Git-blob bytes and executable modes. This check does not imply that every historical source identity is known for every experiment.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2)+'\n')
    print('SOURCE_RESTORE_COVERAGE_PASS', required, sha(args.output), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, default=HERE/'source_repositories.json')
    parser.add_argument('--validation', type=Path, default=HERE/'source_validation.json')
    parser.add_argument('--archives', type=Path, default=HERE/'source_snapshots')
    parser.add_argument('--bundle', action='append', required=True, help='REPOSITORY=SIDECAR.json')
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=HERE/'source_restore_coverage.json')
    main(parser.parse_args())
