#!/usr/bin/env python3
"""Compare archived source bytes and executable modes with their Git objects."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archives', type=Path, default=HERE / 'source_snapshots')
    parser.add_argument('--output', type=Path, default=HERE / 'source_validation.json')
    parser.add_argument('--repository', action='append', default=[], metavar='NAME=PATH')
    args = parser.parse_args()
    overrides = dict(value.split('=', 1) for value in args.repository)
    reports = []
    for meta in sorted(args.archives.glob('*.json')):
        record = json.loads(meta.read_text())
        if 'tree' not in record:
            continue
        repo = Path(overrides.get(record['repository'], record['source_location']))
        raw = subprocess.check_output(['git', '-C', str(repo), 'ls-tree', '-rz', record['commit']])
        algorithm = subprocess.check_output(['git', '-C', str(repo), 'rev-parse',
                                             '--show-object-format'], text=True).strip()
        expected = {}
        for row in raw.split(b'\0'):
            if not row:
                continue
            description, name = row.split(b'\t', 1)
            mode, kind, obj = description.split()
            assert kind == b'blob', 'Submodules require their own complete source archive'
            expected[name.decode()] = (mode.decode(), obj.decode())
        actual = {}
        with tarfile.open(meta.parent / record['file']) as contents:
            for member in contents:
                if member.isdir():
                    continue
                relative = member.name.split('/', 1)[1]
                assert member.isfile()
                with contents.extractfile(member) as source:
                    data = source.read()
                digest = hashlib.new(algorithm, b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
                actual[relative] = digest
                assert digest == expected[relative][1], (record['commit'], relative)
                assert bool(member.mode & 0o111) == (expected[relative][0] == '100755')
        assert set(actual) == set(expected), (record['commit'], set(expected) - set(actual))
        reports.append(dict(repository=record['repository'], commit=record['commit'],
                            files=len(actual), all_git_blob_hashes_match=True,
                            executable_modes_match=True, archive_sha256=record['sha256']))
    assert reports, 'No source-tree archives were checked'
    result = dict(status='pass', source_trees=len(reports), files=sum(x['files'] for x in reports),
                  checks='Independent tar contents compared to every Git blob object hash and executable mode',
                  records=reports)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}))


if __name__ == '__main__':
    main()
