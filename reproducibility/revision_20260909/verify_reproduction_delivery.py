#!/usr/bin/env python3
"""Verify the bytes of a separately delivered reproduction package."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True,
                        help='FM4PDE repository containing the delivered files')
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--allow-missing', action='store_true',
                        help='Before copying, require all existing files to match')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    manifest = json.loads(args.manifest.read_text())
    seen, missing = set(), []
    checked, size = 0, 0
    for record in manifest['files']:
        relative = PurePosixPath(record['path'])
        if (relative.is_absolute() or '..' in relative.parts or
                str(relative) != record['path'] or str(relative) in seen):
            raise ValueError(f'Invalid or duplicate relative path: {relative}')
        seen.add(str(relative))
        path = root.joinpath(*relative.parts)
        for part in (path, *path.parents):
            if part == root:
                break
            if part.is_symlink():
                raise ValueError(f'Symbolic link in delivery path: {part}')
        if not path.exists():
            missing.append(str(relative))
            continue
        if (not path.is_file() or path.stat().st_size != record['bytes'] or
                sha256(path) != record['sha256'] or
                bool(path.stat().st_mode & 0o111) != record['executable']):
            raise ValueError(f'Existing file differs from delivery manifest: {path}')
        checked += 1
        size += record['bytes']
    if missing and not args.allow_missing:
        raise FileNotFoundError(f'{len(missing)} required files missing; first: {missing[0]}')
    result = dict(status='pass', mode='preflight' if args.allow_missing else 'verify',
                  root=str(root), manifest_sha256=sha256(args.manifest),
                  required_files=len(seen), verified_files=checked,
                  verified_bytes=size, missing_files=missing)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open('x') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'missing_files'} |
                     {'missing_count': len(missing)}))


if __name__ == '__main__':
    main()
