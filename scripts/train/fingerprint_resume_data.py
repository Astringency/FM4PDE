"""Fingerprint the exact training shards before dispatching continuation to a host."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import socket
import time


def main(root, output, burger_directory):
    paths = [(pde, shard, root / (burger_directory if pde == 'burger' else pde) /
              f'{pde}_10000-128-128_{shard}.mat')
             for pde in ['poisson', 'darcy', 'burger'] for shard in range(1, 6)]
    assert all(path.is_file() for _, _, path in paths)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    def fingerprint(item):
        pde, shard, path = item
        before = path.stat()
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 << 20), b''):
                digest.update(block)
        after = path.stat()
        assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        return dict(pde=pde, shard=shard, path=str(path.resolve()), bytes=after.st_size,
                    sha256=digest.hexdigest())

    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(fingerprint, item) for item in paths]):
            row = future.result()
            rows.append(row)
            print('HASHED', row['pde'], row['shard'], row['sha256'], flush=True)
    value = dict(status='complete', host=socket.gethostname(), checked_unix=time.time(),
                 rows=sorted(rows, key=lambda row: (row['pde'], row['shard'])))
    temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--burger-directory', default='burgers')
    args = parser.parse_args()
    main(args.data_root, args.output, args.burger_directory)
