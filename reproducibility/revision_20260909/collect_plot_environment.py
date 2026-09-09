"""Record the installed chart-rendering environment without importing PyTorch."""
from __future__ import annotations
import argparse
import datetime
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import socket
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New JSON observation file')
    parser.add_argument('--font-dir', type=Path, required=True)
    args = parser.parse_args()
    versions = args.output.with_suffix('.requirements.txt')
    assert not args.output.exists() and not versions.exists(), 'Preserve earlier observations'
    import matplotlib
    import matplotlib.ft2font
    import numpy
    import PIL
    assert 'torch' not in sys.modules
    packages = sorted([{'name': d.metadata.get('Name', ''), 'version': d.version}
                       for d in metadata.distributions()], key=lambda d: (d['name'].lower(), d['version']))
    fonts = {}
    for name in ('times.ttf', 'timesbd.ttf', 'timesi.ttf', 'timesbi.ttf'):
        path = (args.font_dir/name).resolve(strict=True)
        fonts[name] = dict(path=str(path), bytes=path.stat().st_size,
                          sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    record = dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        hostname=socket.gethostname(), python_executable=sys.executable, python=sys.version,
        platform=platform.platform(), matplotlib=matplotlib.__version__,
        freetype=matplotlib.ft2font.__freetype_version__, pillow=PIL.__version__, numpy=numpy.__version__,
        packages=packages, fonts=fonts, cuda_initialized=False, torch_imported=False,
        scope='Installed environment used to create the final paper figures; metadata observation only.',
        font_restore='export FM4PDE_FONT_DIR="$FM_REPO/reproducibility/revision_20260909/dependencies/fonts/times_new_roman"',
        environment_restore='Use an isolated Python environment with the recorded interpreter and rendering package versions; verify the bundled Matplotlib FreeType version and font hashes before exact pixel comparison.',
        collector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(json.dumps(record, indent=2, sort_keys=True)+'\n')
    with versions.open('x') as stream:
        stream.write('# Installed package versions; see adjacent JSON for interpreter, rendering libraries and fonts.\n')
        stream.writelines(item['name']+'=='+item['version']+'\n' for item in packages)
    print(json.dumps({k: record[k] for k in ['python_executable', 'matplotlib', 'freetype', 'pillow', 'numpy']}))


if __name__ == '__main__':
    main()
