#!/usr/bin/env python3
"""Restore the frozen paper into a new directory and compile stable references."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

from paper_assets import DEFAULT_ARCHIVE, STEMS, restore, sha, verify


def compile_paper(archive, output):
    if output.exists():
        raise FileExistsError(f'Use a new output directory: {output}')
    if archive == output or archive in output.parents:
        raise ValueError('Build output must be outside the frozen paper archive')
    before = verify(archive)
    output.mkdir(parents=True)
    paper = output / 'paper'
    restore(archive, paper)
    records = []
    for stem in STEMS:
        commands = []
        def run(command):
            start = time.monotonic()
            with (output / (stem + '.build.log')).open('ab') as log:
                process = subprocess.run(command, cwd=paper, stdout=log,
                                         stderr=subprocess.STDOUT)
            commands.append(dict(command=command, exit_code=process.returncode,
                                 elapsed_seconds=time.monotonic() - start))
            process.check_returncode()

        latex = ['pdflatex', '-interaction=nonstopmode', '-halt-on-error', stem + '.tex']
        run(latex)
        if stem != 'response_to_reviewers_scoped_0906':
            # This mode does not depend on the optional system 88591lat.csf file.
            run(['bibtex8', '-8', stem])

        def references():
            return {suffix: sha(paper / (stem + suffix))
                    for suffix in ('.aux', '.out', '.toc', '.lof', '.lot')
                    if (paper / (stem + suffix)).exists()}

        stable = False
        for _ in range(6):
            prior = references()
            run(latex)
            final_log = (paper / (stem + '.log')).read_text(errors='replace')
            unresolved = re.search(
                r'(?:Citation|Reference).*undefined|There were undefined|'
                r'Label\(s\) may have changed|Rerun to get cross-references|'
                r'rerunfilecheck Warning: File .* has changed', final_log)
            if references() == prior and not unresolved:
                stable = True
                break
        if not stable:
            raise RuntimeError(f'References did not stabilize: {stem}')
        if re.search(r'Overfull \\[hv]box|Too many unprocessed floats|'
                     r'LaTeX Error:|Package .* Error:', final_log):
            raise RuntimeError(f'Unresolved layout/build error: {stem}')
        original_text = subprocess.check_output(['pdftotext', '-layout', str(archive / (stem + '.pdf')), '-'])
        rebuilt_text = subprocess.check_output(['pdftotext', '-layout', str(paper / (stem + '.pdf')), '-'])
        if rebuilt_text != original_text:
            raise RuntimeError(f'Rebuilt PDF text differs from frozen PDF: {stem}')
        info = subprocess.check_output(['pdfinfo', str(paper / (stem + '.pdf'))], text=True)
        record = dict(document=stem, status='pass', references_stable=True,
                      pdf_text_bytes_equal=True,
                      pdf_text_sha256=hashlib.sha256(rebuilt_text).hexdigest(),
                      pages=int(re.search(r'^Pages:\s+(\d+)', info, re.M).group(1)),
                      pdf_sha256=sha(paper / (stem + '.pdf')),
                      commands=commands,
                      remaining_diagnostics=[line for line in final_log.splitlines()
                                             if 'Warning' in line or 'Underfull' in line])
        records.append(record)
        print(json.dumps({k: record[k] for k in ('document', 'status', 'pages')}), flush=True)
    if verify(archive) != before:
        raise RuntimeError('Frozen paper archive changed during compilation')
    result = dict(status='pass', complete=True,
                  completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  archive_verification=before, immutable_archive_unchanged=True,
                  script_sha256=sha(Path(__file__)), documents=records)
    with (output / 'verification_complete.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compile_paper(args.archive.resolve(), args.output.resolve())
    print(json.dumps(dict(status=result['status'], documents=len(result['documents']),
                         output=str(args.output.resolve()))))
