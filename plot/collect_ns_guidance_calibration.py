"""Mirror committed calibration/evaluation predictions without touching source jobs."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import time

from run_ns_loss_study import write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    remote = 'zhangxifeng@192.168.191.197'
    root = '/research_data/users/zhangxifeng/C01Python/ns_loss_spectrum_20260907/guidance_calibration_v1'
    script = f'''import json,pathlib
p=pathlib.Path({root!r});files=[f.name for f in p.glob('*.json')];rows=[]
for stage in ['calibration','evaluation']:
 for f in sorted((p/stage).rglob('*.json')):
  r=json.loads(f.read_text());rows.append(r);files.append(str(f.relative_to(p)))
  if r.get('prediction_sha256'):files.append(str(f.with_suffix('.pt').relative_to(p)))
print(json.dumps(dict(files=files,receipts=rows)))
'''
    args.study.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            report = json.loads(subprocess.check_output(
                ['ssh', '-o', 'ConnectTimeout=20', remote, 'python3', '-'],
                input=script, text=True, timeout=60))
            listing = args.study / 'ns_calibration_collect_files.txt'
            listing.write_text('\n'.join(report['files']) + '\n')
            dest = args.study / 'guidance_calibration_v1'
            dest.mkdir(exist_ok=True)
            subprocess.run(['rsync', '-a', '--timeout=60', '--files-from=' + str(listing),
                            remote + ':' + root + '/', str(dest) + '/'], check=True)
            counts = {stage: Counter(r['status'] for r in report['receipts'] if r['stage'] == stage)
                      for stage in ['calibration', 'evaluation']}
            complete = (sum(counts['calibration'].values()) == 540
                        and sum(counts['evaluation'].values()) == 576
                        and all(f'{stage}_complete_{j}.json' in report['files']
                                for stage in counts for j in [0, 1]))
            summary = dict(checked_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
                           outcomes={k: dict(v) for k, v in counts.items()},
                           expected=dict(calibration=540, evaluation=576),
                           selection_frozen='selection.json' in report['files'],
                           sampling_complete=complete)
            write(args.study / 'calibration_progress.json', summary)
            print(json.dumps(summary), flush=True)
            if complete:
                subprocess.run([sys.executable, str(Path(__file__).with_name('audit_ns_guidance_calibration.py')),
                                '--inputs', str(args.study / 'inputs_v2'), '--results', str(dest),
                                '--baseline-audit', str(args.study / 'ns_partial_audit'),
                                '--output', str(args.study / 'calibration_complete_audit'), '--require-complete'], check=True)
                subprocess.run([sys.executable, str(Path(__file__).with_name('export_ns_guidance_calibration.py')),
                                '--audit', str(args.study / 'calibration_complete_audit'),
                                '--output', str(args.study / 'calibration_complete_report')], check=True)
                print('COMPLETE: calibration and paired evaluation audited and exported; visual QA and paper integration remain.', flush=True)
                return
        except (subprocess.SubprocessError, ValueError, OSError) as exc:
            print('Collection failed; source jobs are unchanged:', repr(exc), flush=True)
            if not args.watch:
                raise
        if not args.watch:
            return
        # Runs in an independent local tmux; the assistant tool is not blocked.
        time.sleep(600)


if __name__ == '__main__':
    main()
