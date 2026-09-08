"""Small, separately labelled inverse step-budget diagnostic on four fixed inputs.

Reuse the exact checkpoint worker's prediction and persistence path. Select the
first four already-frozen evaluation IDs by order, not by observed outcomes.
All three models and both step budgets are retained; no hyperparameters tuned.
"""
from pathlib import Path
import argparse
import copy
import json
import subprocess

import run_ns_checkpoint_comparison as core
from run_ns_loss_study import sha, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['freeze', 'worker'])
    p.add_argument('--base-results', type=Path, required=True)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--weights-map', type=Path, required=True)
    p.add_argument('--shard', type=int, choices=[0, 1], default=0)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == 'freeze':
        assert not (args.output / 'protocol.json').exists()
        base = json.loads((args.base_results / 'protocol.json').read_text())
        protocol = copy.deepcopy(base)
        protocol.pop('upstream_protocol_sha256', None)
        protocol.pop('upstream_reuse', None)
        protocol.update(version='inverse_step_probe_v1', workers=2,
                        parent_protocol_sha256=sha(args.base_results / 'protocol.json'),
                        evaluation_ids=base['evaluation_ids'][:4], tasks=['inverse'],
                        variants=[dict(name=f'{label}_{config}{steps}', checkpoint=label,
                                       configuration=config, steps=steps)
                                  for label, config in [('current','common'),('v260904','common'),('bak','legacy')]
                                  for steps in [100,1000]], formal_calls=4*3*3*2,
                        selection='First four IDs in the pre-existing frozen evaluation order. No outcome-based choice. All three checkpoints, seeds and both budgets retained.',
                        scope='Small exploratory inverse-only step-budget diagnostic, four inputs and three seeds. Same GPU for both budgets and all checkpoints within an input. Not a 32- or 1000-input ranking, not hyperparameter selection, not an added training experiment.',
                        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=core.ROOT,text=True).strip())
        protocol['code_sha256']['plot/run_ns_checkpoint_step_probe.py'] = sha(Path(__file__))
        write(args.output / 'protocol.json', protocol)
        for shard, base_shard in [(0,2),(1,3)]:
            path = args.base_results / f'implementation_check_{base_shard}.json'
            checks = json.loads(path.read_text())
            assert checks['status'] == 'pass' and len(checks['checks']) == 4
            # This receipt explicitly distinguishes inherited tests from new tests.
            # The worker still checks the actual NFE of every 100/1000-step call.
            write(args.output / f'implementation_check_{shard}.json',
                  dict(status='inherited_100_step_implementation_checks', source=str(path),
                       source_sha256=sha(path), base_checks=checks,
                       scope='Same fm_predict implementation already tested for repetition, hidden-value independence, observation sensitivity and native-runner equivalence at 100 steps. This probe changes only the loop length; no separate 1000-step native-runner equivalence test is claimed.'))
        print('FROZEN',protocol['formal_calls'],protocol['evaluation_ids'],sha(args.output/'protocol.json'),flush=True)
    else:
        core.TASKS = ['inverse']
        args.pilot_only = False
        core.worker(args)


if __name__ == '__main__':
    main()
