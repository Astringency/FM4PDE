"""Read atomic checkpoints, recompute diagnostics, and save a truthful status."""
import argparse,datetime,io,json
from pathlib import Path
import torch
from experiments.strict_chain_diagnostics import diagnose
from scripts.train.resume_study import write


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    rows=[]
    for file in sorted(args.output.glob('*/state.pt')):
        if not (file.parent/'protocol.json').exists():
            continue
        spec=json.loads((file.parent/'protocol.json').read_text())
        s=torch.load(io.BytesIO(file.read_bytes()),map_location='cpu',weights_only=False)
        trace=s.get('trace',[])
        diag=diagnose(trace)
        row=dict(name=file.parent.name,input_id=spec['input_id'],iteration=s['iteration'],
            total_iterations=spec['warmup']+spec['keep'],retained=len(trace),
            computation_finished=(file.parent/'complete.json').exists(),
            stopped=(file.parent/'stopped.json').exists(),strict_diagnostics=diag,
            seconds=s['seconds'],code_commit=spec['code_commit'])
        rows.append(row)
    status=dict(as_of=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        stage='Full learned-FM target convergence development',runs=rows,
        convergence_established=False,independent_validation_passed=False,formal_1000_completed=0,
        limitation='Development diagnostics alone never authorize a scientific claim of improved 1000-input accuracy')
    write(args.output/'status.json',status)
    for row in rows:
        diag=row.pop('strict_diagnostics')
        row.update({k:diag[k] for k in ['max_rhat','min_bulk_ess','min_tail_ess','passed','reason'] if k in diag})
    print(json.dumps(rows,ensure_ascii=False))


if __name__=='__main__':main()
