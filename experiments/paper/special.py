"""Dispatch studies with repeated draws, two priors, or paired sampler timing."""
import json


def run(spec, args):
    if args.plan_only:
        print(json.dumps(spec, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    if spec['engine']=='averaging':
        from experiments.paper.averaging import run
    elif spec['engine']=='architecture':
        from experiments.paper.architecture import run
    elif spec['engine'] in {'timing','traces'}:
        from experiments.paper.paired_samplers import run
    else:
        raise ValueError(f"Unknown engine: {spec['engine']}")
    run(spec, args)
