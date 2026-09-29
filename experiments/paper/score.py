"""Compute manuscript Q from complete per-sample joint-comparison CSV files."""
import argparse
from collections import defaultdict
import csv
import math
from pathlib import Path

from experiments.paper.run import digest, write_json

METHODS = ('fm4pde', 'recfno', 'senseiver', 'voronoicnn')
METRICS = ('relative_l2_joint', 'pde_loss', 'solver_relative_l2')


def score_means(means):
    """Normalize three cohort means by their method-wise minima, then cube-root."""
    if set(means) != set(METHODS):
        raise ValueError(f'Q requires exactly these four methods: {METHODS}')
    if any(not math.isfinite(row[key]) or row[key] <= 0
           for row in means.values() for key in METRICS):
        raise ValueError('Q requires finite, strictly positive mean metrics')
    minima = {key: min(row[key] for row in means.values()) for key in METRICS}
    return {method: math.exp(sum(math.log(row[key]/minima[key]) for key in METRICS)/3)
            for method, row in means.items()}


def collect(inputs, expected_count=100):
    groups = defaultdict(dict)
    sources = []
    for method, path in inputs:
        if method not in METHODS:
            raise ValueError(f'Unknown joint-comparison method: {method}')
        sources.append(dict(method=method, path=str(path.resolve()), sha256=digest(path)))
        with path.open() as stream:
            for row in csv.DictReader(stream):
                if row.get('task', 'both') != 'both' or row['pde'] == 'burger':
                    raise ValueError('Q table uses paired-field joint reconstruction only')
                # Older residual_mse is deliberately not accepted as pde_loss.
                loss = float(row['pde_loss'])
                a = float(row.get('relative_l2_a', row.get('rel_l2_a')))
                u = float(row.get('relative_l2_u', row.get('rel_l2_u')))
                solver = float(row.get('solver_relative_l2', row.get('solver_defect_relative')))
                values = dict(relative_l2_joint=(a+u)/2, pde_loss=loss, solver_relative_l2=solver)
                if any(not math.isfinite(x) or x < 0 for x in values.values()):
                    raise ValueError(f'Invalid per-sample metrics in {path}')
                key = row['pde'], row['distribution'], method
                index = int(row['index'])
                if index in groups[key]:
                    raise ValueError(f'Duplicate input {index} in {key}')
                groups[key][index] = values
    rows = []
    for pde, distribution in sorted({key[:2] for key in groups}):
        means = {}
        for method in METHODS:
            samples = groups[pde, distribution, method]
            if sorted(samples) != list(range(expected_count)):
                raise ValueError(f'{pde}/{distribution}/{method}: expected input IDs 0–{expected_count-1}')
            means[method] = {key: math.fsum(row[key] for row in samples.values())/expected_count
                             for key in METRICS}
        scores = score_means(means)
        rows.append(dict(pde=pde, distribution=distribution, count=expected_count,
                         means=means, Q=scores))
    if not rows:
        raise ValueError('No joint-comparison samples supplied')
    return dict(rows=rows, inputs=sources, pde_loss_definition='componentwise_mse_sum',
                field_error_definition='arithmetic_mean_of_fieldwise_relative_l2',
                relative_error_units='ratio', aggregation='means_before_normalization')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', action='append', required=True, metavar='METHOD=CSV',
                        help='Repeat for all four methods; multiple shards per method are allowed')
    parser.add_argument('--expected-count', type=int, default=100)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.expected_count < 1:
        parser.error('--expected-count must be positive')
    inputs = [(method, Path(path)) for method, path in (value.split('=', 1) for value in args.input)]
    write_json(args.output, collect(inputs, args.expected_count))
