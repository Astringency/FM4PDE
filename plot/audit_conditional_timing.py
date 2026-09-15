#!/usr/bin/env python3
"""Report all measured times and a transparently restricted occupancy subset.

This companion does not change predictions, prefix estimates, or recorded times.
The exposure interval starts conservatively at external-process creation; memory
occupancy was observed later, so it is not a measurement of active GPU contention.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import statistics

KS = (1, 3, 10, 100, 1000)
TASKS = ('forward', 'inverse', 'both')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantile(values, probability):
    values = sorted(values)
    position = (len(values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (position - lower) * (values[upper] - values[lower])


def describe(values):
    if not values:
        return dict(n=0, mean_seconds=None, sd_seconds=None, median_seconds=None,
                    q25_seconds=None, q75_seconds=None, p95_seconds=None)
    return dict(n=len(values), mean_seconds=statistics.mean(values),
                sd_seconds=statistics.stdev(values) if len(values) > 1 else 0.,
                median_seconds=statistics.median(values),
                q25_seconds=quantile(values, .25), q75_seconds=quantile(values, .75),
                p95_seconds=quantile(values, .95))


def exposure_index(initial, handoff):
    starts = {x['gpu']: x['process_started_unix'] for x in initial['external']}
    assert set(starts) == {6, 7}
    result = {}
    for row in handoff['rows']:
        key = row['task'], row['offset'], row['start'], row['stop']
        assert key not in result, ('Duplicate batch in handoff snapshot', key)
        seconds = row['seconds']
        end = row['receipt_mtime_unix']
        assert seconds > 0 and row['gpu'] in starts
        # The receipt is written just after the timed interval. This estimate
        # includes a small serialization offset and is explicitly approximate.
        start = end - seconds
        overlap = max(0., min(seconds, end - starts[row['gpu']]))
        result[key] = dict(**row, estimated_active_start_unix=start,
                           potential_overlap_seconds=overlap,
                           potentially_exposed=overlap > 0)
    return result


def build(results, initial_path, handoff_path, allow_partial=False):
    initial = json.loads(initial_path.read_text())
    handoff = json.loads(handoff_path.read_text())
    exposed = exposure_index(initial, handoff)
    exposed_pools = {key[:2] for key, row in exposed.items() if row['potentially_exposed']}
    all_rows = []
    pool_rows = []
    seen = set()
    source_files = {}
    for marker in sorted(results.glob('cases/*/offset*/complete.json')):
        done = json.loads(marker.read_text())
        task, offset = done['binding']['task'], done['binding']['offset']
        key = task, offset
        assert key not in seen and task in TASKS and offset in range(1500, 1532)
        assert done['status'] == 'complete' and done['predictions'] == 1000
        seen.add(key)
        batches = []
        covered = []
        for path in sorted(marker.parent.glob('batches/*/receipt.json')):
            receipt = json.loads(path.read_text())
            ids = receipt['seed_indices']
            assert ids == list(range(ids[0], ids[-1] + 1))
            covered.extend(ids)
            batch_key = task, offset, ids[0], ids[-1] + 1
            timing = exposed.get(batch_key)
            seconds = receipt['active_seconds']
            assert seconds > 0
            if timing:
                assert abs(seconds - timing['seconds']) < 1e-8, batch_key
                assert abs(path.stat().st_mtime - timing['receipt_mtime_unix']) < 1e-3, batch_key
            batches.append(dict(stop=ids[-1]+1, seconds=seconds,
                flagged_seconds=seconds if timing and timing['potentially_exposed'] else 0.,
                estimated_overlap_seconds=timing['potential_overlap_seconds'] if timing else 0.))
            source_files[str(path.relative_to(results))] = sha(path)
        assert covered == list(range(1000)), key
        prefix_overhead = 0.
        pool_exposed = key in exposed_pools
        for k in KS:
            path = marker.parent / f'prefix_{k}.json'
            prefix = json.loads(path.read_text())
            assert prefix['K'] == k and prefix['binding'] == done['binding']
            prefix_overhead += prefix['mean_seconds']
            before = [x for x in batches if x['stop'] <= k]
            generation = sum(x['seconds'] for x in before)
            assert abs(prefix['seconds'] - generation - prefix_overhead) < 1e-6
            all_rows.append(dict(task=task, offset=offset, K=k,
                seconds=prefix['seconds'], active_generation_seconds=generation,
                prefix_mean_seconds=prefix_overhead,
                pool_potentially_exposed=pool_exposed,
                prefix_potentially_exposed=any(x['flagged_seconds'] for x in before),
                potentially_exposed_batch_seconds=sum(x['flagged_seconds'] for x in before),
                estimated_overlap_active_seconds=sum(x['estimated_overlap_seconds'] for x in before)))
            source_files[str(path.relative_to(results))] = sha(path)
        pool_rows.append(all_rows[-1])
    expected = {(task, offset) for task in TASKS for offset in range(1500, 1532)}
    if not allow_partial:
        assert seen == expected, ('Incomplete timing cohort', len(seen))
        assert exposed_pools <= seen
    summary = []
    for task in TASKS:
        for k in KS:
            rows = [r for r in all_rows if r['task'] == task and r['K'] == k]
            for cohort in ('all_inputs', 'no_recorded_external_occupancy'):
                selected = rows if cohort == 'all_inputs' else [r for r in rows if not r['pool_potentially_exposed']]
                summary.append(dict(task=task, K=k, cohort=cohort,
                                    **describe([r['seconds'] for r in selected])))
    task_impact = []
    for task in TASKS:
        rows = [r for r in pool_rows if r['task'] == task]
        denominator = sum(r['active_generation_seconds'] for r in rows)
        flagged = sum(r['potentially_exposed_batch_seconds'] for r in rows)
        estimated = sum(r['estimated_overlap_active_seconds'] for r in rows)
        task_impact.append(dict(task=task, completed_pools=len(rows),
            potentially_exposed_offsets=sorted(offset for t, offset in exposed_pools if t == task),
            potentially_exposed_pool_count=sum(r['pool_potentially_exposed'] for r in rows),
            all_generation_active_seconds=denominator,
            potentially_exposed_batch_seconds=flagged,
            potentially_exposed_batch_active_fraction=flagged / denominator if denominator else None,
            estimated_overlap_active_seconds=estimated,
            estimated_overlap_active_fraction=estimated / denominator if denominator else None))
    report = dict(status='complete' if seen == expected else 'partial_snapshot',
        complete=seen == expected, physical_inputs_per_task=32,
        completed_pools=len(seen), timing_rows=len(all_rows), task_impact=task_impact,
        exposure_scope='Conservative interval beginning at external-process creation; no continuous utilization trace was recorded.',
        exposure_time_method='Batch end is receipt mtime; approximate start is end minus recorded active duration. The full duration of any overlapping batch is also reported as a conservative bound.',
        subset_definition='Exclude any pool with a potentially overlapping batch at any K; use the same remaining input set for all five K.',
        accuracy_scope='All 32 physical inputs per task remain in every accuracy statistic.',
        time_scope='Cumulative active generation, field conversion, transfer and prefix means; model loading, file I/O and resumed downtime excluded.',
        initial_occupancy_source=dict(path=str(initial_path), sha256=sha(initial_path)),
        handoff_source=dict(path=str(handoff_path), sha256=sha(handoff_path)),
        source_files=source_files)
    return report, all_rows, summary


def save_report(output, report, rows, summary):
    output.mkdir(parents=True, exist_ok=True)
    for name, data in [('conditional_timing_per_input.csv', rows),
                       ('conditional_timing_by_occupancy.csv', summary)]:
        path = output / name
        stream = io.StringIO(newline='')
        writer = csv.DictWriter(stream, fieldnames=list(data[0]))
        writer.writeheader(); writer.writerows(data)
        content = stream.getvalue().encode()
        if path.exists():
            assert path.read_bytes() == content, ('Keep conflicting timing exports', path)
        else:
            path.write_bytes(content)
    report['outputs'] = {name: sha(output / name) for name in
                        ['conditional_timing_per_input.csv', 'conditional_timing_by_occupancy.csv']}
    path = output / 'conditional_timing_audit.json'
    content = json.dumps(report, indent=2) + '\n'
    if path.exists():
        assert path.read_text() == content, ('Keep conflicting timing report', path)
    else:
        path.write_text(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--initial-occupancy', type=Path, required=True)
    parser.add_argument('--handoff-snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    report, rows, summary = build(args.results, args.initial_occupancy,
                                 args.handoff_snapshot, args.allow_partial)
    save_report(args.output, report, rows, summary)
    print(json.dumps({k: report[k] for k in ['status', 'completed_pools', 'task_impact']}, indent=2))


if __name__ == '__main__':
    main()
