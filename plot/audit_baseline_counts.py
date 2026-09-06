"""Reconcile baseline workbook rows with archived per-example metric arrays.

Uses the existing audited row-to-run mapping, checks every workbook identity,
and rereads current raw files. This is a count/aggregation audit, not a new
prediction or training-validity audit. --from-frozen verifies the portable
metric extract without mounted raw files.
"""
from pathlib import Path
import argparse
import csv
import gzip
import hashlib
import json
import math
import numpy as np
import openpyxl


DEFAULT_PAPER = Path('/home/tat512/C04Papers/fm4pde_jmlr')
DEFAULT_AUDIT = Path('/home/tat512/C01Python/audit/baseline_vs_fm4pde_20260905_215137')
IDENTITY = ['PDE', 'Method', 'TASK', 'DIST', 'CONDITION', 'SENSOR', 'SEED', 'Ablation']


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def decoded(value):
    return json.loads(value) if isinstance(value, str) else value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper', type=Path, default=DEFAULT_PAPER)
    parser.add_argument('--prior-audit', type=Path, default=DEFAULT_AUDIT)
    parser.add_argument('--from-frozen', action='store_true')
    args = parser.parse_args()
    workbook_path = args.paper/'source_data/results.xlsx'
    workbook = openpyxl.load_workbook(workbook_path, data_only=True, read_only=True)
    sheet = workbook['Results']; headers = next(sheet.values)
    current = {i: dict(zip(headers, row)) for i, row in enumerate(list(sheet.values)[1:], 2)}
    workbook.close()
    archive_path = args.paper/'source_data/baseline_counts_archive.json.gz'
    if args.from_frozen:
        with gzip.open(archive_path, 'rt') as stream:
            archive = json.load(stream)
        assert archive['workbook_sha256'] == sha256(workbook_path)
    else:
        old_path = args.prior_audit/'evidence/workbook_results_baseline.json'
        old = json.loads(old_path.read_text())['rows']
        assert len(old) == len(current) == 267
        for row in old:
            assert all(current[row['row']][key] == row[key] for key in IDENTITY), row['row']
        map_path = args.prior_audit/'metrics_recomputed.csv'
        records = list(csv.DictReader(map_path.open()))
        sources = {}
        for record in records:
            if record['family'] != 'Baseline' or not record.get('summary_path'):
                continue
            row = int(record['row']); path = record['summary_path']
            assert row not in sources or sources[row] == path, row
            sources[row] = path
        archive = dict(workbook_sha256=sha256(workbook_path),
                       row_mapping_source=str(map_path), row_mapping_sha256=sha256(map_path),
                       identity_source_sha256=sha256(old_path), cells=[])
        for row, entry in current.items():
            cell = dict(row=row, identity={k: entry[k] for k in IDENTITY},
                        summary_path=sources.get(row), ids=[], values={'a': [], 'u': []})
            path = Path(sources[row]).with_name('results_raw.jsonl') if row in sources else None
            if path is not None and path.exists():
                cell.update(raw_path=str(path), raw_sha256=sha256(path), raw_batches=0,
                            declared_solution_scopes=[], statuses=[])
                for line in path.open():
                    if not line.strip():
                        continue
                    record = json.loads(line); cell['raw_batches'] += 1
                    ids = decoded(record.get('global_sample_ids', [])); cell['ids'].extend(ids)
                    cell['declared_solution_scopes'].append(record.get('relative_l2_solution_scope', 'not_recorded'))
                    cell['statuses'].append(record.get('status', 'not_recorded'))
                    for field, standard in [('a', 'relative_l2_input_or_coeff'), ('u', 'relative_l2_solution')]:
                        key = ('rel_l2_'+field) if entry['Ablation'] and record.get('rel_l2_'+field+'_values') else standard
                        values = decoded(record.get(key+'_values', []))
                        # Structural absent-field values are stored explicitly as null.
                        cell['values'][field].extend(float(v) if v is not None and math.isfinite(float(v)) else None
                                                     for v in values)
                cell['declared_solution_scopes'] = sorted(set(cell['declared_solution_scopes']))
                cell['statuses'] = sorted(set(cell['statuses']))
            archive['cells'].append(cell)
        with gzip.open(archive_path, 'wt') as stream:
            json.dump(archive, stream, allow_nan=False)

    result = []; missing = []; mismatches = []; complete_cells = 0; counts = set()
    for cell in archive['cells']:
        row = cell['row']; entry = current[row]
        assert cell['identity'] == {k: entry[k] for k in IDENTITY}
        numeric = [f for f in ['a', 'u'] if entry['rel L2('+f+')'] is not None]
        if not numeric:
            missing.append(row); continue
        assert cell['ids'], ('Missing raw source for a populated workbook row', row)
        assert len(set(cell['ids'])) == len(cell['ids']), ('Duplicate raw IDs', row)
        complete_cells += 1; counts.add(len(cell['ids']))
        for field in numeric:
            values = cell['values'][field]
            assert len(values) == len(cell['ids']), ('Metric/ID length mismatch', row, field)
            assert all(v is not None for v in values), ('Nonfinite applicable field', row, field)
            array = np.array(values, dtype=np.float64)
            mean, sd = float(array.mean()), float(array.std(ddof=1))
            expected_mean = float(entry[f'rel L2({field})'])
            expected_sd = float(entry[f'rel L2({field}) std'])
            matched = math.isclose(mean, expected_mean, rel_tol=1e-10, abs_tol=1e-12)
            matched_sd = math.isclose(sd, expected_sd, rel_tol=1e-10, abs_tol=1e-12)
            item = dict(row=row, **cell['identity'], field=field, n=len(values), unique_ids=len(set(cell['ids'])),
                        raw_mean=mean, raw_sd_ddof1=sd, workbook_mean=expected_mean, workbook_sd=expected_sd,
                        mean_difference=mean-expected_mean, sd_difference=sd-expected_sd,
                        mean_matches=matched, sd_ddof1_matches=matched_sd,
                        raw_path=cell['raw_path'], raw_sha256=cell['raw_sha256'],
                        solution_scope=';'.join(cell['declared_solution_scopes']))
            result.append(item)
            if not (matched and matched_sd):
                mismatches.append(item)
    out = args.paper/'source_data/baseline_effective_counts_verified.csv'
    with out.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result[0])); writer.writeheader(); writer.writerows(result)
    report = dict(workbook_sha256=sha256(workbook_path), archive_sha256=sha256(archive_path),
                  mapped_workbook_rows=len(current), populated_rows=complete_cells, blank_rows=missing,
                  raw_example_counts=sorted(counts), field_metrics_checked=len(result),
                  means_and_sample_sds_matched=len(result)-len(mismatches), mismatches=mismatches,
                  script_sha256=sha256(Path(__file__)),
                  scope='Verifies row identities, raw ID counts, finite applicable per-example metric arrays, and mean/sample-SD aggregation. Does not recompute predictions, certify historical training validity, or align cross-method masks.')
    (args.paper/'source_data/baseline_count_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ['mismatches', 'blank_rows']}, indent=2))
    for mismatch in mismatches:
        print('MISMATCH', mismatch['row'], mismatch['Method'], mismatch['field'], mismatch['mean_difference'], mismatch['sd_difference'])


if __name__ == '__main__':
    main()
