"""Compare the data and label mappings of two completed 80-figure exports."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    checks = {}
    for folder, name, count in [('field_figures', 'figure_manifest.json', 38),
                               ('sweep_figures', 'sweep_figure_manifest.json', 30),
                               ('ensemble_figures', 'figure_manifest.json', 12)]:
        old = json.loads((args.reference/folder/name).read_text())
        new = json.loads((args.replay/folder/name).read_text())
        assert old['plotter_sha256'] == new['plotter_sha256']
        captions = {p.name: sha(p) for p in (args.reference/folder).glob('*.tex')}
        assert captions == {p.name: sha(p) for p in (args.replay/folder).glob('*.tex')}
        check = dict(figures=count, plotter_sha256=old['plotter_sha256'], caption_tex_sha256=captions)
        if folder == 'field_figures':
            for key in ['pdes', 'sample_id', 'chart_contract_sha256', 'font']:
                assert old[key] == new[key], key
            assert old['outputs'].keys() == new['outputs'].keys()
            assert Counter(old['sources'].values()) == Counter(new['sources'].values())
            check.update(pdes_sample_contract_font_mapping_equal=True,
                         raw_source_hash_multiset_equal=True, sources=len(old['sources']))
            original = list(csv.DictReader((args.reference/folder/'reconstruction_metrics.csv').open()))
            relocated = list(csv.DictReader((args.replay/folder/'reconstruction_metrics.csv').open()))
            assert len(original) == len(relocated) == 168
            differences = []
            for x, y in zip(original, relocated):
                assert x.keys() == y.keys()
                for key in x:
                    if x[key] != y[key]:
                        a, b = float(x[key]), float(y[key])
                        delta = abs(a-b)
                        assert delta <= 1e-12*max(abs(a), abs(b))+1e-12, (key, a, b)
                        differences.append((delta, delta/max(abs(a), 1e-300)))
            check['annotations'] = dict(rows=168, changed_numeric_values=len(differences),
                max_absolute_difference=max((v[0] for v in differences), default=0),
                max_relative_difference=max((v[1] for v in differences), default=0))
        elif folder == 'sweep_figures':
            assert len(old['figures']) == len(new['figures']) == 30
            for x, y in zip(old['figures'], new['figures']):
                assert {k:v for k,v in x.items() if k != 'outputs'} == {k:v for k,v in y.items() if k != 'outputs'}
            check['pde_group_condition_source_id_mapping_exact'] = True
        else:
            assert old['pdes'] == new['pdes'] and old['contract_sha256'] == new['contract_sha256']
            assert old['outputs'].keys() == new['outputs'].keys()
            gaps = [x['minimum_label_gap_pt'] for x in new['tick_label_checks']
                    if x['minimum_label_gap_pt'] is not None]
            assert min(gaps) > 0
            check.update(pde_contract_output_mapping_exact=True, minimum_spectrum_tick_label_gap_pt=min(gaps))
        checks[folder] = check
    result = dict(status='pass', total_figures=80, checks=checks)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
