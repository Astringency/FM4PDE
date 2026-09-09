"""Check gated NS table candidates against frozen sources and rendered TeX/PDF."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper', required=True, type=Path)
    parser.add_argument('--study', required=True, type=Path)
    parser.add_argument('--export-root', required=True, type=Path,
                        help='Audit directory containing prepared_sources.json and exported/')
    args = parser.parse_args()
    assert (args.paper/'audit').resolve() in args.export_root.resolve().parents
    out = args.export_root/'exported'
    frozen = json.loads((args.export_root/'prepared_sources.json').read_text())
    manifest = json.loads((out/'export_manifest.json').read_text())
    assert manifest['status'] == 'pass' and manifest['tables'] == 8
    for name, expected in manifest['outputs'].items():
        assert sha(out/name) == expected, name
    for name, expected in manifest['input_gate_hashes'].items():
        assert sha(Path(name)) == expected, name
    main_rows = csv_rows(out/'main_figure_values.csv')
    provenance = csv_rows(out/'table_value_provenance.csv')
    diffusion = csv_rows(out/'diffusion_comparison_summary.csv')
    target_labels = {x['table'] for x in main_rows}
    assert len(main_rows) == 264 and len(provenance) == 2710 and len(diffusion) == 51
    counts = {}
    for name, old_rows, new_rows in [('main', frozen['main'], main_rows),
            ('provenance', frozen['provenance'], provenance), ('diffusion', frozen['diffusion'], diffusion)]:
        assert len(old_rows) == len(new_rows)
        changed = 0
        for old, new in zip(old_rows, new_rows):
            if old == new:
                continue
            changed += 1
            assert new['method'] == 'FM4PDE'
            if name == 'diffusion':
                assert new['pde'] == 'nsnonbounded'
                assert old['task'] == new['task'] and old['field'] == new['field']
            else:
                assert new['PDE'] in {'NS', 'Navier--Stokes', 'nsnonbounded'}
                assert new['table'] in target_labels and new['n'] == '1000'
                assert all(old[k] == new[k] for k in ['table', 'PDE', 'distribution', 'metric', 'method'])
        counts[name] = changed
    assert counts == dict(main=18, provenance=18, diffusion=4)
    name_map = {'Poisson':'poisson', 'Helmholtz':'helmholtz', 'Darcy':'darcy',
                'NS':'nsnonbounded', 'Navier--Stokes':'nsnonbounded'}
    name_map.update({name:name for name in list(name_map.values())})
    settings = {'full-forward':'full_forward', 'full-inverse':'full_inverse',
                'sparse-forward':'sparse_forward', 'sparse-inverse':'sparse_inverse', 'sparse-joint':'sparse_joint'}
    summary = {(x['dist'], x['setting']):x for x in csv_rows(args.study/'tables/ns_main_summary.csv')}
    for row in main_rows:
        if name_map[row['PDE']] != 'nsnonbounded' or row['method'] != 'FM4PDE':
            continue
        setting = settings[row['table'].removeprefix('tab:').removesuffix('-results')]
        field = row['metric'][-2]
        expected = summary[row['distribution'].lower(), setting]
        for value in ['mean', 'sd']:
            assert float(row[value]) == float(expected[f'error_{field}_{value}'])
    tex_cells = 0
    for label in sorted(target_labels):
        selected = [x for x in main_rows if x['table'] == label]
        methods = list(dict.fromkeys(x['method'] for x in selected))
        fields = ['a','u'] if 'joint' in label else [selected[0]['metric'][-2]]
        # Method ordering is defined by the rendered header, not by the CSV order.
        methods = (['IFNO','FM4PDE'] if 'full-inverse' in label else
                   ['FNO','DeepONet','IFNO','FM4PDE'] if 'full-forward' in label else
                   ['RecFNO','Senseiver','VoronoiCNN','FM4PDE'])
        index = {(name_map[x['PDE']],x['distribution'],x['method'],x['metric'][-2]):x for x in selected}
        text = (out/(label.removeprefix('tab:')+'.tex')).read_text()
        body = [line for line in text.splitlines() if any(line.startswith(n+' & ') for n in name_map)]
        assert len(body) == 12
        for line in body:
            cells = line.split(' & ')
            pde, dist = name_map[cells[0]], cells[1]
            assert len(cells) == 2+len(methods)*len(fields)
            for j, method in enumerate(methods):
                for k, field in enumerate(fields):
                    row = index[pde,dist,method,field]
                    cell = cells[2+j*len(fields)+k]
                    values = re.findall(r'\d+\.\d+', cell)
                    assert values == [f"{100*float(row['mean']):.2f}",f"{100*float(row['sd']):.2f}"]
                    unique = sorted({float(index[pde,dist,m,field]['mean']) for m in methods})
                    rank = unique.index(float(row['mean']))+1
                    assert (r'\mathbf{' in cell) == (rank == 1)
                    assert (r'^{\dagger}' in cell) == (rank == 2)
                    tex_cells += 1
    assert tex_cells == 264
    diffusion_cells = 0
    di = {(x['task'],x['pde'],x['field'],x['method'],int(x['steps'])):x for x in diffusion}
    for task, fields in [('forward',['u']),('inverse',['a']),('both',['a','u'])]:
        text = (out/f'diffusion-{task}.tex').read_text()
        body = [line for line in text.splitlines() if any(line.startswith(n+' & ') for n in name_map)]
        assert len(body) == 4*len(fields)
        for line in body:
            cells = line.split(' & ')
            assert len(cells) == 5
            pde = name_map[cells[0]]
            field = re.search(r'\\mathbf\{([au])\}',cells[1]).group(1)
            data = [di[task,pde,field,m,k] for m,k in [('FM4PDE',100),('DiffusionPDE',100),('DiffusionPDE',1000)]]
            unique = sorted({float(x['mean']) for x in data})
            for row, cell in zip(data,cells[2:]):
                assert re.findall(r'\d+\.\d+',cell) == [f"{100*float(row['mean']):.2f}",f"{100*float(row['sd']):.2f}"]
                rank = unique.index(float(row['mean']))+1
                assert (r'\mathbf{' in cell) == (rank == 1)
                assert (r'^{\dagger}' in cell) == (rank == 2)
                if pde == 'nsnonbounded' and row['method'] == 'FM4PDE':
                    setting = {'forward':'sparse_forward','inverse':'sparse_inverse','both':'sparse_joint'}[task]
                    for value in ['mean','sd']:
                        assert float(row[value]) == float(summary['smooth',setting][f'error_{field}_{value}'])
                diffusion_cells += 1
    assert diffusion_cells == 48
    xml_path = args.export_root/'table_export_font.xml'
    with xml_path.open('w') as stream:
        subprocess.run(['pdftohtml','-xml','-hidden','-i','-zoom','3','-stdout',
                        str(out/'table_fragments_preview.pdf')], stdout=stream, check=True)
    xml = ET.parse(xml_path).getroot()
    fonts = {x.attrib['id']:x.attrib for x in xml.iter('fontspec')}
    pages = []
    for page in xml.findall('page'):
        ordinary = []
        for text in page.findall('text'):
            font = fonts[text.attrib['font']]
            family = font['family']
            # Mathematical subscripts and dagger markers use smaller fonts.
            if re.search(r'(CMR9|CMBX9|CMR10|CMBX10|CMCSC10)$',family):
                ordinary.append(float(font['size'])/3)
        assert ordinary and min(ordinary) >= 9.0, (page.attrib['number'], ordinary)
        pages.append(dict(page=int(page.attrib['number']), minimum_ordinary_font_bp=min(ordinary)))
    assert len(pages) == 8
    stats = json.loads((out/'statistics_to_integrate.json').read_text())
    exceptions = []
    for row in stats['main_after']:
        for pde in {'poisson','helmholtz','darcy','nsnonbounded'}-set(row['rough_is_largest_pdes']):
            values = {x['distribution']:100*float(x['mean']) for x in main_rows
                      if x['table']==row['table'] and name_map[x['PDE']]==pde and
                      x['method']==row['method'] and x['metric'][-2]==row['field']}
            assert values['Rough'] < max(values.values())
            exceptions.append(dict(setting=row['setting'],field=row['field'],method=row['method'],
                                   pde=pde,mean_percent=values))
    report = dict(status='pass',source_row_counts=dict(main=264,provenance=2710,diffusion=51),
        changed_rows=counts,unchanged_rows_equal_frozen_sources=True,main_tex_cells_checked=tex_cells,
        diffusion_tex_cells_checked=diffusion_cells,all_diffusion_displayed_values_and_rank_marks_match_sources=True,
        all_main_displayed_values_and_rank_marks_match_unrounded_sources=True,
        main_ns_values_equal_complete_summary=True,export_manifest_hashes_pass=True,
        preview_pages=pages,font_estimate='Poppler font sizes at zoom 3; rounded to one third of a PDF point.',rough_not_largest=exceptions,
        fm_macro_and_wins=[x for x in stats['main_after'] if x['method']=='FM4PDE'],
        source_sha256=sha(Path(__file__)),export_manifest_sha256=sha(out/'export_manifest.json'))
    (args.export_root/'final_export_qa.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PASS: 264 main + 48 diffusion cells, source scope, eight PDF pages, and unrounded ranking; Rough exceptions:',len(exceptions))


if __name__ == '__main__':
    main()
