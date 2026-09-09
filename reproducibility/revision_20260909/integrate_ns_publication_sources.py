"""Integrate audited NS source tables/timing; leave main manuscript and response untouched."""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--paper',required=True,type=Path)
    args=p.parse_args(); paper=args.paper.resolve()
    tables=paper/'audit/revision_0909/ns_table_updates/exported'
    timing=paper/'audit/revision_0909/ns_main_section_updates/timing_export'
    audit=paper/'audit/revision_0909/ns_source_integration'
    assert not audit.exists(), 'Use the preserved manifest to inspect an already completed integration'
    gate=json.loads((tables/'export_manifest.json').read_text())
    assert gate['status']=='pass'
    for name,value in gate['outputs'].items():assert sha(tables/name)==value,name
    for name,value in gate['input_gate_hashes'].items():assert sha(Path(name))==value,name
    qa=json.loads((timing/'FINAL_TIMING_AUDIT.json').read_text())
    assert qa['status']=='pass' and qa['all_400_calls_source_reaudited'] is True
    assert qa['old_320_call_rows_unchanged'] is True and qa['old_16_summary_rows_unchanged'] is True
    tm=json.loads((timing/'source_data/diffusion_fm_timing_manifest.json').read_text())
    for name,value in tm['outputs'].items():assert sha(timing/name)==value,name
    old_timing_rows=rows(paper/'source_data/diffusion_fm_timing_summary.csv')
    assert [x for x in old_timing_rows if x['pde']!='nsnonbounded']==[x for x in rows(timing/'source_data/diffusion_fm_timing_summary.csv') if x['pde']!='nsnonbounded']
    audit.mkdir(parents=True); changed=[]

    def save(relative, content, source=None):
        target=paper/relative
        backup=audit/'before'/relative
        previous=sha(target) if target.exists() else None
        if target.exists():
            backup.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(target,backup)
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(content if isinstance(content,bytes) else content.encode())
        changed.append(dict(path=relative,before_sha256=previous,after_sha256=sha(target),source=source))

    main_names=['full-forward-results','full-inverse-results','sparse-forward-results',
                'sparse-inverse-results','sparse-joint-results']
    main_blocks={name:(tables/(name+'.tex')).read_text() for name in main_names}
    for name,block in main_blocks.items():save('source_data/main_tables/'+name+'.tex',block,str(tables/(name+'.tex')))
    complete=(paper/'source_data/complete_main_tables.tex').read_text();replaced=[]
    def replace(block):
        for name,new in main_blocks.items():
            if r'\label{tab:'+name+'}' in block.group():
                replaced.append(name);return new.rstrip()
        return block.group()
    complete=re.sub(r'\\begin\{table\*?\}.*?\\end\{table\*?\}',replace,complete,flags=re.S)
    assert sorted(replaced)==sorted(main_names)
    save('source_data/complete_main_tables.tex',complete,'five validated main-table fragments')
    old_caption='Results use the archived method-specific Smooth files and sensor locations.'
    new_caption=(r'Results use method-specific Smooth examples and sensor locations; the Navier--Stokes FM4PDE entries use the main model described in Appendix~\ref{subsec:training-details}.')
    for task in ['forward','inverse','both']:
        source=tables/f'diffusion-{task}.tex';text=source.read_text()
        assert text.count(old_caption)==1
        save(f'source_data/diffusion_comparison_{task}.tex',text.replace(old_caption,new_caption),str(source))
    for name in ['main_figure_values.csv','diffusion_comparison_summary.csv']:
        save('source_data/'+name,(tables/name).read_bytes(),str(tables/name))
    save('audit/table_value_provenance.csv',(tables/'table_value_provenance.csv').read_bytes(),str(tables/'table_value_provenance.csv'))
    manifest_path=paper/'source_data/diffusion_comparison_manifest.json'
    manifest=json.loads(manifest_path.read_text());old_manifest=copy.deepcopy(manifest)
    updated={ (x['task'],x['pde'],x['field'],x['method'],int(x['steps'])):x for x in rows(tables/'diffusion_comparison_summary.csv') }
    count=0
    for row in manifest['rows']:
        if row['pde']!='nsnonbounded' or row['method']!='FM4PDE':continue
        source=updated[row['task'],row['pde'],row['field'],row['method'],row['steps']]
        for key in row:
            row[key]=(float(source[key]) if key in {'mean','sd'} else
                      int(source[key]) if key in {'n','steps'} else source[key])
        count+=1
    assert count==4
    for old,new in zip(old_manifest['rows'],manifest['rows']):
        if old!=new:assert new['pde']=='nsnonbounded' and new['method']=='FM4PDE'
    for name in manifest['outputs']:manifest['outputs'][name]=sha(paper/'source_data'/name)
    manifest['scope']='Archived DiffusionPDE and non-NS FM4PDE results, with four NS FM4PDE Smooth entries replaced by the completed main-model evaluation. Method-specific test examples are not paired.'
    manifest['ns_main_replacement']=dict(entries=4,examples_per_entry=1000,
        summary_sha256=sha(Path('/home/tat512/C01Python/audit/ns_main_revision_0909/tables/ns_main_summary.csv')),
        export_manifest_sha256=sha(tables/'export_manifest.json'),
        archived_snapshot_scope='snapshot_sha256, records and complete_cells continue to identify the preserved original DiffusionPDE snapshot')
    save('source_data/diffusion_comparison_manifest.json',json.dumps(manifest,indent=2)+'\n','four validated NS entries; updated source-output hashes')
    for source in sorted((timing/'source_data').iterdir()):
        save('source_data/'+source.name,source.read_bytes(),str(source))
    for suffix in ['pdf','png']:
        source=timing/f'figures/diffusion_fm_timing.{suffix}'
        save(f'figures/diffusion_fm_timing.{suffix}',source.read_bytes(),str(source))
    # Only the timing section of the supplement is changed.
    supp=paper/'supplement.tex';text=supp.read_text()
    start=text.index(r'\section{Numerical Sampling Times}')
    end=text.index(r'\section{NS Common-Observation and Residual Diagnostics}',start)
    segment=text[start:end]
    source_table=(timing/'source_data/diffusion_fm_timing_table.tex').read_text()
    ns_line=next(x for x in source_table.splitlines() if x.startswith('Navier--Stokes & '))
    segment,n=re.subn(r'^Navier--Stokes & .*$',lambda _:ns_line,segment,flags=re.M);assert n==1
    note=(r'Navier--Stokes timings use the $44{,}121{,}218$-parameter FM4PDE model used in the main comparisons. '
          r'The original Navier--Stokes model is retained in the ablation studies below. '
          r'For each PDE, both methods use the same 20 inputs, observation masks, and RTX~4090 GPU, '
          r'with float32 arithmetic and TF32 disabled.'+'\n\n')
    segment=segment.replace(r'\section{Numerical Sampling Times}'+'\n',r'\section{Numerical Sampling Times}'+'\n'+note,1)
    save('supplement.tex',text[:start]+segment+text[end:],'updated NS timing row and model/precision description only')
    # Keep the standalone source table's caption below its tabular, with correct protocol scope.
    path=paper/'source_data/diffusion_fm_timing_table.tex';content=path.read_text()
    caption=re.search(r'\\caption\{[^\n]+\}\n\\label\{[^\n]+\}\n',content).group()
    revised=caption.replace('All calls use the frozen protocol','The PDE-specific timing protocols are specified')
    content=content.replace(caption,'').replace(r'\end{table}',revised+r'\end{table}')
    path.write_text(content)
    for item in changed:
        if item['path']=='source_data/diffusion_fm_timing_table.tex':item['after_sha256']=sha(path);item['presentation_change']='caption below table; protocol wording covers separate NS protocol'
    tm['outputs']['source_data/diffusion_fm_timing_table.tex']=sha(path)
    tm['presentation_update']='Source-table caption placed below the table and references PDE-specific protocols; numerical values and all other exported artifacts unchanged.'
    path=paper/'source_data/diffusion_fm_timing_manifest.json';path.write_text(json.dumps(tm,indent=2)+'\n')
    for item in changed:
        if item['path']=='source_data/diffusion_fm_timing_manifest.json':item['after_sha256']=sha(path)
    for name,value in tm['outputs'].items():assert sha(paper/name)==value,name
    assert [x for x in rows(paper/'source_data/diffusion_fm_timing_summary.csv') if x['pde']!='nsnonbounded']==[x for x in old_timing_rows if x['pde']!='nsnonbounded']
    for name in main_names:assert sha(paper/f'source_data/main_tables/{name}.tex')==sha(tables/(name+'.tex'))
    assert len(changed)==23
    report=dict(status='pass',modified_files=changed,count=23,main_tables=5,diffusion_tables=3,
        candidate_source_csvs_copied_byte_identically=True,non_ns_timing_summary_rows_unchanged=16,
        timing_calls=400,new_ns_timing_calls=80,main_and_response_untouched=True,
        supplement_change_scope='Numerical Sampling Times section only',
        script_sha256=sha(Path(__file__)))
    (audit/'integration_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print('INTEGRATED 23 specified files; source and timing hashes pass:',audit)


if __name__=='__main__':main()
