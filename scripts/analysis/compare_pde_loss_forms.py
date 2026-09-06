#!/usr/bin/env python3
"""Compare independently tuned MSE/RMS guidance on identical held-out samples."""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
import math
import statistics
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.analysis.analyze_pde_guidance_schedules import LABELS, PDE_LABELS, bootstrap_ratio
from scripts.tuning.compare_pde_guidance_schedules import SCHEDULES, write_csv, write_json

SQL = """SELECT form, stage, pde, task, schedule, zeta_pde, COUNT(*) AS n,
SUM(CASE WHEN status <> 'ok' OR primary_error IS NULL THEN 1 ELSE 0 END) AS failed,
CASE WHEN COUNT(primary_error)=COUNT(*) THEN AVG(primary_error) END AS primary_error,
CASE WHEN COUNT(pde_residual_norm)=COUNT(*) THEN AVG(pde_residual_norm) END AS pde_residual_norm,
CASE WHEN COUNT(rel_l2_a)=COUNT(*) THEN AVG(rel_l2_a) END AS rel_l2_a,
CASE WHEN COUNT(rel_l2_u)=COUNT(*) THEN AVG(rel_l2_u) END AS rel_l2_u
FROM loss_observations
GROUP BY form, stage, pde, task, schedule, zeta_pde
ORDER BY form, stage, pde, task, schedule, zeta_pde"""


def read_rows(root, form):
    with (root / 'per_sample.csv').open() as handle:
        return [dict(row, form=form) for row in csv.DictReader(handle) if row['task'] == 'both']


def ground_truth_reference(root, protocol, output):
    """Check the same evaluation operator on held-out true states (no fitting)."""
    import torch
    from sampling.config import AblationConfig
    from sampling.losses import compute_guidance_losses
    from sampling.masks import PairMasks
    from sampling.metrics import pde_residual_norm_per_sample
    from sampling.state import SplitState
    from scripts.tuning.compare_pde_guidance_schedules import combine_truths
    rows=[]
    for pde,task in protocol['cells']:
        if task != 'both': continue
        cache=root/'cache'/f'{pde}_ground_truth.pt'
        if not cache.exists(): continue  # Partial report snapshots contain receipts only.
        truths=torch.load(cache,map_location='cpu',weights_only=False)['truths']
        cfg=AblationConfig(**protocol['configs'][f'{pde}/{task}'])
        cfg.device='cpu';cfg.guidance_components='obs_only';cfg.zeta_pde=0.0
        ids=protocol['holdout_sample_ids']
        for begin in range(0,len(ids),protocol['batch_size']):
            batch_ids=ids[begin:begin+protocol['batch_size']]
            gt=combine_truths(truths,batch_ids,'cpu')
            masks=PairMasks(torch.zeros_like(gt.coef),torch.zeros_like(gt.sol),{})
            with torch.no_grad():
                losses=compute_guidance_losses(SplitState(gt.coef,gt.sol),gt,masks,cfg)
                values=pde_residual_norm_per_sample(losses.pde_residual,losses.pde_residual_mask)
            rows.extend(dict(pde=pde,sample_id=sample_id,reference_residual=value,residual_mode=cfg.residual_mode)
                        for sample_id,value in zip(batch_ids,values))
    summary=[]
    for pde in dict.fromkeys(r['pde'] for r in rows):
        values=[r['reference_residual'] for r in rows if r['pde']==pde]
        summary.append(dict(pde=pde,n=len(values),reference_residual_mean=statistics.mean(values)))
    write_csv(output/'ground_truth_reference_per_sample.csv',rows)
    write_csv(output/'ground_truth_reference.csv',summary)
    return summary


def compare(mse_root, rms_root, output, allow_partial=False, schedule_report_root=None):
    output.mkdir(parents=True, exist_ok=True)
    protocols = {form: json.loads((root / 'protocol.json').read_text())
                 for form, root in [('mse', mse_root), ('rms', rms_root)]}
    for key in ['seed', 'tune_sample_ids', 'holdout_sample_ids', 'batch_size', 'steps', 'time_grid',
                'sampler', 'clip_mode', 'clip_threshold', 'sensor_mode', 'num_obs', 'test_type', 'schedules']:
        if protocols['mse'][key] != protocols['rms'][key]:
            raise ValueError(f'Unpaired protocols: {key}')
    for key, config in protocols['rms']['configs'].items():
        if config != protocols['mse']['configs'][key]:
            raise ValueError(f'Unpaired original main configs: {key}')
    if protocols['rms'].get('pde_guidance_reduction') != 'rms':
        raise ValueError('RMS root does not specify RMS guidance')
    if not allow_partial and not all((root/'complete.json').exists() for root in [mse_root,rms_root]):
        raise ValueError('Both experiments must be complete')
    rows = read_rows(mse_root, 'mse') + read_rows(rms_root, 'rms')
    fields = ['form','stage','pde','task','schedule','zeta_pde','sample_id','status',
              'primary_error','pde_residual_norm','rel_l2_a','rel_l2_u']
    numeric = {'zeta_pde','primary_error','pde_residual_norm','rel_l2_a','rel_l2_u'}
    with sqlite3.connect(output/'loss_comparison.sqlite') as conn:
        conn.execute('DROP TABLE IF EXISTS loss_observations')
        conn.execute('CREATE TABLE loss_observations (form TEXT, stage TEXT, pde TEXT, task TEXT, schedule TEXT, zeta_pde REAL, sample_id INTEGER, status TEXT, primary_error REAL, pde_residual_norm REAL, rel_l2_a REAL, rel_l2_u REAL)')
        values = [[None if r[f]=='' else float(r[f]) if f in numeric else int(r[f]) if f=='sample_id' else r[f] for f in fields] for r in rows]
        conn.executemany('INSERT INTO loss_observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',values)
        conn.row_factory=sqlite3.Row
        summary=[dict(row) for row in conn.execute(SQL)]
    write_csv(output/'loss_summary.csv',summary)
    write_csv(output/'loss_per_sample.csv',rows)
    (output/'loss_summary.sql').write_text(SQL+';\n')
    by_key={tuple(r[k] for k in ['form','stage','pde','task','schedule','zeta_pde']):r for r in summary}
    tune_n=len(protocols['mse']['tune_sample_ids'])
    hold_n=len(protocols['mse']['holdout_sample_ids'])
    comparisons=[]
    fingerprints={}
    audited=set()
    import torch
    def get_holdout(form,pde,candidate):
        schedule,zeta=candidate['schedule'],candidate['zeta_pde']
        group=sorted([r for r in rows if r['form']==form and r['pde']==pde and r['stage']=='holdout' and r['schedule']==schedule and float(r['zeta_pde'])==zeta],key=lambda r:int(r['sample_id']))
        if len(group)!=hold_n:
            if allow_partial: return None,None
            raise ValueError(f'Missing holdout: {form}/{pde}/{candidate}')
        if [int(r['sample_id']) for r in group]!=sorted(protocols['mse']['holdout_sample_ids']):
            raise ValueError('Unpaired holdout sample IDs')
        if any(r['status']!='ok' or r['primary_error']=='' for r in group): return None,None
        for row in group:
            path=Path(row['run_dir'])/'result.pt'
            if str(path) in audited: continue
            payload=torch.load(path,map_location='cpu',weights_only=False)
            cfg=payload['config']
            if cfg.get('pde_guidance_reduction','mse')!=form: raise ValueError(f'Incorrect loss form: {path}')
            batch_ids=tuple(payload['ground_truth_metadata']['sample_ids'])
            key=(pde,batch_ids)
            digest=hashlib.sha256()
            for tensor in [payload['coef_ground_truth'],payload['sol_ground_truth'],payload['masks']['coef'],payload['masks']['sol']]:
                digest.update(tensor.contiguous().numpy().tobytes())
            digest.update(json.dumps({k:cfg[k] for k in ['sample_seed','mask_seed','batch_size','time_grid','num_steps','sampler_phase','zeta_obs_a','zeta_obs_u','clip_threshold','loss_state','gradient_target','checkpoint_path']},sort_keys=True).encode())
            if key in fingerprints and fingerprints[key]!=digest.hexdigest(): raise ValueError(f'Unpaired cross-loss run: {path}')
            fingerprints[key]=digest.hexdigest(); audited.add(str(path))
        return by_key[(form,'holdout',pde,'both',schedule,zeta)], [float(r['primary_error']) for r in group]
    for pde,task in protocols['rms']['cells']:
        for cut in ['tune_selected', *SCHEDULES]:
            selected={}
            for form in ['mse','rms']:
                candidates=[r for r in summary if r['form']==form and r['stage']=='tune' and r['pde']==pde and r['task']==task and r['n']==tune_n and r['failed']==0 and r['primary_error'] is not None and (cut=='tune_selected' or r['schedule']==cut)]
                if not candidates: break
                selected[form]=min(candidates,key=lambda r:(r['primary_error'],r['zeta_pde'],r['schedule']))
            if len(selected)!=2: continue
            a,av=get_holdout('mse',pde,selected['mse']); b,bv=get_holdout('rms',pde,selected['rms'])
            if allow_partial and (a is None or b is None): continue
            ratio=b['primary_error']/a['primary_error'] if a and b else None
            lo,hi=bootstrap_ratio(bv,av) if ratio is not None else (None,None)
            row=dict(pde=pde,pde_label=PDE_LABELS[pde],task=task,comparison=cut,
                     comparison_label='筛选集同时选时机与权重' if cut=='tune_selected' else LABELS[cut],
                     mse_schedule=selected['mse']['schedule'],rms_schedule=selected['rms']['schedule'],
                     mse_zeta=selected['mse']['zeta_pde'],rms_zeta=selected['rms']['zeta_pde'],
                     mse_error=a['primary_error'] if a else None,rms_error=b['primary_error'] if b else None,
                     ratio_rms_to_mse=ratio,change_pct=(ratio-1)*100 if ratio is not None else None,
                     ci95_low=lo,ci95_high=hi,paired_rms_wins=sum(y<x for x,y in zip(av,bv)) if av and bv else None,
                     n=hold_n,mse_residual=a['pde_residual_norm'] if a else None,rms_residual=b['pde_residual_norm'] if b else None,
                     residual_change_pct=(b['pde_residual_norm']/a['pde_residual_norm']-1)*100 if a and b and a['pde_residual_norm'] else None,
                     status='ok' if a and b else 'failed')
            for form,result in [('mse',a),('rms',b)]:
                for field in ['rel_l2_a','rel_l2_u']: row[f'{form}_{field}']=result[field] if result else None
            comparisons.append(row)
    if not comparisons: raise ValueError('No complete paired loss comparisons yet')
    write_csv(output/'loss_comparison.csv',comparisons)
    chosen=[r for r in comparisons if r['comparison']=='tune_selected']
    write_csv(output/'loss_tune_selected.csv',chosen)
    audit=dict(cross_loss_runs_checked=len(audited),paired_batches=len(fingerprints),tune_holdout_disjoint=not set(protocols['mse']['tune_sample_ids'])&set(protocols['mse']['holdout_sample_ids']),complete=not allow_partial,scope='five PDEs / both; MSE schedule control also covers forward and inverse')
    write_json(output/'loss_validation.json',audit)
    diagnostics=[]
    for form,root in [('mse',mse_root),('rms',rms_root)]:
        for pde,task in protocols['rms']['cells']:
            curves=[]
            for receipt in (root/'runs'/'tune'/pde/task/'always_on_z10').glob('batch*/receipt.json'):
                record=json.loads(receipt.read_text())
                if record['rows'][0]['status']!='ok': continue
                with (Path(record['rows'][0]['run_dir'])/'curves.csv').open() as handle:
                    curves.extend(csv.DictReader(handle))
            if not curves: continue
            initial=[r for r in curves if int(r['step'])==0]
            early=[r for r in curves if int(r['step'])<10]
            diagnostics.append(dict(form=form,pde=pde,schedule='always_on',zeta_pde=10,
                initial_guidance_residual=statistics.mean(float(r['guidance_pde_residual_norm']) for r in initial),
                initial_pde_gradient_norm=statistics.mean(float(r['grad_norm_pde']) for r in initial),
                first10_max_pde_gradient_norm=max(float(r['grad_norm_pde']) for r in early),
                batch_steps=len(curves),batch_steps_with_any_clipping=sum(float(r['clip_scale'])<0.999999 for r in curves),
                minimum_mean_clip_scale=min(float(r['clip_scale']) for r in curves)))
    write_csv(output/'loss_gradient_diagnostic.csv',diagnostics)
    references=ground_truth_reference(mse_root,protocols['mse'],output)
    for pde,task in protocols['rms']['cells']:
        pair=[r for r in diagnostics if r['pde']==pde]
        if len(pair)==2 and pair[0]['batch_steps']==pair[1]['batch_steps'] and not math.isclose(pair[0]['initial_guidance_residual'],pair[1]['initial_guidance_residual'],rel_tol=1e-6,abs_tol=1e-9):
            raise ValueError(f'Initial guidance states differ between loss forms: {pde}')
    schedule_report_root=schedule_report_root or mse_root
    artifact=copy.deepcopy(json.loads((schedule_report_root/'artifact.json').read_text()))
    manifest=artifact['manifest']; snapshot=artifact['snapshot']
    title='PDE Guidance: Schedule, Clipping and RMS Loss'
    manifest['title']=title
    manifest['generatedAt']=snapshot['generatedAt']=datetime.now(timezone.utc).isoformat()
    source=dict(id='loss_form_comparison',label='Paired MSE versus RMS guidance',path='loss_comparison.csv',query=dict(language='sql',engine='sqlite',sql=SQL,tables_used=['loss_observations'],description='Aggregate original per-sample MSE/RMS run metrics in loss_comparison.sqlite. Select schedule and zeta on tune only; compare held-out sample IDs with paired bootstrap intervals.'))
    manifest['sources'].append(source); artifact['sources'].append(copy.deepcopy(source))
    snapshot['datasets']['loss_comparisons']=comparisons
    snapshot['datasets']['loss_selected']=chosen
    snapshot['datasets']['loss_selected_display']=[dict(
        pde_label=r['pde_label'],
        mse_setting=f"{LABELS[r['mse_schedule']][0]} / {r['mse_zeta']:g}",
        rms_setting=f"{LABELS[r['rms_schedule']][0]} / {r['rms_zeta']:g}",
        mse_error=f"{r['mse_error']:.6g}" if r['mse_error'] is not None else '失败',
        rms_error=f"{r['rms_error']:.6g}" if r['rms_error'] is not None else '失败',
        change_pct=f"{r['change_pct']:+.4g}" if r['change_pct'] is not None else '—',
    ) for r in chosen]
    snapshot['datasets']['loss_schedules']=[r for r in comparisons if r['comparison']!='tune_selected' and r['ratio_rms_to_mse'] is not None]
    def prose(id,body): return dict(id=id,type='markdown',body=body,sourceId=source['id'])
    findings=[]
    for row in chosen:
        if row['change_pct'] is None:
            findings.append(f"- **{row['pde_label']}**：至少一种形式在复核集失败，不能报告有效误差比值。")
            continue
        meaning='近似持平' if abs(row['change_pct'])<1 else 'RMS 误差更低' if row['change_pct']<0 else 'MSE 误差更低'
        physics=f" 最终 PDE 残差 RMS 变化 {row['residual_change_pct']:+.1f}%。" if row['residual_change_pct'] is not None else ''
        findings.append(f"- **{row['pde_label']}**：{meaning}；RMS 相对 MSE 的重建均值变化 {row['change_pct']:+.4g}%，误差比值的配对 bootstrap 95% 区间为 [{row['ci95_low']:.6g}, {row['ci95_high']:.6g}]。"+physics)
    blocks=[prose('loss_intro','## 不平方的 PDE loss 是否更好\n\n比较使用相同逐样本总梯度裁剪阈值 50 的 MSE 与 RMS 引导，覆盖五种 PDE 的 both 任务。时机和权重均在 4 个筛选样本上独立选择，再用相同的 8 个复核样本评估；下列结论未使用复核误差挑选时机。\n\n'+ '\n'.join(findings)),
            prose('loss_read_table','### 各形式先独立选参数，再比较复核误差\n\n表中主误差为 a、u 相对 L2 误差的等权平均，Burger 为完整解场误差；越低越好。“相对变化”是 RMS / MSE − 1，负值表示 RMS 更低。接近零的变化不足以支持更换默认 loss。完整分量误差、逐样本胜出数和区间保存在配套结果中。'),
            dict(id='loss_selected_block',type='table',tableId='loss_selected_table'),
            prose('loss_physics_note','### 重建误差与物理残差分别判断\n\n下表仍是同一组按重建误差选出的参数，比较最终预测的 PDE 残差 RMS。评估定义在两种引导形式之间保持一致，所以这些数值可以直接比较；负变化表示该评价算子的残差更低，不能表述为重建精度显著提升。NS 使用端点近似，应结合真值参考残差解释；不同 PDE 的原始残差尺度不可横向排名。'),
            dict(id='loss_physics_block',type='table',tableId='loss_physics_table'),
            prose('loss_chart_note','### 固定启用方式后，比较两种 loss\n\n每组柱表示相同启用方式下、分别调权重后的 RMS / MSE 复核误差比值。小于 1 表示 RMS 更低，大于 1 表示 MSE 更低；这能区分 loss 的效果与启用时机选择的效果。失败候选不绘制比值，保留在结果 CSV 中。'),
            dict(id='loss_chart_block',type='chart',chartId='loss_ratio_chart'),
            prose('loss_method','### RMS 的定义与解释边界\n\n每个样本、每个残差分量使用 ||r||₂ / √N（掩码下 N 为有效点数），对样本取均值后按原来的内部、边界和端点权重相加。它是未平方的 L2 型目标，区别于 MAE 和历史的 ||r||₂ / N。MSE 与 RMS 的数值尺度不同，因此分别搜索原 main 权重及 0.1、1、10。所有评估 MSE、物理残差和相对重建误差保持原定义。\n\n当前 main 在端点预测上计算引导。曲线中当前噪声状态的 L_pde 与实际优化的 guidance_L_pde 不是同一个状态上的 loss；失稳诊断应看后者及引导梯度。RMS 减弱大残差的线性放大，但模型/PDE 的雅可比和全局裁剪仍会影响实际更新，不能保证更准确。\n\n本轮是 ID 数据的小样本筛选，只对 both 任务比较 loss 形式。区间未做多重比较校正；接近持平或区间跨过 1 时应扩大样本复核，再决定是否修改 main。')]
    manifest['blocks'][0]['body']='# '+title+'\n'
    manifest['blocks'][1:1]=blocks
    ns_reference=next((r for r in references if r['pde']=='nsnonbounded'),None)
    if ns_reference:
        ref_source=dict(id='ground_truth_reference',label='Same evaluation operator on held-out true states',path='ground_truth_reference.csv')
        manifest['sources'].append(ref_source);artifact['sources'].append(copy.deepcopy(ref_source))
        index=next(i for i,b in enumerate(manifest['blocks']) if b['id']=='loss_physics_note')+1
        manifest['blocks'].insert(index,dict(id='loss_ns_reference',type='markdown',sourceId=ref_source['id'],body=f"NS 的同一 endpoint_secant 算子在 {ns_reference['n']} 个复核真值上的平均残差为 {ns_reference['reference_residual_mean']:.5g}，真值也不对应零残差。因此，这个近似指标更低不一定代表真实时间轨迹更准确，NS 应优先结合重建误差判断。"))
    if schedule_report_root != mse_root:
        blocks[0]['body'] += '\n\nLoss 形式的比较使用服务器上重新运行的同机 MSE 对照；本地完成的时机实验在后文单独列出，避免将跨 GPU/环境的差异归因于 loss。'
    manifest['tables'].append(dict(id='loss_selected_table',title='独立选参后的 MSE / RMS 复核结果',dataset='loss_selected_display',sourceId=source['id'],columns=[
        dict(field='pde_label',label='PDE',type='text'),dict(field='mse_setting',label='MSE 方式 / ζ',type='text'),dict(field='mse_error',label='MSE 误差',type='text'),dict(field='rms_setting',label='RMS 方式 / ζ',type='text'),dict(field='rms_error',label='RMS 误差',type='text'),dict(field='change_pct',label='变化 (%)',type='text')]))
    manifest['tables'].append(dict(id='loss_physics_table',title='同一组参数下的最终 PDE 残差',dataset='loss_selected',sourceId=source['id'],columns=[dict(field='pde_label',label='PDE',type='text'),dict(field='mse_residual',label='MSE 引导后的残差 RMS',format='number'),dict(field='rms_residual',label='RMS 引导后的残差 RMS',format='number'),dict(field='residual_change_pct',label='残差变化 (%)',format='number')]))
    manifest['charts'].append(dict(id='loss_ratio_chart',title='相同启用方式下的 RMS / MSE 误差',dataset='loss_schedules',type='bar',sourceId=source['id'],settings={'groupMode':'grouped'},palette={'kind':'categorical'},referenceLines=[{'axis':'y','value':1,'label':'MSE 对照'}],encodings={'x':{'field':'pde_label','type':'nominal','label':'PDE'},'y':{'field':'ratio_rms_to_mse','type':'quantitative','label':'RMS / MSE 误差'},'color':{'field':'comparison_label','type':'nominal','label':'启用方式'}}))
    if allow_partial:
        snapshot['status']='partial'
        snapshot['accessIssues']=[{'id':'partial_loss','message':'实验尚未全部完成。'}]
    write_json(output/'artifact.json',artifact)
    write_json(output/'source_notes.json',dict(protocols=protocols,roots={'mse':str(mse_root),'rms':str(rms_root),'schedule_report':str(schedule_report_root)},audit=audit,chart_map={'loss_ratio_chart':'Grouped bar, one held-out ratio per PDE and schedule; tuning independent by loss form.'},required_structure='Technical summary, paired loss findings, MSE schedule findings, definitions, methods, limitations and next steps. Further questions are integrated in next-step prose.'))
    print(json.dumps({'selected':chosen,'audit':audit},indent=2,ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mse-root',type=Path,required=True)
    parser.add_argument('--rms-root',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--allow-partial',action='store_true')
    parser.add_argument('--schedule-report-root',type=Path,help='Optional separate complete MSE schedule study for the report; cross-loss metrics always use mse-root')
    args=parser.parse_args()
    compare(args.mse_root,args.rms_root,args.output_root,args.allow_partial,args.schedule_report_root)

if __name__=='__main__': main()
