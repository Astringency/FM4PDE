"""Audit and render each completed PDE comparison while other jobs continue."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from experiments.optimizer_diagnostics.hard_sampling import location
from experiments.optimizer_diagnostics.study import PDES, write


def stage(root,name,module,*args):
    with (root/'hard_sampling'/f'{name}.log').open('a') as stream:
        process=subprocess.run([sys.executable,'-u','-m',module,'--root',str(root),*args],stdout=stream,stderr=subprocess.STDOUT)
    write(root/'hard_sampling'/f'{name}.exit.json',dict(exit_code=process.returncode,time=time.time()))
    assert process.returncode==0,(name,process.returncode)


def main(root):
    processed={};continuation_done=False
    while True:
        pending=[];summaries=[]
        for pde in PDES:
            out=location(root,pde)
            expected=['original','lr_control_128','beta2_128' if pde=='nsnonbounded' else 'selected_128']
            required=expected+(['beta2_512','lr_control_512'] if pde=='nsnonbounded' else [])
            completed=[]
            for name in required:
                receipt=out/f'{name}.exit.json'
                if receipt.exists():
                    assert json.loads(receipt.read_text())['exit_code']==0,(pde,name)
                    assert (out/'runs'/name/'complete.json').exists()
                    completed.append(name)
            # During NS continuation a complete.json may precede its process exit.
            present=sorted(p.parent.name for p in (out/'runs').glob('*/complete.json'))
            ready=all(v in completed for v in expected) and sorted(completed)==present
            if pde in ['helmholtz','darcy']: ready=ready and (root/'runs'/pde/'run.exit.json').exists()
            signature=tuple(completed)
            if ready and processed.get(pde)!=signature:
                stage(root,'audit_'+pde,'experiments.optimizer_diagnostics.sampling_audit','--pde',pde)
                stage(root,'figures_'+pde,'experiments.optimizer_diagnostics.sampling_figures','--pde',pde)
                processed[pde]=signature
            if processed.get(pde):
                result=json.loads((out/'summary.json').read_text())
                summaries.append(dict(pde=pde,**result))
            if any(v not in completed for v in required) or processed.get(pde)!=signature:
                pending.append(pde)
        queue=root/'continuation/nsnonbounded/queue.json'
        if not continuation_done and queue.exists() and json.loads(queue.read_text()).get('state')=='complete':
            stage(root,'audit_continuation','experiments.optimizer_diagnostics.continuation_audit')
            continuation_done=True
        result=dict(state='complete' if not pending and continuation_done else 'running',pending=pending,
            continuation_verified=continuation_done,models=summaries,time=time.time())
        write(root/'hard_sampling/finalization.json',result)
        write(root/'report/sampling_summary.json',result)
        lines=['# 五类 PDE 困难样本采样对照','',
            '每类按历史原模型的解场误差预先固定最差 16 个 ID 样本。两个随机种子先按样本平均；每个候选与相同 batch 重跑的原模型严格配对，观测、采样参数和全部 101 次噪声一致。正改善率表示误差降低。',
            '历史采样值仅用于选样，避免把 batch 数值差异算作续训收益。这些结果仅适用于选定的困难样本，不能代表总体泛化性能。','',
            '| PDE | 候选 | 原模型解场误差 | 续训解场误差 | 相对改善 | 改善例数 | ≥10% / ≥20% | 误差差值的 95% 区间（百分点） |',
            '|---|---|---:|---:|---:|---:|---:|---|']
        for model in summaries:
            for variant in model['variants']:
                d=variant['fields']['u'];ci=d['paired_difference_ci95']
                lines.append(f"| {model['pde']} | {variant['variant']} | {100*d['original_mean']:.3f}% | {100*d['candidate_mean']:.3f}% | {d['aggregate_improvement_pct']:+.3f}% | {d['improved']}/16 | {d['improved_ge10']} / {d['improved_ge20']} | [{100*ci[0]:+.3f}, {100*ci[1]:+.3f}] |")
        lines.extend(['','区间按 16 个输入做配对 bootstrap，未校正多重比较。独立复算以保存的物理预测和真值为准，CPU float64 误差与采样记录逐例一致；预测文件、模型和输入均核对 SHA256。',
            '各 PDE 的 README、per_sample.csv、audit.json 和 figures/ 含完整系数场指标、分种子结果及所有 16 例的解场对照图。图中的 seed 0 场使用每例一致的色标；统计表使用两个种子。',
            '', '当前状态：'+result['state']+'；等待：'+(', '.join(pending) or '无')+('；NS 继续训练状态待核验' if not continuation_done else '')])
        (root/'report/SAMPLING.md').write_text('\n'.join(lines)+'\n')
        print('SAMPLING_FINALIZATION',result['state'],pending,flush=True)
        if result['state']=='complete': return
        time.sleep(30)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    main(p.parse_args().root)
