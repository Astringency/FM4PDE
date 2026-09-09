from pathlib import Path
import csv,hashlib,json
import torch
ROOT=Path(__file__).resolve().parent
torch.set_num_threads(2)
rows=[]
for i in [130,196,651,484]:
    for seed in [0,1,2]:
        for variant in ["current_common100","v260904_common100","bak_legacy100"]:
            paths=[ROOT/r/"evaluation/inverse"/variant/f"sample{i}_seed{seed}.pt" for r in ["results_v2","step_probe"]]
            aa,bb=[torch.load(p,map_location="cpu",weights_only=False) for p in paths]
            assert all(torch.equal(a,b) for a,b in zip(aa["truth"],bb["truth"]))
            assert all(torch.equal(a,b) for a,b in zip(aa["masks"],bb["masks"]))
            num=sum(float((a.double()-b.double()).square().sum()) for a,b in zip(aa["prediction"],bb["prediction"]))
            den=sum(float(a.double().square().sum()) for a in aa["prediction"])
            rows.append(dict(variant=variant,sample_id=i,seed=seed,bitwise_equal=all(torch.equal(a,b) for a,b in zip(aa["prediction"],bb["prediction"])),prediction_relative_difference=(num/den)**.5,max_absolute_difference=max(float((a.double()-b.double()).abs().max()) for a,b in zip(aa["prediction"],bb["prediction"])),main_sha256=hashlib.sha256(paths[0].read_bytes()).hexdigest(),probe_sha256=hashlib.sha256(paths[1].read_bytes()).hexdigest()))
with (ROOT/"overlap_100_step.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary=dict(pairs=len(rows),bitwise_equal_pairs=sum(r["bitwise_equal"] for r in rows),max_prediction_relative_difference=max(r["prediction_relative_difference"] for r in rows),scope="Inverse-only duplicate 100-step settings across the main comparison and the separate step probe; all matched predictions retained.")
(ROOT/"overlap_100_step_qa.json").write_text(json.dumps(summary,indent=2)+"\n")
print(summary)
