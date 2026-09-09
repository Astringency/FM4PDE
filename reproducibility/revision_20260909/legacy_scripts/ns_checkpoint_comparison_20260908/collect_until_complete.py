from pathlib import Path
import collections, datetime, json, subprocess, time
ROOT=Path(__file__).resolve().parent
REMOTE=["zhangxifeng@192.168.191.197:/research_data/users/zhangxifeng/C01Python/ns_checkpoint_comparison_20260908/results_v2/", "zjinzxf2025@175.102.135.216:/data1/zjinzxf2025/C01Python/ns_checkpoint_comparison_20260908/results_v2/"]
while True:
    errors=[]
    for source in REMOTE:
        try:
            subprocess.run(["rsync","-a","--exclude=*.tmp","--timeout=90",source,str(ROOT/"results_v2")+"/"],check=True,timeout=120,capture_output=True)
        except Exception as exc:errors.append(str(exc))
    rows=[json.loads(p.read_text()) for p in (ROOT/"results_v2/evaluation").rglob("sample*.json")]
    groups=collections.Counter((r["task"],r["variant"]) for r in rows)
    terminal=[i for i in range(4) if (ROOT/f"results_v2/complete_{i}.json").exists()]
    state=dict(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),planned=1152,saved=len(rows),finite=sum(r["status"]=="complete" for r in rows),remaining=1152-len(rows),terminal_workers=terminal,groups={"/".join(k):v for k,v in sorted(groups.items())},collector_errors=errors)
    (ROOT/"progress.json").write_text(json.dumps(state,indent=2)+"\n")
    print(state["checked_utc"],len(rows),"/1152",terminal,errors,flush=True)
    if len(rows)==1152 and len(terminal)==4:
        with (ROOT/"report.log").open("w") as log:
            run=subprocess.run(["/home/tat512/.conda/envs/fm4pde/bin/python","/home/tat512/C01Python/FM4PDE/plot/report_ns_checkpoint_comparison.py","--results",str(ROOT/"results_v2"),"--inputs","/home/tat512/C01Python/audit/ns_loss_spectrum_20260907/inputs_v2","--diffusion-results","/home/tat512/C01Python/audit/ns_loss_spectrum_20260907/ns_results_v3","--output",str(ROOT/"report_v2")],stdout=log,stderr=subprocess.STDOUT,timeout=1800)
        state.update(report_status="complete" if run.returncode==0 else "failed",report_exit=run.returncode)
        (ROOT/"progress.json").write_text(json.dumps(state,indent=2)+"\n")
        print("REPORT",run.returncode,flush=True)
        break
    time.sleep(45)
