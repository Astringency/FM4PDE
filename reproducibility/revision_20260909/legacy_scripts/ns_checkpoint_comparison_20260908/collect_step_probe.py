from pathlib import Path
import datetime,json,subprocess,time
ROOT=Path(__file__).resolve().parent
while True:
    errors=[]
    try:
        subprocess.run(["rsync","-a","--exclude=*.tmp","--timeout=90","-e","ssh -o ConnectTimeout=15 -o ServerAliveInterval=20","zjinzxf2025@175.102.135.216:/data1/zjinzxf2025/C01Python/ns_checkpoint_comparison_20260908/step_probe/",str(ROOT/"step_probe")+"/"],check=True,timeout=120)
    except Exception as exc:
        errors.append(str(exc))
    rows=list((ROOT/"step_probe/evaluation").rglob("sample*.json"))
    done=[s for s in [0,1] if (ROOT/f"step_probe/complete_{s}.json").exists()]
    state=dict(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),saved=len(rows),planned=72,remaining=72-len(rows),terminal_workers=done,collector_errors=errors)
    (ROOT/"step_probe_progress.json").write_text(json.dumps(state,indent=2)+"\n")
    print(state,flush=True)
    if len(rows)==72 and len(done)==2:
        with (ROOT/"step_probe_report.log").open("w") as log:
            r=subprocess.run(["/home/tat512/.conda/envs/fm4pde/bin/python","/home/tat512/C01Python/FM4PDE/plot/report_ns_checkpoint_step_probe.py","--results",str(ROOT/"step_probe"),"--inputs","/home/tat512/C01Python/audit/ns_loss_spectrum_20260907/inputs_v2","--diffusion-results","/home/tat512/C01Python/audit/ns_loss_spectrum_20260907/ns_results_v3","--output",str(ROOT/"step_probe_report")],stdout=log,stderr=subprocess.STDOUT,timeout=600)
        state.update(report_status="complete" if r.returncode==0 else "failed",report_exit=r.returncode)
        (ROOT/"step_probe_progress.json").write_text(json.dumps(state,indent=2)+"\n")
        print("REPORT",r.returncode,flush=True)
        break
    time.sleep(45)
