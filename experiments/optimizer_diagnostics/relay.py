"""Use a temporary A800 cache, then verify and publish all outputs to server197."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time

CANONICAL="/research_data/users/zhangxifeng/C01Python/FM4PDE/outputs/pretrained/optimizer_diagnostics_20260919"
CACHE="/data1/zjinzxf2025/C01Python/FM4PDE/reproducibility/optimizer_diagnostics_20260919"
PYTHON="/data1/zjinzxf2025/miniconda3/envs/fm4pde/bin/python"


def ssh(host,command,**kwargs):
    return subprocess.run(["ssh","-o","BatchMode=yes","-o","ConnectTimeout=15",host,command],check=True,**kwargs)


def pipe(source_host,source_command,target_host,target_command):
    source=subprocess.Popen(["ssh",source_host,source_command],stdout=subprocess.PIPE)
    target=subprocess.Popen(["ssh",target_host,target_command],stdin=source.stdout)
    source.stdout.close()
    target_code=target.wait()
    source_code=source.wait()
    if target_code or source_code:
        raise RuntimeError(f"Transfer failed source={source_code} target={target_code}")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--pde",choices=["helmholtz","darcy","burger","nsnonbounded"],required=True)
    parser.add_argument("--gpu",type=int,required=True)
    args=parser.parse_args()
    pde=args.pde
    session=f"fm_opt_0919_{pde}"
    mapping=dict(pde=pde,gpu=args.gpu,canonical_output=f"{CANONICAL}/runs/{pde}",
        temporary_compute_cache=CACHE,source_host="server197",compute_host="server216",
        publish_policy="All result files SHA256 verified before atomic publication",session=session)
    ssh("server197",f"mkdir -p {CANONICAL}/remote_execution && cat > {CANONICAL}/remote_execution/{pde}.json",
        input=json.dumps(mapping,indent=2),text=True)
    ssh("server216",f"mkdir -p {CACHE}/inputs {CACHE}/spool")
    print("TRANSFER_START",pde,time.time(),flush=True)
    pipe("server197",f"tar -C {CANONICAL}/inputs -cf - {pde}",
        "server216",f"tar -C {CACHE}/inputs -xf -")
    print("TRANSFER_COMPLETE",pde,time.time(),flush=True)
    run=f"cd {CACHE}/code && {PYTHON} -u -m experiments.optimizer_diagnostics.queue run --pdes {pde} --gpu {args.gpu} --inputs {CACHE}/inputs --output {CACHE}/spool --max-memory-gib 56 --minimum-free-mib 68000 --steps 128"
    ssh("server216",f"tmux new-session -d -s {session} {shlex.quote(run)}")
    while True:
        try:
            response=ssh("server216",f"if test -f {CACHE}/spool/{pde}/run.exit.json; then cat {CACHE}/spool/{pde}/run.exit.json; else printf null; fi",capture_output=True,text=True)
            status=json.loads(response.stdout)
            if status is not None:
                print("REMOTE_EXIT",pde,status,flush=True)
                break
        except (subprocess.CalledProcessError,json.JSONDecodeError) as exc:
            print("POLL_RETRY",pde,str(exc),flush=True)
        time.sleep(30)
    # Publish failures as evidence as well, without presenting them as successes.
    ssh("server216",f"cd {CACHE}/spool/{pde} && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS")
    stage=f"{CANONICAL}/incoming_{pde}"
    ssh("server197",f"mkdir -p {stage}")
    pipe("server216",f"tar -C {CACHE}/spool -cf - {pde}","server197",f"tar -C {stage} -xf -")
    ssh("server197",f"cd {stage}/{pde} && sha256sum -c SHA256SUMS > {stage}/verification.log && test ! -e {CANONICAL}/runs/{pde} && mv {stage}/{pde} {CANONICAL}/runs/{pde}")
    print("PUBLISHED",pde,time.time(),flush=True)
    if status["exit_code"]:
        raise SystemExit(status["exit_code"])


if __name__=="__main__":
    main()
