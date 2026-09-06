"""Terminal progress for both legacy-comparison launchers; no GPU imports."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def progress_lines(index, pid, state, step):
    if state.get("pid") != pid:
        return [f"GPU worker {index}: starting (pid {pid})", "  Waiting for worker status..."]
    done, total = state["completed"], state["total"]
    fraction = done / total if total else (1.0 if state["status"] == "complete" else 0.0)
    filled = min(24, int(24 * fraction))
    bar = "#" * filled + "-" * (24 - filled)
    line = f"{state['device']} [{bar}] {done}/{total} {fraction:6.1%} {state['status']}"
    detail = f"  {state['label']}"
    if step and state["status"] == "sampling":
        detail += f" | step {step.get('step', 0)}/{step.get('num_steps', '?')}"
        error = step.get("rel_l2_u")
        if error is not None:
            detail += f" | err u={error:.4g}"
    return [line, detail]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pids", type=int, nargs="+", required=True)
    args = parser.parse_args()
    stopped = False

    def stop(*_):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    tty = sys.stdout.isatty()
    previous_lines = 0
    last_print = -float("inf")
    while True:
        live = False
        lines = []
        for index, pid in enumerate(args.pids):
            try:
                os.kill(pid, 0)
                live = True
            except ProcessLookupError:
                pass
            state = read_json(args.output / f"worker{index}.status.json")
            step = read_json(args.output / f"worker{index}.progress.json")
            lines.extend(progress_lines(index, pid, state, step))
        final = stopped or not live
        if tty or final or time.monotonic() - last_print >= 30:
            if tty:
                if previous_lines:
                    sys.stdout.write(f"\033[{previous_lines}A")
                for line in lines:
                    # Keep each display line on one terminal row.
                    try:
                        width = os.get_terminal_size(sys.stdout.fileno()).columns
                    except OSError:
                        width = 120
                    sys.stdout.write("\033[2K" + line[:max(1, width - 1)] + "\n")
            else:
                sys.stdout.write("\n".join(lines) + "\n")
            sys.stdout.flush()
            previous_lines = len(lines)
            last_print = time.monotonic()
        if final:
            break
        time.sleep(0.5)


if __name__ == "__main__":
    main()
