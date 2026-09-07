# SPDX-License-Identifier: Apache-2.0
"""Run real UMT5 encoding in a separate process with sampled memory/time limits."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mode", choices=["cpu-bf16", "stream-cpu-fp32", "stream-mps-fp32"], default="stream-mps-fp32")
    parser.add_argument("--max-seconds", type=float, default=600)
    parser.add_argument("--max-rss-gib", type=float, default=17)
    parser.add_argument("--min-available-gib", type=float, default=2)
    args = parser.parse_args()
    if any(not math.isfinite(v) or v <= 0 for v in [args.max_seconds, args.max_rss_gib, args.min_available_gib]):
        parser.error("All limits must be finite and positive")
    args.output.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    command = [sys.executable, "-m", "experiments.wan_adapter.text_cache.worker", "--weights", str(args.weights.resolve()),
               "--output", str(args.output.resolve()), "--mode", args.mode]
    started = time.monotonic()
    reason, peak_rss, minimum_available = None, 0, psutil.virtual_memory().available
    with (args.output / "encoding.log").open("x") as log, (args.output / "memory.jsonl").open("x") as memory:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment, start_new_session=True)
        monitored = psutil.Process(process.pid)
        try:
            while process.poll() is None:
                try:
                    rss = monitored.memory_info().rss
                except psutil.NoSuchProcess:
                    break
                available = psutil.virtual_memory().available
                elapsed = time.monotonic() - started
                peak_rss, minimum_available = max(peak_rss, rss), min(minimum_available, available)
                memory.write(json.dumps({"elapsed_seconds": elapsed, "rss_bytes": rss, "system_available_bytes": available}) + "\n")
                memory.flush()
                if rss > args.max_rss_gib * 2**30:
                    reason = "process_rss_limit"
                elif available < args.min_available_gib * 2**30:
                    reason = "system_available_memory_limit"
                elif elapsed > args.max_seconds:
                    reason = "elapsed_time_limit"
                if reason:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                    break
                time.sleep(.5)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            process.wait()
    completed = process.returncode == 0 and reason is None and (args.output / "manifest.json").is_file()
    result = {"status": "complete" if completed else "failed", "exit_code": process.returncode,
        "stop_reason": reason, "elapsed_seconds": time.monotonic() - started,
        "sampled_peak_rss_bytes": peak_rss, "sampled_minimum_available_bytes": minimum_available,
        "sampling_interval_seconds": .5, "limits": {"max_seconds": args.max_seconds,
            "max_rss_gib": args.max_rss_gib, "min_available_gib": args.min_available_gib},
        "scope": "Monitors encoder process RSS and host available RAM; sampled values can miss short peaks."}
    with (args.output / "watchdog.json").open("x") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if completed else 1)


if __name__ == "__main__":
    main()
