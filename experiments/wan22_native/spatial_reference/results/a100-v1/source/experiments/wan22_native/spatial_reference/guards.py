# SPDX-License-Identifier: Apache-2.0
"""Explicit stage deadlines and sampled process/CUDA resource guards."""

import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time

import psutil
import torch

from ..cuda_reference import guards as frozen
from ..cuda_reference.guards import hardware, validate_hardware


STAGE_SECONDS = {"codec": 600.0, "pair": 900.0, "clip": 1800.0}
POLL_SECONDS = 0.25


def limits(mode):
    if not isinstance(mode, str) or mode not in STAGE_SECONDS:
        raise ValueError("An explicit codec, pair or clip mode is required")
    return dict(frozen.LIMITS, seconds=STAGE_SECONDS[mode])


def atomic(path, value):
    """Replace a JSON record atomically, retaining the old file on failure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _deadline(deadline, mode, now=None):
    now = time.monotonic() if now is None else now
    cap = limits(mode)["seconds"]
    if (isinstance(deadline, bool) or not isinstance(deadline, (int, float))
            or not math.isfinite(deadline) or not math.isfinite(now)):
        raise ValueError("A finite monotonic deadline is required")
    if deadline > now + cap:
        raise ValueError("Deadline extends beyond the explicit stage limit")
    if now >= deadline:
        raise RuntimeError(f"{mode} deadline reached")


def _integer_bytes(value):
    if type(value) is not int or value < 0:
        raise RuntimeError("Invalid memory sample")
    return value


def check_sample(row, deadline, mode, now=None):
    _deadline(deadline, mode, now)
    cap = limits(mode)
    for key in ("host_rss_bytes", "cuda_reserved_bytes", "host_available_bytes", "cuda_available_bytes"):
        _integer_bytes(row[key])
    if "cuda_allocated_bytes" in row:
        _integer_bytes(row["cuda_allocated_bytes"])
    if row["host_rss_bytes"] > cap["host_rss_bytes"] or row["cuda_reserved_bytes"] > cap["cuda_reserved_bytes"]:
        raise RuntimeError("Fixed host or CUDA memory cap exceeded")
    if (row["host_available_bytes"] < cap["minimum_host_available_bytes"]
            or row["cuda_available_bytes"] < cap["minimum_cuda_available_bytes"]):
        raise RuntimeError("Available host or CUDA memory fell below 8 GiB")


class Monitor:
    def __init__(self, out, deadline, mode):
        self.out = Path(out)
        self.mode = mode
        self.limits = limits(mode)
        self.deadline = deadline
        self.started = time.monotonic()
        self.stop = threading.Event()
        self.error = None
        self.samples = 0
        self.thread_started = False
        self.thread = threading.Thread(target=self.watch, daemon=True)

    def sample(self):
        free, _ = torch.cuda.mem_get_info(0)
        return {"seconds": time.monotonic() - self.started,
                "host_rss_bytes": psutil.Process().memory_info().rss,
                "host_available_bytes": psutil.virtual_memory().available,
                "cuda_reserved_bytes": torch.cuda.memory_reserved(0),
                "cuda_allocated_bytes": torch.cuda.memory_allocated(0),
                "cuda_available_bytes": free}

    def _record_stop(self, error):
        self.error = str(error)
        atomic(self.out / "watchdog-stop.json", {"status": "stopped", "mode": self.mode,
               "limits": self.limits, "error_type": type(error).__name__, "reason": self.error})

    def _terminal(self, error=None):
        atomic(self.out / "monitor-terminal.json", {
            "status": "failed" if error is not None or self.error else "complete",
            "mode": self.mode, "limits": self.limits, "sample_count": self.samples,
            "elapsed_seconds": time.monotonic() - self.started,
            "error_type": type(error).__name__ if error is not None else None,
            "error": str(error) if error is not None else self.error,
        })

    def check(self):
        if self.error:
            raise RuntimeError(self.error)
        try:
            # Reject an invalid deadline before calling any CUDA sampler.
            _deadline(self.deadline, self.mode)
            check_sample(self.sample(), self.deadline, self.mode)
        except BaseException as error:
            self._record_stop(error)
            raise

    def watch(self):
        try:
            with (self.out / "memory.jsonl").open("x") as stream:
                while not self.stop.is_set():
                    row = self.sample()
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    stream.flush()
                    self.samples += 1
                    check_sample(row, self.deadline, self.mode)
                    self.stop.wait(POLL_SECONDS)
        except BaseException as error:
            try:
                self._record_stop(error)
                self._terminal(error)
            finally:
                # A stuck CUDA call in the main thread must not disable the guard.
                os._exit(124)

    def __enter__(self):
        try:
            self.check()
            self.thread.start()
            self.thread_started = True
            return self
        except BaseException as error:
            self._terminal(error)
            raise

    def __exit__(self, kind, error, traceback):
        self.stop.set()
        try:
            if self.thread_started:
                self.thread.join(timeout=3)
                if self.thread.is_alive():
                    raise RuntimeError("Resource-monitor thread did not stop")
            if error is None and self.error:
                raise RuntimeError(self.error)
            if error is None:
                _deadline(self.deadline, self.mode)
        except BaseException as cleanup_error:
            self._terminal(cleanup_error)
            raise
        self._terminal(error)
        return False


def stop_child(proc):
    """Terminate one owned worker, escalating to kill after a bounded wait."""
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
    except ProcessLookupError:
        pass
    except BaseException:
        # A failed graceful termination must still attempt the hard stop.
        proc.kill()
        proc.wait(timeout=3)
        return
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=3)


def _tree_sample(proc, began):
    """Sample the supervisor and this worker's descendants, once per PID."""
    parent = psutil.Process()
    processes = {parent.pid: parent}
    try:
        worker = psutil.Process(proc.pid)
        processes[worker.pid] = worker
        for child in worker.children(recursive=True):
            processes[child.pid] = child
    except psutil.NoSuchProcess:
        if proc.poll() is None:
            raise RuntimeError("Worker disappeared before its exit could be observed")
    rss = 0
    counted = 0
    for process in processes.values():
        try:
            rss += _integer_bytes(process.memory_info().rss)
            counted += 1
        except psutil.NoSuchProcess:
            continue
    available = _integer_bytes(psutil.virtual_memory().available)
    return {"seconds": time.monotonic() - began, "combined_rss_bytes": rss,
            "host_available_bytes": available, "processes_sampled": counted}


def supervise(proc, out, deadline, mode):
    """Watch the owned worker; retain terminal evidence even if cleanup fails."""
    out = Path(out)
    began = time.monotonic()
    error = None
    cleanup_error = None
    peak = 0
    minimum = None
    cap = None
    try:
        cap = limits(mode)
        _deadline(deadline, mode)
        with (out / "parent-memory.jsonl").open("x") as stream:
            while proc.poll() is None:
                _deadline(deadline, mode)
                row = _tree_sample(proc, began)
                peak = max(peak, row["combined_rss_bytes"])
                minimum = row["host_available_bytes"] if minimum is None else min(minimum, row["host_available_bytes"])
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                if (row["combined_rss_bytes"] > cap["host_rss_bytes"]
                        or row["host_available_bytes"] < cap["minimum_host_available_bytes"]):
                    raise RuntimeError("Parent memory guard stopped the worker")
                time.sleep(POLL_SECONDS)
            _deadline(deadline, mode)
    except BaseException as caught:
        error = caught
        atomic(out / "watchdog-stop.json", {"status": "stopped", "mode": mode, "limits": cap,
               "error_type": type(caught).__name__, "reason": str(caught)})
    finally:
        try:
            stop_child(proc)
        except BaseException as caught:
            cleanup_error = caught
        finally:
            atomic(out / "terminal.json", {
                "status": "complete" if proc.returncode == 0 and error is None and cleanup_error is None else "failed",
                "exit_code": proc.returncode, "mode": mode, "limits": cap,
                "error_type": type(error).__name__ if error is not None else None,
                "error": str(error) if error is not None else None,
                "cleanup_error_type": type(cleanup_error).__name__ if cleanup_error is not None else None,
                "cleanup_error": str(cleanup_error) if cleanup_error is not None else None,
                "elapsed_seconds": time.monotonic() - began,
                "peak_combined_rss_bytes": peak, "minimum_host_available_bytes": minimum,
            })
    if error is not None:
        raise error
    if cleanup_error is not None:
        raise cleanup_error
