# SPDX-License-Identifier: Apache-2.0
"""Mocked memory and process lifecycle checks. No GPU, model or cloud calls."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import types
import unittest
from unittest import mock

from ..cuda_reference import guards as frozen
from . import guards


GIB = 2**30


def memory(**changes):
    value = {"seconds": 0.0, "host_rss_bytes": 2 * GIB,
             "host_available_bytes": 16 * GIB, "cuda_reserved_bytes": 20 * GIB,
             "cuda_allocated_bytes": 19 * GIB, "cuda_available_bytes": 60 * GIB}
    return dict(value, **changes)


class Child:
    def __init__(self, code=None, stubborn=False, termination_error=False, kill_error=False):
        self.pid = 45678
        self.returncode = code
        self.stubborn = stubborn
        self.termination_error = termination_error
        self.kill_error = kill_error
        self.calls = []

    def poll(self):
        return self.returncode

    def terminate(self):
        self.calls.append("terminate")
        if self.termination_error:
            raise OSError("Injected terminate failure")

    def kill(self):
        self.calls.append("kill")
        if self.kill_error:
            raise OSError("Injected kill failure")
        self.returncode = -9

    def wait(self, timeout):
        self.calls.append(("wait", timeout))
        if self.returncode is None and self.stubborn:
            raise subprocess.TimeoutExpired("fake-worker", timeout)
        if self.returncode is None:
            self.returncode = -15
        return self.returncode


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.out = Path(self.temp.name)
        self.clock = mock.patch.object(guards.time, "monotonic", return_value=100.0)
        self.clock.start()

    def tearDown(self):
        self.clock.stop()
        self.temp.cleanup()

    def read(self, name):
        return json.loads((self.out / name).read_text())

    def test_explicit_stage_limits_and_exact_hardware_aliases(self):
        self.assertIs(guards.hardware, frozen.hardware)
        self.assertIs(guards.validate_hardware, frozen.validate_hardware)
        for mode, seconds in (("codec", 600), ("pair", 900), ("clip", 1800)):
            value = guards.limits(mode)
            self.assertEqual(value["seconds"], seconds)
            for key in frozen.LIMITS:
                if key != "seconds":
                    self.assertEqual(value[key], frozen.LIMITS[key])
            value["host_rss_bytes"] = 0
            self.assertEqual(guards.limits(mode)["host_rss_bytes"], 48 * GIB)
        for bad in (None, "", "core", "decode", 900, {}):
            with self.assertRaises(ValueError):
                guards.limits(bad)

    def test_deadline_types_stage_caps_and_exact_expiry(self):
        for mode, seconds in (("codec", 600), ("pair", 900), ("clip", 1800)):
            guards.check_sample(memory(), 100 + seconds, mode, now=100)
            with self.assertRaises(ValueError):
                guards.check_sample(memory(), 100 + seconds + 0.001, mode, now=100)
            with self.assertRaises(RuntimeError):
                guards.check_sample(memory(), 100, mode, now=100)
        for bad in (float("inf"), float("nan"), True, "700", None):
            with self.assertRaises(ValueError):
                guards.check_sample(memory(), bad, "codec", now=100)

    def test_memory_boundaries_and_malformed_samples(self):
        guards.check_sample(memory(host_rss_bytes=48 * GIB, cuda_reserved_bytes=60 * GIB,
                                   host_available_bytes=8 * GIB, cuda_available_bytes=8 * GIB), 700, "codec")
        for key, value in (("host_rss_bytes", 48 * GIB + 1), ("cuda_reserved_bytes", 60 * GIB + 1),
                           ("host_available_bytes", 8 * GIB - 1), ("cuda_available_bytes", 8 * GIB - 1)):
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                guards.check_sample(memory(**{key: value}), 700, "codec")
        for key in ("host_rss_bytes", "cuda_reserved_bytes", "host_available_bytes", "cuda_available_bytes", "cuda_allocated_bytes"):
            for bad in (-1, True, 1.5, float("nan")):
                with self.subTest(key=key, bad=bad), self.assertRaises(RuntimeError):
                    guards.check_sample(memory(**{key: bad}), 700, "codec")

    def test_atomic_write_preserves_original_on_serialization_failure(self):
        path = self.out / "record.json"
        guards.atomic(path, {"status": "first"})
        original = path.read_bytes()
        with self.assertRaises(ValueError):
            guards.atomic(path, {"value": float("nan")})
        self.assertEqual(path.read_bytes(), original)
        self.assertFalse(list(self.out.glob("*.tmp")))
        guards.atomic(path, {"status": "second"})
        self.assertEqual(self.read("record.json")["status"], "second")

    def test_monitor_rejects_bad_deadline_before_any_cuda_sample(self):
        monitor = guards.Monitor(self.out, float("inf"), "codec")
        with mock.patch.object(monitor, "sample") as sample:
            with self.assertRaises(ValueError):
                with monitor:
                    self.fail("Invalid guard entered")
        sample.assert_not_called()
        self.assertEqual(self.read("watchdog-stop.json")["error_type"], "ValueError")
        self.assertEqual(self.read("monitor-terminal.json")["status"], "failed")

    def test_monitor_normal_and_body_failure_have_terminal_records(self):
        for fail in (False, True):
            directory = self.out / str(fail)
            directory.mkdir()
            monitor = guards.Monitor(directory, 700, "codec")
            with mock.patch.object(monitor, "sample", return_value=memory()), \
                    mock.patch.object(monitor.thread, "start"), mock.patch.object(monitor.thread, "join") as join, \
                    mock.patch.object(monitor.thread, "is_alive", return_value=False):
                try:
                    with monitor:
                        if fail:
                            raise KeyboardInterrupt("Injected body interruption")
                except KeyboardInterrupt:
                    self.assertTrue(fail)
                join.assert_called_once_with(timeout=3)
            terminal = json.loads((directory / "monitor-terminal.json").read_text())
            self.assertEqual(terminal["status"], "failed" if fail else "complete")
            self.assertEqual(terminal["mode"], "codec")

    def test_monitor_thread_memory_failure_records_before_hard_exit(self):
        monitor = guards.Monitor(self.out, 700, "codec")
        with mock.patch.object(monitor, "sample", return_value=memory(cuda_reserved_bytes=61 * GIB)), \
                mock.patch.object(guards.os, "_exit", side_effect=SystemExit(124)) as hard_exit:
            with self.assertRaises(SystemExit) as caught:
                monitor.watch()
        self.assertEqual(caught.exception.code, 124)
        hard_exit.assert_called_once_with(124)
        self.assertEqual(self.read("watchdog-stop.json")["status"], "stopped")
        self.assertEqual(self.read("monitor-terminal.json")["status"], "failed")
        self.assertEqual(len((self.out / "memory.jsonl").read_text().splitlines()), 1)

    def test_monitor_file_error_and_failed_join_cannot_pass(self):
        (self.out / "memory.jsonl").write_text("existing evidence\n")
        monitor = guards.Monitor(self.out, 700, "codec")
        with mock.patch.object(monitor, "sample") as sample, \
                mock.patch.object(guards.os, "_exit", side_effect=SystemExit(124)):
            with self.assertRaises(SystemExit):
                monitor.watch()
        sample.assert_not_called()
        self.assertEqual((self.out / "memory.jsonl").read_text(), "existing evidence\n")
        another = guards.Monitor(self.out, 700, "codec")
        with mock.patch.object(another, "sample", return_value=memory()), \
                mock.patch.object(another.thread, "start"), mock.patch.object(another.thread, "join"), \
                mock.patch.object(another.thread, "is_alive", return_value=True):
            with self.assertRaises(RuntimeError):
                with another:
                    pass
        self.assertEqual(self.read("monitor-terminal.json")["status"], "failed")

    def test_tree_sample_counts_only_parent_worker_and_distinct_descendants(self):
        def process(pid, rss):
            return types.SimpleNamespace(pid=pid, memory_info=lambda: types.SimpleNamespace(rss=rss))
        parent = process(1, 2 * GIB)
        worker = process(2, 3 * GIB)
        grandchild = process(3, 4 * GIB)
        worker.children = mock.Mock(return_value=[grandchild, grandchild])
        parent.children = mock.Mock(side_effect=AssertionError("Unrelated child enumeration"))
        proc = Child(); proc.pid = 2
        with mock.patch.object(guards.psutil, "Process", side_effect=lambda pid=None: parent if pid is None else worker), \
                mock.patch.object(guards.psutil, "virtual_memory", return_value=types.SimpleNamespace(available=16 * GIB)):
            row = guards._tree_sample(proc, 99)
        self.assertEqual(row["combined_rss_bytes"], 9 * GIB)
        self.assertEqual(row["processes_sampled"], 3)
        worker.children.assert_called_once_with(recursive=True)
        parent.children.assert_not_called()

    def test_stop_child_graceful_escalation_and_terminate_error(self):
        done = Child(code=0); guards.stop_child(done); self.assertEqual(done.calls, [])
        graceful = Child(); guards.stop_child(graceful)
        self.assertEqual(graceful.calls, ["terminate", ("wait", 3)])
        stubborn = Child(stubborn=True); guards.stop_child(stubborn)
        self.assertEqual(stubborn.calls, ["terminate", ("wait", 3), "kill", ("wait", 3)])
        failed_signal = Child(termination_error=True); guards.stop_child(failed_signal)
        self.assertEqual(failed_signal.calls, ["terminate", "kill", ("wait", 3)])

    def test_supervisor_normal_completion_and_nonzero_exit_records(self):
        proc = Child()
        row = {"seconds": 0, "combined_rss_bytes": 3 * GIB, "host_available_bytes": 16 * GIB, "processes_sampled": 2}
        with mock.patch.object(guards, "_tree_sample", return_value=row), \
                mock.patch.object(guards.time, "sleep", side_effect=lambda _: setattr(proc, "returncode", 0)):
            guards.supervise(proc, self.out, 1000, "pair")
        terminal = self.read("terminal.json")
        self.assertEqual(terminal["status"], "complete")
        self.assertEqual(terminal["peak_combined_rss_bytes"], 3 * GIB)
        self.assertFalse((self.out / "watchdog-stop.json").exists())
        directory = self.out / "nonzero"; directory.mkdir()
        guards.supervise(Child(code=2), directory, 1000, "pair")
        self.assertEqual(json.loads((directory / "terminal.json").read_text())["status"], "failed")

    def test_supervisor_caps_deadline_and_interrupt_terminate_child(self):
        cases = (("memory", 1000, "pair"), ("expired", 100, "pair"),
                 ("overlong", 1001, "pair"), ("invalid", float("inf"), "pair"),
                 ("mode", 1000, "unknown"), ("interrupt", 1000, "pair"))
        for label, deadline, mode in cases:
            directory = self.out / label; directory.mkdir()
            proc = Child()
            row = {"seconds": 0, "combined_rss_bytes": 49 * GIB, "host_available_bytes": 16 * GIB, "processes_sampled": 2}
            with mock.patch.object(guards, "_tree_sample", side_effect=KeyboardInterrupt("Injected") if label == "interrupt" else None,
                                   return_value=row), mock.patch.object(guards.time, "sleep"):
                with self.assertRaises(BaseException):
                    guards.supervise(proc, directory, deadline, mode)
            self.assertIn("terminate", proc.calls)
            self.assertEqual(json.loads((directory / "terminal.json").read_text())["status"], "failed")
            self.assertTrue((directory / "watchdog-stop.json").exists())

    def test_terminal_survives_failed_termination_and_kill(self):
        proc = Child(stubborn=True, kill_error=True)
        with self.assertRaises(RuntimeError):
            guards.supervise(proc, self.out, 100, "pair")
        terminal = self.read("terminal.json")
        self.assertEqual(terminal["status"], "failed")
        self.assertEqual(terminal["cleanup_error_type"], "OSError")
        self.assertIsNone(terminal["exit_code"])
        self.assertIn("kill", proc.calls)

    def test_exit_after_deadline_is_failed_even_when_child_returns_zero(self):
        proc = Child()
        row = {"seconds": 0, "combined_rss_bytes": 3 * GIB, "host_available_bytes": 16 * GIB, "processes_sampled": 2}
        clock = mock.Mock(return_value=100.0)
        def exit_late(_):
            proc.returncode = 0
            clock.return_value = 701.0
        with mock.patch.object(guards.time, "monotonic", clock), \
                mock.patch.object(guards, "_tree_sample", return_value=row), \
                mock.patch.object(guards.time, "sleep", side_effect=exit_late):
            with self.assertRaises(RuntimeError):
                guards.supervise(proc, self.out, 700, "codec")
        self.assertEqual(self.read("terminal.json")["status"], "failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise ValueError("Fresh output directory required")
    files = {"guards.py": Path(guards.__file__), "test_guards.py": Path(__file__),
             "frozen_guards.py": Path(frozen.__file__)}
    before = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}
    started = time.monotonic()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    after = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}
    report = {"status": "passed" if result.wasSuccessful() and before == after else "failed",
              "tests": result.testsRun, "elapsed_seconds": time.monotonic() - started,
              "source_sha256": after, "sources_unchanged": before == after,
              "scope": "Mocked memory and worker lifecycle only", "GPU_calls": 0,
              "model_calls": 0, "cloud_calls": 0, "real_child_processes_started": 0}
    if args.output:
        args.output.mkdir(parents=True)
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        checked = args.output / "checked-source"; checked.mkdir()
        for name, path in files.items():
            shutil.copyfile(path, checked / name)
    print(json.dumps(report))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
