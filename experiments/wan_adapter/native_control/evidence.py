# SPDX-License-Identifier: Apache-2.0
"""Bounded evidence collection for the single native-style control clip."""
import json
import os
from pathlib import Path
import threading
import time
import psutil
import torch

GIB = 1024**3


def write_json(path, value):
    temporary = Path(path).with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


class Evidence:
    def __init__(self, directory):
        self.path = Path(directory)
        if self.path.exists() or self.path.is_symlink() or self.path.resolve().is_relative_to(Path(__file__).parent.resolve()):
            raise ValueError('Output must be a new directory outside native_control')
        self.path.mkdir(parents=True)
        torch.mps.set_per_process_memory_fraction(min(1., 18*GIB/torch.mps.recommended_max_memory()))
        self.started = time.perf_counter()
        self.stage = 'validate'
        self.stop = threading.Event()
        self.peaks = {'rss_bytes': 0, 'mps_active_bytes': 0, 'mps_driver_bytes': 0}
        self.report = {'status': 'running', 'max_seconds': 900, 'max_memory_gib': 18,
                       'minimum_available_gib': 2, 'timings': [], 'steps': []}
        self.save()
        def monitor():
            process = psutil.Process()
            with (self.path/'memory.jsonl').open('x') as log:
                while not self.stop.is_set():
                    row = {'seconds': time.perf_counter()-self.started, 'stage': self.stage,
                           'rss_bytes': process.memory_info().rss, 'available_system_bytes': psutil.virtual_memory().available,
                           'mps_active_bytes': torch.mps.current_allocated_memory(), 'mps_driver_bytes': torch.mps.driver_allocated_memory()}
                    for key in self.peaks:
                        self.peaks[key] = max(self.peaks[key], row[key])
                    log.write(json.dumps(row)+'\n'); log.flush()
                    reason = None
                    if row['seconds'] > 900: reason = 'time-cap-900-seconds'
                    elif max(row['rss_bytes'], row['mps_driver_bytes']) > 18*GIB: reason = 'memory-cap-18-GiB'
                    elif row['available_system_bytes'] < 2*GIB: reason = 'available-system-memory-below-2-GiB'
                    if reason:
                        write_json(self.path/'watchdog-stop.json', {'status': 'stopped', 'reason': reason, 'last_sample': row})
                        os._exit(124)
                    self.stop.wait(.5)
        self.thread = threading.Thread(target=monitor, daemon=True)
        self.thread.start()

    def save(self):
        self.report['peaks_sampled'] = dict(self.peaks)
        write_json(self.path/'metrics.json', self.report)

    def measure(self, label, function):
        self.stage = label
        torch.mps.synchronize()
        start = time.perf_counter()
        result = function()
        torch.mps.synchronize()
        self.report['timings'].append({'stage': label, 'seconds': time.perf_counter()-start})
        self.save()
        return result

    def close(self, error=None):
        self.stop.set(); self.thread.join(timeout=2)
        self.report['status'] = 'failed' if error else 'passed'
        if error:
            self.report.update(error_type=type(error).__name__, error=str(error))
        self.report['elapsed_seconds'] = time.perf_counter()-self.started
        self.save()
