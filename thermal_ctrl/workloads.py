"""
workloads.py - Workload generators for Thermal-Aware Background Workload Controller.
Provides:
1. Background batch job running inside a Windows Job Object.
2. Foreground probe process measuring real-time task responsiveness (median, p95, p99).
"""

import os
import sys
import time
import math
import hashlib
import ctypes
from ctypes import wintypes
import multiprocessing
import numpy as np
import pandas as pd
import psutil

# Windows Job Object API constants via ctypes
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

JobObjectCpuRateControlInformation = 15
JOB_OBJECT_CPU_RATE_CONTROL_ENABLE = 0x1
JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP = 0x4

PROCESS_SET_QUOTA = 0x0100
PROCESS_TERMINATE = 0x0001
PROCESS_ALL_ACCESS = 0x1FFFFF


class JOBOBJECT_CPU_RATE_CONTROL_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("ControlFlags", wintypes.DWORD),
        ("CpuRate", wintypes.DWORD),
    ]


class WindowsJobObject:
    """Manages a Windows Job Object and CPU Rate Control."""

    def __init__(self, name: str = None):
        self.name = name
        self.handle = kernel32.CreateJobObjectW(None, name)
        if not self.handle:
            err = ctypes.get_last_error()
            raise OSError(f"Failed to create Job Object: Windows error code {err}")

    def assign_process(self, pid: int) -> bool:
        """Assigns a running process to this job object."""
        h_proc = kernel32.OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, False, pid)
        if not h_proc:
            # Fallback to ALL_ACCESS
            h_proc = kernel32.OpenProcess(PROCESS_ALL_ACCESS, False, pid)
        if not h_proc:
            return False
        try:
            ok = kernel32.AssignProcessToJobObject(self.handle, h_proc)
            return bool(ok)
        finally:
            kernel32.CloseHandle(h_proc)

    def set_cpu_rate_cap(self, percent: float) -> bool:
        """Sets a CPU rate hard cap on all processes in this job.
        percent: 1.0 to 100.0 (e.g. 25.0 means 25% max CPU share).
        100.0 removes the cap.
        """
        clamped = max(1.0, min(100.0, float(percent)))
        info = JOBOBJECT_CPU_RATE_CONTROL_INFORMATION()
        info.ControlFlags = JOB_OBJECT_CPU_RATE_CONTROL_ENABLE | JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP
        # CpuRate is expressed in 100ths of a percent: 25% = 2500, 100% = 10000
        info.CpuRate = int(clamped * 100)

        ok = kernel32.SetInformationJobObject(
            self.handle,
            JobObjectCpuRateControlInformation,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        return bool(ok)

    def close(self):
        if self.handle:
            kernel32.CloseHandle(self.handle)
            self.handle = None

    def __del__(self):
        self.close()


def _background_worker(worker_id: int, num_chunks: int, chunk_size: int, stop_event: multiprocessing.Event = None):
    """Worker loop doing fixed computation or continuous burning until stop_event."""
    # Priority if needed
    try:
        p = psutil.Process()
    except Exception:
        pass

    chunks_done = 0
    # In continuous mode, num_chunks is <= 0 and worker stops when stop_event is set
    continuous = (num_chunks <= 0)

    val = 123456789
    while continuous or (chunks_done < num_chunks):
        if stop_event is not None and stop_event.is_set():
            break

        # Fixed chunk computation: integer hashing / arithmetic
        h = hashlib.sha256()
        for i in range(chunk_size):
            val = (val * 1103515245 + 12345) & 0x7FFFFFFF
            if (i & 0x3FF) == 0:
                h.update(val.to_bytes(4, byteorder="little"))
        _ = h.digest()
        chunks_done += 1


class BackgroundJob:
    """Batch background job running N worker processes inside a Windows Job Object."""

    def __init__(
        self,
        n_workers: int,
        total_work_chunks: int = 120,
        chunk_size: int = 40000,
        affinity_cpus: list = None,
        job_object: WindowsJobObject = None,
    ):
        self.n_workers = max(1, n_workers)
        self.total_work_chunks = total_work_chunks
        self.chunk_size = chunk_size
        self.affinity_cpus = affinity_cpus
        self.job_object = job_object or WindowsJobObject()
        self.procs = []
        self.stop_event = multiprocessing.Event()
        self.start_time = None
        self.end_time = None

    def start(self, continuous: bool = False):
        """Starts worker processes and assigns them to the Windows Job Object."""
        self.start_time = time.time()
        chunks_per_worker = -1 if continuous else max(1, self.total_work_chunks // self.n_workers)

        for i in range(self.n_workers):
            p = multiprocessing.Process(
                target=_background_worker,
                args=(i, chunks_per_worker, self.chunk_size, self.stop_event),
                name=f"bg_worker_{i}",
            )
            p.daemon = True
            p.start()
            self.procs.append(p)
            self.job_object.assign_process(p.pid)

            # Apply CPU affinity if specified
            if self.affinity_cpus is not None:
                try:
                    proc = psutil.Process(p.pid)
                    proc.cpu_affinity(self.affinity_cpus)
                except Exception:
                    pass

    def set_priority(self, priority_class):
        """Sets the priority class for all worker processes (e.g. IDLE or BELOW_NORMAL)."""
        for p in self.procs:
            try:
                proc = psutil.Process(p.pid)
                proc.nice(priority_class)
            except Exception:
                pass

    def wait(self, timeout: float = None) -> float:
        """Waits for all workers to finish. Returns duration in seconds."""
        for p in self.procs:
            p.join(timeout=timeout)
        self.end_time = time.time()
        return self.end_time - self.start_time

    def is_alive(self) -> bool:
        return any(p.is_alive() for p in self.procs)

    def terminate(self):
        """Signals stop and terminates worker processes."""
        self.stop_event.set()
        for p in self.procs:
            if p.is_alive():
                p.terminate()
        for p in self.procs:
            p.join(timeout=1.0)
        self.end_time = time.time()


def _probe_worker_proc(
    pinned_cpu: int,
    array_size: int,
    ready_event: multiprocessing.Event,
    stop_event: multiprocessing.Event,
    pipe_conn,
    csv_path: str = None,
):
    """Worker process for Foreground Probe."""
    # Set priority ABOVE_NORMAL and pin CPU
    try:
        current_proc = psutil.Process()
        current_proc.nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
        if pinned_cpu is not None:
            current_proc.cpu_affinity([pinned_cpu])
    except Exception as e:
        print(f"[Probe] Warning: Could not set priority/affinity: {e}")

    # Fixed input data
    arr = np.linspace(0.0, 1000.0, array_size, dtype=np.float64)

    # Warmup
    _ = np.sin(arr) + np.cos(arr)

    records = []
    stats = {"count": 0, "median_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}

    try:
        # Signal ready to parent
        ready_event.set()

        # Loop until stopped
        while not stop_event.is_set():
            t0 = time.perf_counter()
            t_wall = time.time()

            # Fixed task (~5 ms)
            _ = np.sin(arr) + np.cos(arr)

            duration_ms = (time.perf_counter() - t0) * 1000.0
            records.append((t_wall, duration_ms))

            # Sleep 5 ms
            time.sleep(0.005)

        # Save to CSV if requested
        if csv_path and records:
            os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
            df = pd.DataFrame(records, columns=["timestamp", "task_time_ms"])
            df.to_csv(csv_path, index=False)

        # Compute summary stats
        durations = [r[1] for r in records]
        if durations:
            arr_dur = np.array(durations)
            stats = {
                "count": len(arr_dur),
                "median_ms": round(float(np.median(arr_dur)), 3),
                "p95_ms": round(float(np.percentile(arr_dur, 95)), 3),
                "p99_ms": round(float(np.percentile(arr_dur, 99)), 3),
                "min_ms": round(float(np.min(arr_dur)), 3),
                "max_ms": round(float(np.max(arr_dur)), 3),
            }
    except Exception as e:
        print(f"[Probe] Exception: {e}")
    finally:
        try:
            if pipe_conn:
                pipe_conn.send(stats)
                pipe_conn.close()
        except Exception:
            pass


class ForegroundProbe:
    """Foreground probe process measuring responsiveness at ~5ms work intervals."""

    def __init__(self, pinned_cpu: int = 0, target_work_ms: float = 5.0):
        self.pinned_cpu = pinned_cpu
        self.target_work_ms = target_work_ms
        self.array_size = 200000
        self.ready_event = multiprocessing.Event()
        self.stop_event = multiprocessing.Event()
        self.parent_conn = None
        self.child_conn = None
        self.proc = None
        self.stats = None
        self.csv_path = None
        self._calibrate_task_size()

    def _calibrate_task_size(self):
        """Calibrates array size to reach approximately 5 ms of compute."""
        test_size = 200000
        test_arr = np.linspace(0.0, 1000.0, test_size, dtype=np.float64)
        # Warmup and time 5 evaluations
        _ = np.sin(test_arr) + np.cos(test_arr)
        t0 = time.perf_counter()
        for _ in range(5):
            _ = np.sin(test_arr) + np.cos(test_arr)
        dt_single_ms = ((time.perf_counter() - t0) / 5.0) * 1000.0
        if dt_single_ms > 0:
            scale = self.target_work_ms / dt_single_ms
            self.array_size = max(10000, int(test_size * scale))
        else:
            self.array_size = 200000

    def start(self, csv_path: str = None):
        """Starts the probe process in background."""
        self.ready_event.clear()
        self.stop_event.clear()
        self.csv_path = csv_path
        self.parent_conn, self.child_conn = multiprocessing.Pipe(duplex=False)
        self.proc = multiprocessing.Process(
            target=_probe_worker_proc,
            args=(
                self.pinned_cpu,
                self.array_size,
                self.ready_event,
                self.stop_event,
                self.child_conn,
                csv_path,
            ),
            name="fg_probe",
        )
        self.proc.daemon = True
        self.proc.start()
        # Wait until child has actually started and is running its loop
        self.ready_event.wait(timeout=10.0)

    def stop(self, timeout: float = 5.0) -> dict:
        """Stops the probe process and returns summary metrics."""
        if not self.proc:
            return {}
        self.stop_event.set()
        # Drain pipe before join to avoid deadlock
        stats = {}
        if self.parent_conn:
            try:
                if self.parent_conn.poll(timeout=3.0):
                    stats = self.parent_conn.recv()
                self.parent_conn.close()
            except Exception:
                pass
        self.proc.join(timeout=timeout)

        # Fallback to reading CSV if pipe did not return stats
        if (not stats or stats.get("count", 0) == 0) and self.csv_path and os.path.exists(self.csv_path):
            try:
                df = pd.read_csv(self.csv_path)
                if len(df) > 0:
                    arr = df["task_time_ms"].values
                    stats = {
                        "count": len(arr),
                        "median_ms": round(float(np.median(arr)), 3),
                        "p95_ms": round(float(np.percentile(arr, 95)), 3),
                        "p99_ms": round(float(np.percentile(arr, 99)), 3),
                        "min_ms": round(float(np.min(arr)), 3),
                        "max_ms": round(float(np.max(arr)), 3),
                    }
            except Exception:
                pass

        self.stats = stats
        return self.stats


def test_workloads():
    """Quick validation of workloads."""
    print("Testing ForegroundProbe calibration and run...")
    probe = ForegroundProbe(pinned_cpu=0)
    print(f"Calibrated probe task: array_size={probe.array_size} (~5 ms)")
    probe.start(csv_path="results/test_probe.csv")
    time.sleep(2.0)
    stats = probe.stop()
    print(f"Probe stats (2s test): median={stats.get('median_ms', 0):.2f} ms, p95={stats.get('p95_ms', 0):.2f} ms, p99={stats.get('p99_ms', 0):.2f} ms")

    print("\nTesting WindowsJobObject CPU Cap...")
    job = WindowsJobObject()
    bg = BackgroundJob(n_workers=2, total_work_chunks=4, chunk_size=20000, job_object=job)
    bg.start()
    ok_cap = job.set_cpu_rate_cap(25.0)
    print(f"Applied 25% CPU rate cap to Job Object: {ok_cap}")
    bg.wait(timeout=10.0)
    print("Background job finished successfully.")
    job.close()


if __name__ == "__main__":
    test_workloads()
