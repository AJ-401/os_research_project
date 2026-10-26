"""demo_cap.py - live demo of the Job Object CPU cap on a continuous background load.

Save this file inside the thermal_ctrl folder, then run from the project root:
    python thermal_ctrl/demo_cap.py

It does not modify any existing file and writes no results.
"""
import time
import psutil
from workloads import WindowsJobObject, BackgroundJob


def job_cpu_percent(procs, n_cpus):
    """CPU used by all workers, as a percentage of the WHOLE machine."""
    total = 0.0
    for p in procs:
        try:
            total += p.cpu_percent(interval=None)
        except psutil.Error:
            pass
    return total / n_cpus


def main():
    n_cpus = psutil.cpu_count(logical=True)
    job = WindowsJobObject()
    bg = BackgroundJob(n_workers=n_cpus, job_object=job)
    print(f"Starting {n_cpus} CPU-burning workers inside one Job Object...")
    bg.start(continuous=True)
    procs = [psutil.Process(p.pid) for p in bg.procs]

    try:
        print("Warming up (workers are starting)...")
        for p in procs:
            p.cpu_percent(interval=None)  # prime the counters
        time.sleep(8)

        for cap in (100, 50, 25, 10, 100):
            ok = job.set_cpu_rate_cap(cap)
            print(f"\n>>> CAP = {cap}%   (Windows accepted the call: {ok})")
            job_cpu_percent(procs, n_cpus)  # reset the measurement window
            for _ in range(8):
                time.sleep(1)
                used = job_cpu_percent(procs, n_cpus)
                system = psutil.cpu_percent(interval=None)
                print(f"   workers use {used:5.1f}% of total CPU   | system CPU {system:5.1f}%")
    finally:
        bg.terminate()
        job.close()
        print("\nDone.")


if __name__ == "__main__":
    main()