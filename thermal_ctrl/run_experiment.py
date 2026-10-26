"""
run_experiment.py - Executes thermal and workload experiments.
Supports:
1. Multi-mode comparison (M0, M1, M2, M3) with randomized order per repeat.
2. Phase-1 experiment isolating core contention vs shared package thermal/power throttling.
Outputs:
- results/run_<mode>_<repeat>.csv (telemetry)
- results/run_<mode>_<repeat>_probe.csv (foreground probe times)
- results/run_<mode>_<repeat>_ctrl.csv (controller decisions)
- results/summary.csv
- results/phase1_raw.csv
"""

import os
import sys
import time
import json
import random
import argparse
import psutil
import pandas as pd
import numpy as np

# Add local path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from telemetry import TelemetryReader, get_active_power_plan
from workloads import WindowsJobObject, BackgroundJob, ForegroundProbe
from controller import WorkloadController


def load_config(config_path: str = "config.json") -> dict:
    """Loads configuration created by calibrate.py, or falls back to sane defaults."""
    # Check current dir or thermal_ctrl
    candidates = [config_path, os.path.join("thermal_ctrl", config_path)]
    for p in candidates:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    print(f"Warning: {config_path} not found. Using default calibration settings.")
    return {
        "heavy_workers": psutil.cpu_count(logical=True) or 8,
        "T_warn": 70.0,
        "T_hot": 72.0,
        "cooldown_target": 67.8,
        "baseline_temp_c": 62.8,
        "baseline_freq_mhz": 1866.0,
    }


def wait_for_cooldown(telemetry: TelemetryReader, cooldown_target: float, max_wait_s: int = 120):
    """Waits until temperature drops to or below cooldown_target (up to max_wait_s)."""
    print(f"[Cooldown] Waiting for temp <= {cooldown_target:.1f} C (max {max_wait_s}s)...", end="", flush=True)
    t_start = time.time()
    while (time.time() - t_start) < max_wait_s:
        data = telemetry.read()
        cur_temp = data.get("temp_c", 0.0)
        if cur_temp <= cooldown_target:
            print(f" Ready ({cur_temp:.1f} C).")
            return
        time.sleep(2.0)
        print(".", end="", flush=True)
    last_temp = telemetry.read().get("temp_c", 0.0)
    print(f" [Warning] Cooldown timed out at {last_temp:.1f} C. Continuing anyway.")


def run_single_mode(
    mode: str,
    repeat: int,
    config: dict,
    total_work_chunks: int = 60,
    chunk_size: int = 40000,
    post_probe_s: int = 5,
    cooldown_max_s: int = 120,
    results_dir: str = "results",
) -> dict:
    """Executes one full experiment run for a given mode and repeat index."""
    os.makedirs(results_dir, exist_ok=True)
    run_id = f"{mode}_{repeat}"
    telemetry_csv = os.path.join(results_dir, f"run_{run_id}.csv")
    probe_csv = os.path.join(results_dir, f"run_{run_id}_probe.csv")
    ctrl_csv = os.path.join(results_dir, f"run_{run_id}_ctrl.csv")

    heavy_workers = config.get("heavy_workers", psutil.cpu_count(logical=True) or 8)
    cooldown_target = config.get("cooldown_target", 67.8)
    T_hot = config.get("T_hot", 72.0)

    print(f"\n============================================================")
    print(f"RUNNING EXPERIMENT: Mode {mode} (Repeat {repeat})")
    print(f"Workers: {heavy_workers} | Work Chunks: {total_work_chunks}")
    print(f"============================================================")

    telemetry = TelemetryReader(interval_s=1.0)

    # 1. Wait for cooldown
    wait_for_cooldown(telemetry, cooldown_target, max_wait_s=cooldown_max_s)

    # 2. Setup Job Object, Background Job, Foreground Probe, and Controller
    job_obj = WindowsJobObject()
    bg_job = BackgroundJob(
        n_workers=heavy_workers,
        total_work_chunks=total_work_chunks,
        chunk_size=chunk_size,
        job_object=job_obj,
    )
    probe = ForegroundProbe(pinned_cpu=0)
    controller = WorkloadController(
        mode=mode,
        job_object=job_obj,
        background_job=bg_job,
        config=config,
        log_csv=ctrl_csv,
    )

    telemetry_records = []

    try:
        # Start probe and controller
        probe.start(csv_path=probe_csv)
        controller.on_start()

        # Start background job
        t_bg_start = time.time()
        bg_job.start(continuous=False)

        # Main execution loop: sample telemetry and step controller every 1s
        print(f"Background job active. Streaming telemetry & controller decisions...")
        bg_finished = False
        t_bg_end = None

        while True:
            t0 = time.time()
            data = telemetry.read()
            telemetry_records.append(data)

            # Controller step
            ctrl_dec = controller.step(data)

            # Check if background job finished
            if not bg_finished and not bg_job.is_alive():
                t_bg_end = time.time()
                bg_finished = True
                print(f"\n-> Background job completed in {t_bg_end - t_bg_start:.2f} seconds.")
                print(f"-> Recording {post_probe_s}s post-work cool-down probe data...")
                time.sleep(max(0.0, 1.0 - (time.time() - t0)))
                break

            time.sleep(max(0.0, 1.0 - (time.time() - t0)))

        # Continue probe for post_probe_s cool-down data
        t_post_start = time.time()
        while (time.time() - t_post_start) < post_probe_s:
            t0 = time.time()
            data = telemetry.read()
            telemetry_records.append(data)
            controller.step(data)
            time.sleep(max(0.0, 1.0 - (time.time() - t0)))

    finally:
        bg_job.terminate()
        job_obj.close()
        probe_stats = probe.stop()
        telemetry.close()
        controller.close()

    bg_time_s = round(t_bg_end - t_bg_start, 2) if t_bg_end else 0.0

    # Save telemetry CSV
    df_telem = pd.DataFrame(telemetry_records)
    df_telem.to_csv(telemetry_csv, index=False)

    # Compute summary metrics
    max_temp = float(df_telem["temp_c"].max()) if len(df_telem) else 0.0
    mean_freq = float(df_telem["freq_mhz"].mean()) if len(df_telem) else 0.0
    time_above_T_hot = int((df_telem["temp_c"] >= T_hot).sum()) if len(df_telem) else 0

    ctrl_cpus = [r["controller_cpu_percent"] for r in controller.log_records]
    ctrl_cpu_pct = float(np.mean(ctrl_cpus)) if ctrl_cpus else 0.0

    summary_entry = {
        "mode": mode,
        "repeat": repeat,
        "background_time_s": bg_time_s,
        "probe_median_ms": probe_stats.get("median_ms", 0.0),
        "probe_p95_ms": probe_stats.get("p95_ms", 0.0),
        "probe_p99_ms": probe_stats.get("p99_ms", 0.0),
        "max_temp_c": round(max_temp, 2),
        "mean_freq_mhz": round(mean_freq, 1),
        "time_above_T_hot_s": time_above_T_hot,
        "controller_cpu_percent": round(ctrl_cpu_pct, 2),
    }

    print("\nRun Summary:")
    print(f"  Bg Duration: {bg_time_s:.2f} s")
    print(f"  Probe p95:   {summary_entry['probe_p95_ms']:.2f} ms (p99: {summary_entry['probe_p99_ms']:.2f} ms)")
    print(f"  Max Temp:    {max_temp:.2f} C (Time >= T_hot: {time_above_T_hot}s)")
    print(f"  Mean Freq:   {mean_freq:.1f} MHz")
    print(f"  Ctrl CPU:    {ctrl_cpu_pct:.2f}%")

    return summary_entry


def append_summary(entry: dict, summary_path: str = "results/summary.csv"):
    """Appends summary row to summary.csv."""
    os.makedirs(os.path.dirname(os.path.abspath(summary_path)) or ".", exist_ok=True)
    df_row = pd.DataFrame([entry])
    if not os.path.exists(summary_path):
        df_row.to_csv(summary_path, index=False)
    else:
        df_row.to_csv(summary_path, mode="a", header=False, index=False)


def run_phase1_experiment(
    repeats: int = 3,
    duration_s: int = 15,
    results_dir: str = "results",
):
    """Executes Phase-1 experiment:
    A: Foreground alone.
    B: Background on OTHER physical cores than foreground.
    C: Background on SAME core as foreground.
    D: Background on SAME core at low priority.
    """
    os.makedirs(results_dir, exist_ok=True)
    logical_cpus = psutil.cpu_count(logical=True) or 4
    physical_cpus = psutil.cpu_count(logical=False) or 2

    # Map core topology: Assume core 0 is physical CPU 0 (pair 0, 1)
    fg_cpu = 0
    # Other physical cores
    other_cpus = list(range(2, logical_cpus)) if logical_cpus > 2 else [1]
    same_cpus = [0, 1] if logical_cpus > 1 else [0]

    print("============================================================")
    print("PHASE-1 EXPERIMENT: CONTENTION VS THERMAL THROTTLING")
    print(f"Logical: {logical_cpus}, Physical: {physical_cpus}")
    print(f"Foreground CPU: {fg_cpu}")
    print(f"Other physical CPUs: {other_cpus[:6]}...")
    print(f"Same CPU pair: {same_cpus}")
    print("============================================================")

    cases = ["A", "B", "C", "D"]
    records = []

    for rep in range(1, repeats + 1):
        print(f"\n>>> Phase-1 Repeat {rep}/{repeats} <<<")
        for case in cases:
            print(f"\n--- Case {case} (Repeat {rep}) ---")
            telemetry = TelemetryReader(interval_s=1.0)
            time.sleep(2.0) # short settle

            probe_csv = os.path.join(results_dir, f"phase1_{case}_{rep}_probe.csv")
            probe = ForegroundProbe(pinned_cpu=fg_cpu)
            probe.start(csv_path=probe_csv)

            bg_job = None
            job_obj = None

            if case == "A":
                # Foreground alone
                time.sleep(duration_s)

            elif case == "B":
                # Background pinned to other physical cores
                job_obj = WindowsJobObject()
                bg_job = BackgroundJob(n_workers=min(4, len(other_cpus)), affinity_cpus=other_cpus, job_object=job_obj)
                bg_job.start(continuous=True)
                time.sleep(duration_s)

            elif case == "C":
                # Background on same core as foreground
                job_obj = WindowsJobObject()
                bg_job = BackgroundJob(n_workers=len(same_cpus), affinity_cpus=same_cpus, job_object=job_obj)
                bg_job.start(continuous=True)
                time.sleep(duration_s)

            elif case == "D":
                # Background on same core at low priority
                job_obj = WindowsJobObject()
                bg_job = BackgroundJob(n_workers=len(same_cpus), affinity_cpus=same_cpus, job_object=job_obj)
                bg_job.start(continuous=True)
                bg_job.set_priority(psutil.BELOW_NORMAL_PRIORITY_CLASS)
                time.sleep(duration_s)

            if bg_job:
                bg_job.terminate()
            if job_obj:
                job_obj.close()

            stats = probe.stop()
            telem_data = telemetry.read()
            telemetry.close()

            rec = {
                "case": case,
                "repeat": rep,
                "probe_median_ms": stats.get("median_ms", 0.0),
                "probe_p95_ms": stats.get("p95_ms", 0.0),
                "probe_p99_ms": stats.get("p99_ms", 0.0),
                "temp_c": telem_data.get("temp_c", 0.0),
                "freq_mhz": telem_data.get("freq_mhz", 0.0),
            }
            records.append(rec)
            print(f"Case {case} done: probe p95 = {rec['probe_p95_ms']:.2f} ms, freq = {rec['freq_mhz']:.0f} MHz")

    # Save Phase 1 raw
    df_p1 = pd.DataFrame(records)
    phase1_csv = os.path.join(results_dir, "phase1_raw.csv")
    df_p1.to_csv(phase1_csv, index=False)
    print(f"\nPhase-1 experiment completed. Saved to {phase1_csv}")
    return df_p1


def main():
    parser = argparse.ArgumentParser(description="Thermal-Aware Workload Experiment Runner.")
    parser.add_argument("--modes", nargs="+", default=["M0", "M1", "M2", "M3"], help="Modes to run (e.g. M0 M1 M2 M3)")
    parser.add_argument("--repeats", type=int, default=1, help="Number of repeats per mode (default: 1)")
    parser.add_argument("--phase1", action="store_true", help="Run Phase-1 contention vs throttling experiment")
    parser.add_argument("--work-chunks", type=int, default=60, help="Work chunks for background job")
    parser.add_argument("--post-probe-s", type=int, default=5, help="Post-job cooldown probe duration")
    parser.add_argument("--cooldown-max-s", type=int, default=60, help="Max wait for cooldown target in seconds")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config.json")
    args = parser.parse_args()

    if args.phase1:
        run_phase1_experiment(repeats=args.repeats, duration_s=15)
        return

    config = load_config(args.config)
    modes = args.modes
    repeats = args.repeats

    print(f"Beginning Experiment Campaign: Modes={modes}, Repeats={repeats}")

    for r in range(1, repeats + 1):
        # Shuffle mode order each repeat to eliminate systematic warm-up bias
        shuffled = list(modes)
        random.shuffle(shuffled)
        print(f"\n--- Repeat {r}/{repeats} execution order: {shuffled} ---")

        for m in shuffled:
            entry = run_single_mode(
                mode=m,
                repeat=r,
                config=config,
                total_work_chunks=args.work_chunks,
                post_probe_s=args.post_probe_s,
                cooldown_max_s=args.cooldown_max_s,
            )
            append_summary(entry, "results/summary.csv")

    print("\nExperiment campaign completed successfully! Results recorded in results/summary.csv.")


if __name__ == "__main__":
    main()
