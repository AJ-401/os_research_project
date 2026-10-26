"""
calibrate.py - Thermal and workload calibration for Windows.
Finds laptop's "heavy" workload level, throttle onset, and temperature thresholds (T_warn, T_hot, cooldown_target).
Outputs:
- config.json
- results/calibration.png
"""

import os
import sys
import time
import json
import argparse
import psutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure thermal_ctrl path is accessible
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from telemetry import TelemetryReader, get_active_power_plan
from workloads import BackgroundJob, WindowsJobObject


def run_calibration(
    idle_duration_s: int = 30,
    step_duration_s: int = 20,
    max_workers: int = None,
    output_config: str = "config.json",
    output_plot: str = "results/calibration.png",
    output_csv: str = "results/calibration_data.csv",
):
    """Executes the calibration procedure."""
    logical_cpus = psutil.cpu_count(logical=True) or 4
    physical_cpus = psutil.cpu_count(logical=False) or 2
    if max_workers is None:
        max_workers = min(logical_cpus, 16) # default up to 16 or logical_cpus

    print("============================================================")
    print("STARTING HARDWARE THERMAL CALIBRATION")
    print("============================================================")
    power_plan = get_active_power_plan()
    print(f"Active Power Plan: {power_plan}")
    print(f"CPUs: {logical_cpus} logical, {physical_cpus} physical")
    print(f"Idle duration: {idle_duration_s}s | Step duration: {step_duration_s}s | Max workers: {max_workers}")
    print("------------------------------------------------------------")

    telemetry = TelemetryReader(interval_s=1.0)
    all_readings = []
    step_markers = [] # (time_sec, label, worker_count)

    t_start = time.time()

    # Step 1: Idle phase
    print(f"\n[Phase 1] Recording baseline idle for {idle_duration_s} seconds...")
    step_markers.append((0.0, "Idle (0 workers)", 0))
    for s in range(idle_duration_s):
        time.sleep(1.0)
        data = telemetry.read()
        elapsed = round(time.time() - t_start, 1)
        data["elapsed_s"] = elapsed
        data["workers"] = 0
        data["phase"] = "idle"
        all_readings.append(data)
        if (s + 1) % 10 == 0 or (s + 1) == idle_duration_s:
            print(f"  Idle {s+1:2d}/{idle_duration_s}s: Temp = {data['temp_c']:.1f} C, Freq = {data['freq_mhz']:.0f} MHz")

    idle_df = pd.DataFrame([r for r in all_readings if r["phase"] == "idle"])
    baseline_temp = float(idle_df["temp_c"].mean())
    baseline_freq = float(idle_df["freq_mhz"].mean())
    baseline_perf = float(idle_df["perf_percent"].mean()) if "perf_percent" in idle_df else 100.0
    print(f"\n-> Baseline established: Temp = {baseline_temp:.2f} C, Freq = {baseline_freq:.1f} MHz, Perf = {baseline_perf:.1f}%")

    # Step 2: Ramp stress phase
    print(f"\n[Phase 2] Ramp stress: Stepping workers 1 to {max_workers} ({step_duration_s}s each)...")
    
    current_job = None
    throttle_onset_worker = None
    throttle_onset_temp = None
    throttle_onset_freq = None
    throttle_onset_perf = None
    peak_early_freq = baseline_freq
    peak_early_perf = baseline_perf
    throttles_detected = False
    throttle_type = "none"  # "power", "thermal", "both", or "none"

    try:
        for w in range(1, max_workers + 1):
            elapsed_now = round(time.time() - t_start, 1)
            step_markers.append((elapsed_now, f"{w} workers", w))
            print(f"\n--- Worker Step {w}/{max_workers} ({step_duration_s}s) ---")

            # Clean previous workers and start w workers in continuous burner mode
            if current_job:
                current_job.terminate()
                time.sleep(0.5)

            current_job = BackgroundJob(n_workers=w)
            current_job.start(continuous=True)

            step_readings = []
            for s in range(step_duration_s):
                time.sleep(1.0)
                data = telemetry.read()
                elapsed = round(time.time() - t_start, 1)
                data["elapsed_s"] = elapsed
                data["workers"] = w
                data["phase"] = f"ramp_{w}"
                all_readings.append(data)
                step_readings.append(data)

            step_df = pd.DataFrame(step_readings)
            step_mean_freq = float(step_df["freq_mhz"].mean())
            step_mean_temp = float(step_df["temp_c"].mean())
            step_mean_perf = float(step_df["perf_percent"].mean()) if "perf_percent" in step_df else 100.0
            step_limit_flag = int(step_df["perf_limit_flag"].max())

            # Track early peak frequency & delivered performance (from 1-2 workers)
            if w <= 2:
                peak_early_freq = max(peak_early_freq, step_mean_freq)
                peak_early_perf = max(peak_early_perf, step_mean_perf)

            # Check throttle conditions:
            # 1. Frequency drop > 10%
            freq_drop_pct = ((peak_early_freq - step_mean_freq) / peak_early_freq) * 100.0 if peak_early_freq > 0 else 0
            # 2. Performance % drop > 8% from early peak under load
            perf_drop_pct = ((peak_early_perf - step_mean_perf) / peak_early_perf) * 100.0 if peak_early_perf > 0 else 0
            # 3. Limit flag
            flag_tripped = (step_limit_flag > 0)

            # Distinguish power vs thermal throttle:
            # Thermal throttle: temp elevated (>= 75C or >= baseline + 15C) with freq/perf drop
            # Power throttle: perf/freq drop while temp is still moderate (< 75C)
            step_power_throttled = (perf_drop_pct >= 8.0) or flag_tripped
            step_thermal_throttled = (freq_drop_pct >= 10.0 and step_mean_temp >= 75.0) or (step_mean_temp >= 85.0)
            is_throttled = step_power_throttled or step_thermal_throttled

            print(f"Step {w} Summary: Temp = {step_mean_temp:.1f} C, Freq = {step_mean_freq:.0f} MHz (Drop: {freq_drop_pct:+.1f}%), "
                  f"Perf = {step_mean_perf:.1f}% (Drop: {perf_drop_pct:+.1f}%), LimitFlag = {step_limit_flag}")

            if is_throttled and throttle_onset_worker is None:
                throttle_onset_worker = w
                throttle_onset_temp = step_mean_temp
                throttle_onset_freq = step_mean_freq
                throttle_onset_perf = step_mean_perf
                throttles_detected = True

                if step_power_throttled and step_thermal_throttled:
                    throttle_type = "both"
                elif step_power_throttled:
                    throttle_type = "power"
                else:
                    throttle_type = "thermal"

                print(f"==> THROTTLE ONSET DETECTED at {w} workers! Type: [{throttle_type.upper()}] "
                      f"(Perf Drop = {perf_drop_pct:.1f}%, Freq Drop = {freq_drop_pct:.1f}%, Temp = {step_mean_temp:.1f} C)")

            # If we've confirmed throttling for 3 consecutive steps past onset, conclude ramp
            if throttles_detected and (w >= throttle_onset_worker + 3):
                print(f"\nThrottle behavior confirmed over {w} workers. Concluding ramp early to protect thermal budget.")
                break

    finally:
        if current_job:
            current_job.terminate()
            current_job = None
        telemetry.close()

    # Step 3: Compute thresholds
    max_observed_temp = float(max(r["temp_c"] for r in all_readings))
    max_observed_perf = float(max(r.get("perf_percent", 100.0) for r in all_readings))
    min_observed_perf = float(min(r.get("perf_percent", 100.0) for r in all_readings))
    
    if throttles_detected and throttle_onset_worker is not None:
        heavy_workers = throttle_onset_worker
        # Set thermal thresholds safely
        T_warn = round(max(baseline_temp + 3.0, throttle_onset_temp - 5.0), 1)
        T_hot = round(max(T_warn + 2.0, throttle_onset_temp - 2.0), 1)
        perf_warn_threshold = round(peak_early_perf * 0.95, 1)
        perf_hot_threshold = round(peak_early_perf * 0.90, 1)
    else:
        # Never throttled: use all logical CPUs
        heavy_workers = logical_cpus
        throttles_detected = False
        throttle_onset_worker = heavy_workers
        throttle_onset_temp = max_observed_temp
        throttle_onset_freq = baseline_freq
        throttle_onset_perf = peak_early_perf
        T_warn = round(max(baseline_temp + 3.0, max_observed_temp - 3.0), 1)
        T_hot = round(max(T_warn + 2.0, max_observed_temp - 1.0), 1)
        perf_warn_threshold = round(peak_early_perf * 0.95, 1)
        perf_hot_threshold = round(peak_early_perf * 0.90, 1)

    cooldown_target = round(baseline_temp + 5.0, 1)

    # Invariant: Ensure T_hot > T_warn
    if T_hot <= T_warn:
        T_hot = round(T_warn + 2.0, 1)

    print("\n------------------------------------------------------------")
    print("CALIBRATION RESULTS & THRESHOLDS")
    print("------------------------------------------------------------")
    print(f"Throttles Detected: {throttles_detected} (Type: {throttle_type.upper()})")
    print(f"Throttle Onset Worker Count: {throttle_onset_worker}")
    print(f"Heavy Workers Count: {heavy_workers}")
    print(f"Baseline: Temp = {baseline_temp:.2f} C, Freq = {baseline_freq:.1f} MHz, Perf = {baseline_perf:.1f}%")
    print(f"Peak Early: Freq = {peak_early_freq:.1f} MHz, Perf = {peak_early_perf:.1f}%")
    print(f"Onset Readings: Temp = {throttle_onset_temp:.2f} C, Freq = {throttle_onset_freq:.1f} MHz, Perf = {throttle_onset_perf:.1f}%")
    print(f"Thermal Thresholds: T_warn = {T_warn:.1f} C, T_hot = {T_hot:.1f} C, Cooldown = {cooldown_target:.1f} C")
    print(f"Power/Perf Thresholds: Warn = {perf_warn_threshold:.1f}%, Hot/Cap = {perf_hot_threshold:.1f}%")
    print("------------------------------------------------------------")

    # Save to config.json
    config_dict = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "power_plan": power_plan,
        "cpu_logical_cores": logical_cpus,
        "cpu_physical_cores": physical_cpus,
        "throttles": throttles_detected,
        "throttle_type": throttle_type,
        "baseline_temp_c": round(baseline_temp, 2),
        "baseline_freq_mhz": round(baseline_freq, 1),
        "baseline_perf_percent": round(baseline_perf, 1),
        "peak_early_freq_mhz": round(peak_early_freq, 1),
        "peak_early_perf_percent": round(peak_early_perf, 1),
        "throttle_onset_worker": int(throttle_onset_worker),
        "heavy_workers": int(heavy_workers),
        "throttle_onset_temp_c": round(throttle_onset_temp, 2),
        "throttle_onset_freq_mhz": round(throttle_onset_freq, 1),
        "throttle_onset_perf_percent": round(throttle_onset_perf, 1),
        "T_warn": float(T_warn),
        "T_hot": float(T_hot),
        "cooldown_target": float(cooldown_target),
        "perf_warn_threshold_percent": float(perf_warn_threshold),
        "perf_hot_threshold_percent": float(perf_hot_threshold),
        "calibration_run": {
            "idle_duration_s": idle_duration_s,
            "step_duration_s": step_duration_s,
            "total_points": len(all_readings),
        },
    }

    # Write config.json
    os.makedirs(os.path.dirname(os.path.abspath(output_config)) or ".", exist_ok=True)
    with open(output_config, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    # Also mirror config.json in root if in subfolder
    alt_configs = ["config.json", os.path.join("thermal_ctrl", "config.json")]
    for p in alt_configs:
        try:
            if os.path.abspath(p) != os.path.abspath(output_config):
                os.makedirs(os.path.dirname(os.path.abspath(p)) or ".", exist_ok=True)
                with open(p, "w", encoding="utf-8") as f:
                    json.dump(config_dict, f, indent=2)
        except Exception:
            pass

    # Save CSV data
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)) or ".", exist_ok=True)
    df_all = pd.DataFrame(all_readings)
    df_all.to_csv(output_csv, index=False)

    # Step 4: Generate results/calibration.png
    os.makedirs(os.path.dirname(os.path.abspath(output_plot)) or ".", exist_ok=True)
    _generate_calibration_plot(df_all, step_markers, config_dict, output_plot)
    print(f"\nSaved calibration plot to: {output_plot}")
    print(f"Saved configuration to: {output_config}")
    print("============================================================\n")

    return config_dict


def _generate_calibration_plot(df: pd.DataFrame, step_markers: list, cfg: dict, output_path: str):
    """Generates a 3-panel calibration visualization graph (Temp, Frequency, % Processor Performance)."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax_temp, ax_freq, ax_perf) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    times = df["elapsed_s"].values
    temps = df["temp_c"].values
    freqs = df["freq_mhz"].values
    perfs = df["perf_percent"].values if "perf_percent" in df else np.ones_like(times) * 100.0

    # 1. Temperature subplot
    ax_temp.plot(times, temps, color="#e63946", linewidth=2.0, label="Temperature (C)")
    ax_temp.axhline(cfg["T_hot"], color="#d62828", linestyle="--", linewidth=1.5, label=f"T_hot ({cfg['T_hot']:.1f} C)")
    ax_temp.axhline(cfg["T_warn"], color="#f77f00", linestyle="--", linewidth=1.5, label=f"T_warn ({cfg['T_warn']:.1f} C)")
    ax_temp.axhline(cfg["cooldown_target"], color="#457b9d", linestyle=":", linewidth=1.5, label=f"Cooldown ({cfg['cooldown_target']:.1f} C)")
    ax_temp.set_ylabel("Temp (°C)", fontsize=11, fontweight="bold")
    ax_temp.set_title(f"Thermal & Power Calibration (Throttle Type: {cfg.get('throttle_type', 'none').upper()})", fontsize=13, fontweight="bold")
    ax_temp.legend(loc="upper left", framealpha=0.9)

    # 2. Frequency subplot
    ax_freq.plot(times, freqs, color="#1d3557", linewidth=2.0, label="CPU Frequency (MHz)")
    ax_freq.axhline(cfg["peak_early_freq_mhz"], color="#2a9d8f", linestyle="--", label=f"Peak Early ({cfg['peak_early_freq_mhz']:.0f} MHz)")
    ax_freq.set_ylabel("Frequency (MHz)", fontsize=11, fontweight="bold")
    ax_freq.legend(loc="upper left", framealpha=0.9)

    # 3. % Processor Performance subplot
    ax_perf.plot(times, perfs, color="#6a4c93", linewidth=2.0, label="% Processor Performance")
    if "perf_warn_threshold_percent" in cfg:
        ax_perf.axhline(cfg["perf_warn_threshold_percent"], color="#e76f51", linestyle="--", label=f"Perf Warn ({cfg['perf_warn_threshold_percent']:.1f}%)")
    if "perf_hot_threshold_percent" in cfg:
        ax_perf.axhline(cfg["perf_hot_threshold_percent"], color="#b5179e", linestyle=":", label=f"Perf Limit Cap ({cfg['perf_hot_threshold_percent']:.1f}%)")
    ax_perf.set_ylabel("Delivered Perf (%)", fontsize=11, fontweight="bold")
    ax_perf.set_xlabel("Elapsed Time (seconds)", fontsize=11, fontweight="bold")
    ax_perf.legend(loc="upper left", framealpha=0.9)

    # Mark worker ramp steps
    for t_step, label, w in step_markers:
        for ax in (ax_temp, ax_freq, ax_perf):
            ax.axvline(t_step, color="gray", linestyle=":", alpha=0.6)
        ax_temp.text(t_step + 0.5, ax_temp.get_ylim()[0] + 2, label, rotation=90, fontsize=8, color="#333333", alpha=0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)

    # Also mirror to results/ in current and thermal_ctrl dirs
    alt_plots = ["results/calibration.png", os.path.join("thermal_ctrl", "results", "calibration.png")]
    for p in alt_plots:
        try:
            if os.path.abspath(p) != os.path.abspath(output_path):
                os.makedirs(os.path.dirname(os.path.abspath(p)) or ".", exist_ok=True)
                import shutil
                shutil.copy2(output_path, p)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Calibrate thermal thresholds and heavy background worker count.")
    parser.add_argument("--idle-duration", type=int, default=30, help="Idle phase duration in seconds (default: 30)")
    parser.add_argument("--step-duration", type=int, default=20, help="Duration of each worker step in seconds (default: 20)")
    parser.add_argument("--max-workers", type=int, default=None, help="Maximum worker count to ramp up to")
    parser.add_argument("--output-config", type=str, default="config.json", help="Path to write config.json")
    parser.add_argument("--output-plot", type=str, default="results/calibration.png", help="Path to write calibration plot")
    args = parser.parse_args()

    run_calibration(
        idle_duration_s=args.idle_duration,
        step_duration_s=args.step_duration,
        max_workers=args.max_workers,
        output_config=args.output_config,
        output_plot=args.output_plot,
    )


if __name__ == "__main__":
    main()
