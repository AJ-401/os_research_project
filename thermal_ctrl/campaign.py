"""
campaign.py - one-session, randomised policy-comparison campaign.

Save inside the thermal_ctrl folder. Run from the project root:

    python thermal_ctrl/campaign.py --repeats 3 --duration 150
    python thermal_ctrl/campaign.py --summarize

It reuses telemetry.py, workloads.py, controller.py and run_experiment.py and
modifies none of them. All output goes to results_v2/ (nothing in results/ is touched).

Policies
  M0   no control
  M1   background at BELOW_NORMAL priority
  M2   fixed threshold: 40% cap at T_hot, release 4 C lower
  M3   trend-aware cap ladder (original controller)
  M4   placement: background kept OFF the foreground's physical core (CPUs 0 and 1)
  M5   power-limit-aware: cap ladder driven by delivered CPU performance (perf_percent)
  M6   hybrid: M4 placement (background off CPUs 0 and 1) + M5 power-limit-aware cap
  M7   M5 with back-off on raising the cap (damped)
  M8   M4 placement + M7 damped controller
  S50  static 50% cap for the whole run
  S25  static 25% cap for the whole run

Per run: continuous background load (all workers), a warm-up that is NOT measured,
then a measured window. Foreground latency is computed only over that window.
Background cost = CPU time actually delivered to the background workers
(as % of the whole machine); it is a proxy for work done.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys
import time
import json
import random
import argparse
import platform

import numpy as np
import pandas as pd
import psutil
import multiprocessing

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from telemetry import TelemetryReader, get_active_power_plan
from workloads import WindowsJobObject, BackgroundJob, ForegroundProbe
from controller import WorkloadController
from run_experiment import load_config, wait_for_cooldown

POLICIES = ["M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "S50", "S25"]
STATIC_CAPS = {"S50": 50.0, "S25": 25.0}


def _matmul_worker(worker_id, num_chunks, chunk_size, stop_event=None):
    """Heavier background: small matrix multiplications (vector maths draws much more power)."""
    a = np.random.rand(256, 256)
    b = np.random.rand(256, 256)
    n = 0
    while True:
        a = a @ b
        a /= a.max()
        n += 1
        if (n % 200) == 0 and stop_event is not None and stop_event.is_set():
            break


class MatmulBackgroundJob(BackgroundJob):
    """Same as BackgroundJob (same Job Object, same cap), but the workers do matrix maths."""

    def start(self, continuous=True):
        self.start_time = time.time()
        for i in range(self.n_workers):
            p = multiprocessing.Process(target=_matmul_worker,
                                        args=(i, -1, 0, self.stop_event),
                                        name=f"bg_matmul_{i}")
            p.daemon = True
            p.start()
            self.procs.append(p)
            self.job_object.assign_process(p.pid)
            if self.affinity_cpus is not None:
                try:
                    psutil.Process(p.pid).cpu_affinity(self.affinity_cpus)
                except Exception:
                    pass


class PerfController:
    """M5: power-limit-aware controller. Same interface as WorkloadController.

    Idea: the chip slows down when it hits its power limit, and perf_percent
    (delivered speed, 100 = base clock) shows that. Compare a short average of
    perf_percent with the healthy level seen at the start of the run.
      - below DOWN_FRAC of healthy      -> step the cap DOWN one rung
      - at/above UP_FRAC for UP_HOLD s  -> step the cap UP one rung
    A minimum wait (MIN_DWELL) between any two changes stops flapping.
    PARAMETERS ARE FIXED BEFORE THE CAMPAIGN. Do not tune them after seeing results.
    """
    LADDER = [100, 80, 60, 40, 25]
    BASE_N = 10        # first BUSY readings (cpu >= 90%) used to learn the healthy level (median)
    AVG_N = 5          # seconds averaged
    DOWN_FRAC = 0.95
    UP_FRAC = 0.98
    UP_HOLD = 10       # seconds the recovery must last before raising the cap
    MIN_DWELL = 8      # seconds between cap changes

    def __init__(self, job_object, log_csv=None):
        self.job_object = job_object
        self.log_csv = log_csv
        self.current_cap = 100.0
        self.perf_hist = []
        self.loaded_perf = []   # perf readings taken while the CPU is really busy
        self.baseline = None
        self.up_ok = 0
        self.last_change = None
        self.log_records = []

    def on_start(self):
        self.current_cap = 100.0
        if self.job_object:
            self.job_object.set_cpu_rate_cap(100.0)

    def step(self, data):
        now = data.get("time", time.time())
        perf = float(data.get("perf_percent", 0.0))
        util = float(data.get("cpu_util_percent", 100.0))
        self.perf_hist.append(perf)
        # Learn the healthy level only from seconds when the CPU is busy (>= 90%),
        # so the start-up ramp (workers still launching, CPU idle) cannot lower it.
        if self.baseline is None and util >= 90.0:
            self.loaded_perf.append(perf)
            if len(self.loaded_perf) >= self.BASE_N:
                self.baseline = float(np.median(self.loaded_perf[:self.BASE_N]))
        avg = float(np.mean(self.perf_hist[-self.AVG_N:]))
        cap = self.current_cap
        can_change = self.last_change is None or (now - self.last_change) >= self.MIN_DWELL
        idx = self.LADDER.index(int(cap))
        if self.baseline is not None:
            cap = self._decide(now, avg, cap, idx, can_change)
        if cap != self.current_cap:
            if self.job_object:
                self.job_object.set_cpu_rate_cap(cap)
            self.current_cap = cap
            self.last_change = now
        rec = {"timestamp": now, "mode": "M5", "perf_percent": perf,
               "perf_avg": round(avg, 2), "baseline": self.baseline,
               "current_cap": round(self.current_cap, 1)}
        self.log_records.append(rec)
        return rec

    def _decide(self, now, avg, cap, idx, can_change):
        """M5 rule: one rung down when speed is low, one rung up after UP_HOLD s of recovery."""
        if avg < self.DOWN_FRAC * self.baseline:
            self.up_ok = 0
            if can_change and idx < len(self.LADDER) - 1:
                cap = float(self.LADDER[idx + 1])
        elif avg >= self.UP_FRAC * self.baseline:
            self.up_ok += 1
            if can_change and self.up_ok >= self.UP_HOLD and idx > 0:
                cap = float(self.LADDER[idx - 1])
                self.up_ok = 0
        else:
            self.up_ok = 0
        return cap

    def close(self):
        if self.log_csv and self.log_records:
            pd.DataFrame(self.log_records).to_csv(self.log_csv, index=False)



class DampedPerfController(PerfController):
    """M7/M8: M5 with a back-off on raising the cap (fixes the up-down bouncing).

    Same signal and same ladder as M5. The only change: if raising the cap makes
    the speed drop again within RAISE_TEST s (a failed raise), the wait before the
    next raise doubles (up to UP_HOLD_MAX). If a raise holds for STICK_S s, the wait
    resets to UP_HOLD_BASE. PARAMETERS FIXED BEFORE THE RUNS; do not tune after results.
    """
    UP_HOLD_BASE = 20
    UP_HOLD_MAX = 160
    RAISE_TEST = 20
    STICK_S = 60

    def __init__(self, job_object, log_csv=None):
        super().__init__(job_object, log_csv)
        self.up_hold = self.UP_HOLD_BASE
        self.last_raise = None

    def _decide(self, now, avg, cap, idx, can_change):
        if avg < self.DOWN_FRAC * self.baseline:
            self.up_ok = 0
            if can_change and idx < len(self.LADDER) - 1:
                if self.last_raise is not None and (now - self.last_raise) <= self.RAISE_TEST:
                    self.up_hold = min(self.up_hold * 2, self.UP_HOLD_MAX)
                self.last_raise = None
                cap = float(self.LADDER[idx + 1])
        elif avg >= self.UP_FRAC * self.baseline:
            self.up_ok += 1
            if can_change and self.up_ok >= self.up_hold and idx > 0:
                cap = float(self.LADDER[idx - 1])
                self.up_ok = 0
                self.last_raise = now
            if self.last_raise is not None and (now - self.last_raise) >= self.STICK_S:
                self.up_hold = self.UP_HOLD_BASE
                self.last_raise = None
        else:
            self.up_ok = 0
        return cap


def bg_cpu_seconds(procs):
    """Total CPU time (user + system) used so far by the background worker processes."""
    total = 0.0
    for p in procs:
        try:
            t = p.cpu_times()
            total += t.user + t.system
        except psutil.Error:
            pass
    return total


def plugged_in():
    try:
        b = psutil.sensors_battery()
        return None if b is None else bool(b.power_plugged)
    except Exception:
        return None


def pct(arr, q):
    return round(float(np.percentile(arr, q)), 3) if len(arr) else float("nan")


def run_one(policy, repeat, cfg, args):
    os.makedirs(args.out, exist_ok=True)
    tag = f"{policy}_{repeat}"
    ctrl_csv = os.path.join(args.out, f"camp_{tag}_ctrl.csv")
    probe_csv = os.path.join(args.out, f"camp_{tag}_probe.csv")
    telem_csv = os.path.join(args.out, f"camp_{tag}_telemetry.csv")

    n_cpus = psutil.cpu_count(logical=True) or 8
    workers = args.workers or int(cfg.get("heavy_workers", n_cpus))
    T_hot = float(cfg.get("T_hot", 72.0))
    mode = policy if policy in ("M0", "M1", "M2", "M3") else "M0"
    affinity = list(range(2, n_cpus)) if policy in ("M4", "M6", "M8") else None
    static_cap = STATIC_CAPS.get(policy)

    print(f"\n=== {policy} repeat {repeat} | workers={workers} | warmup={args.warmup}s "
          f"measure={args.duration}s ===")

    telemetry = TelemetryReader(interval_s=1.0)
    wait_for_cooldown(telemetry, float(cfg.get("cooldown_target", 67.8)), max_wait_s=args.cooldown_max)
    start_temp = telemetry.read().get("temp_c", 0.0)

    job = WindowsJobObject()
    cap_failures = []
    original_set = job.set_cpu_rate_cap

    def logged_set(percent):
        ok = original_set(percent)
        if not ok:
            cap_failures.append(percent)
        return ok

    job.set_cpu_rate_cap = logged_set

    bg_cls = MatmulBackgroundJob if args.bg == "matmul" else BackgroundJob
    bg = bg_cls(n_workers=workers, job_object=job, affinity_cpus=affinity)
    probe = ForegroundProbe(pinned_cpu=0)
    if policy in ("M5", "M6"):
        ctrl = PerfController(job_object=job, log_csv=ctrl_csv)
    elif policy in ("M7", "M8"):
        ctrl = DampedPerfController(job_object=job, log_csv=ctrl_csv)
    else:
        ctrl = WorkloadController(mode=mode, job_object=job, background_job=bg, config=cfg, log_csv=ctrl_csv)

    rows = []
    t_w0 = t_w1 = None
    cpu0 = cpu1 = 0.0
    try:
        probe.start(csv_path=probe_csv)
        ctrl.on_start()
        bg.start(continuous=True)
        if static_cap is not None:
            logged_set(static_cap)
        procs = [psutil.Process(p.pid) for p in bg.procs]
        t_start = time.time()
        while True:
            tick = time.time()
            elapsed = tick - t_start
            data = telemetry.read()
            rows.append(data)
            ctrl.step(data)
            if t_w0 is None and elapsed >= args.warmup:
                t_w0 = tick
                cpu0 = bg_cpu_seconds(procs)
            if elapsed >= args.warmup + args.duration:
                t_w1 = tick
                cpu1 = bg_cpu_seconds(procs)
                break
            time.sleep(max(0.0, 1.0 - (time.time() - tick)))
    finally:
        bg.terminate()
        job.close()
        probe.stop()
        telemetry.close()
        ctrl.close()

    df_t = pd.DataFrame(rows)
    df_t.to_csv(telem_csv, index=False)

    fg = np.array([])
    if os.path.exists(probe_csv) and t_w0 is not None and t_w1 is not None:
        dfp = pd.read_csv(probe_csv)
        fg = dfp[(dfp["timestamp"] >= t_w0) & (dfp["timestamp"] <= t_w1)]["task_time_ms"].values

    win = df_t[(df_t["time"] >= t_w0) & (df_t["time"] <= t_w1)] if t_w0 is not None else df_t.iloc[0:0]

    caps = [r["current_cap"] for r in ctrl.log_records]
    cap_changes = int(sum(1 for a, b in zip(caps, caps[1:]) if a != b))
    if static_cap is not None:
        mean_cap = static_cap
    else:
        win_caps = [r["current_cap"] for r in ctrl.log_records
                    if t_w0 is not None and t_w0 <= r["timestamp"] <= t_w1]
        mean_cap = round(float(np.mean(win_caps)), 1) if win_caps else 100.0

    span = (t_w1 - t_w0) if (t_w0 is not None and t_w1 is not None) else float("nan")
    bg_pct = round((cpu1 - cpu0) / span / n_cpus * 100.0, 2) if span == span and span > 0 else float("nan")

    row = {
        "policy": policy,
        "repeat": repeat,
        "start_temp_c": start_temp,
        "workers": workers,
        "window_s": round(span, 1) if span == span else float("nan"),
        "fg_samples": int(len(fg)),
        "fg_p50_ms": pct(fg, 50),
        "fg_p95_ms": pct(fg, 95),
        "fg_p99_ms": pct(fg, 99),
        "bg_cpu_pct": bg_pct,
        "max_temp_c": round(float(win["temp_c"].max()), 2) if len(win) else float("nan"),
        "mean_temp_c": round(float(win["temp_c"].mean()), 2) if len(win) else float("nan"),
        "time_above_T_hot_s": int((win["temp_c"] >= T_hot).sum()) if len(win) else 0,
        "mean_freq_mhz": round(float(win["freq_mhz"].mean()), 1) if len(win) else float("nan"),
        "mean_perf_pct": round(float(win["perf_percent"].mean()), 1) if len(win) else float("nan"),
        "perf_p5": round(float(np.percentile(win["perf_percent"], 5)), 1) if len(win) else float("nan"),
        "limit_flag_frac": round(float((win["perf_limit_flag"] > 0).mean()), 3) if len(win) else float("nan"),
        "mean_cap": mean_cap,
        "cap_changes": cap_changes,
        "cap_failures": len(cap_failures),
        "plugged_in": plugged_in(),
    }
    print("   fg p50/p95/p99 = {:.2f}/{:.2f}/{:.2f} ms | bg = {}% of machine | max temp {} | "
          "cap changes {} | mean cap {}".format(row["fg_p50_ms"], row["fg_p95_ms"], row["fg_p99_ms"],
                                                row["bg_cpu_pct"], row["max_temp_c"], cap_changes, mean_cap))
    if cap_failures:
        print(f"   WARNING: {len(cap_failures)} cap call(s) FAILED: {cap_failures}")
    return row


def append_row(row, path):
    df = pd.DataFrame([row])
    if not os.path.exists(path):
        df.to_csv(path, index=False)
    else:
        df.to_csv(path, mode="a", header=False, index=False)


def summarize(path):
    df = pd.read_csv(path)
    cols = ["fg_p50_ms", "fg_p95_ms", "fg_p99_ms", "bg_cpu_pct", "mean_perf_pct", "perf_p5",
            "max_temp_c", "time_above_T_hot_s", "mean_cap", "cap_changes"]
    cols = [c for c in cols if c in df.columns]
    out = df.groupby("policy")[cols].agg(["mean", "std"]).round(2)
    out.insert(0, "n_runs", df.groupby("policy").size())
    print(out.to_string())
    base = df[df["policy"] == "M0"]
    if len(base):
        b95, b99, bbg = base["fg_p95_ms"].mean(), base["fg_p99_ms"].mean(), base["bg_cpu_pct"].mean()
        print("\nRelative to M0 (no control):")
        for pol, g in df.groupby("policy"):
            if pol == "M0":
                continue
            print(f"  {pol:4s} fg p95 {100 * (g['fg_p95_ms'].mean() / b95 - 1):+6.1f}% | "
                  f"fg p99 {100 * (g['fg_p99_ms'].mean() / b99 - 1):+6.1f}% | "
                  f"bg CPU {100 * (g['bg_cpu_pct'].mean() / bbg - 1):+6.1f}%")


def main():
    ap = argparse.ArgumentParser(description="Randomised policy-comparison campaign.")
    ap.add_argument("--policies", nargs="+", default=POLICIES, choices=POLICIES)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--duration", type=int, default=150, help="measured seconds per run")
    ap.add_argument("--warmup", type=int, default=10, help="unmeasured seconds at start")
    ap.add_argument("--workers", type=int, default=None, help="default: heavy_workers from config")
    ap.add_argument("--cooldown-max", type=int, default=240)
    ap.add_argument("--config", type=str, default="config.json")
    ap.add_argument("--out", type=str, default="results_v2")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--bg", choices=["py", "matmul"], default="py",
                    help="background workload: py = original integer/hash loop, matmul = heavier matrix maths")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()

    runs_csv = os.path.join(args.out, "campaign_runs.csv")
    if args.summarize:
        summarize(runs_csv)
        return

    cfg = load_config(args.config)
    os.makedirs(args.out, exist_ok=True)
    meta = {
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "config_used": cfg,
        "power_plan": get_active_power_plan(),
        "plugged_in": plugged_in(),
        "python": platform.python_version(),
        "psutil": psutil.__version__,
        "logical_cpus": psutil.cpu_count(logical=True),
    }
    with open(os.path.join(args.out, "campaign_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)

    random.seed(args.seed)
    for r in range(1, args.repeats + 1):
        order = list(args.policies)
        random.shuffle(order)
        print(f"\n##### Repeat {r}/{args.repeats} | order: {order}")
        for pol in order:
            append_row(run_one(pol, r, cfg, args), runs_csv)

    print("\nDone. Summary:\n")
    summarize(runs_csv)


if __name__ == "__main__":
    main()