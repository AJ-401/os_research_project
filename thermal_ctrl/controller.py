"""
controller.py - Workload Controller implementing M0, M1, M2, and M3 modes.
M0: No control (baseline).
M1: Static low priority (BELOW_NORMAL_PRIORITY_CLASS applied once).
M2: Fixed threshold (hard cap 40% when temp >= T_hot; release at T_hot - 4 C).
M3: Trend-aware (slope calculation over 10s, predicted temp at 15s, multi-step cap ladder down to 25%).
"""

import os
import sys
import time
from collections import deque
import psutil
import numpy as np
import pandas as pd

# Add local path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from workloads import WindowsJobObject, BackgroundJob


class WorkloadController:
    """Manages thermal feedback control on the background Windows Job Object."""

    CAP_LADDER = [100, 80, 60, 40, 25]

    def __init__(
        self,
        mode: str,
        job_object: WindowsJobObject,
        background_job: BackgroundJob,
        config: dict,
        log_csv: str = None,
    ):
        self.mode = mode.upper()
        if self.mode not in ("M0", "M1", "M2", "M3"):
            raise ValueError(f"Unknown mode: {mode}. Must be M0, M1, M2, or M3.")

        self.job_object = job_object
        self.background_job = background_job
        self.config = config
        self.log_csv = log_csv

        self.T_warn = float(config.get("T_warn", 70.0))
        self.T_hot = float(config.get("T_hot", 72.0))
        self.cooldown_target = float(config.get("cooldown_target", 67.8))

        self.current_cap = 100.0
        self.temp_history = deque(maxlen=10) # stores (timestamp, temp_c)
        self.log_records = []
        self.current_proc = psutil.Process()
        # Prime controller CPU measurement
        self.current_proc.cpu_percent(interval=None)
        self._last_cpu_time = time.time()
        self._last_ctrl_cpu = 0.0

        self._initialized = False

    def on_start(self):
        """Called once when background job begins."""
        if self._initialized:
            return

        if self.mode == "M1":
            # Static low priority once at start
            self.background_job.set_priority(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        
        self.current_cap = 100.0
        if self.job_object:
            self.job_object.set_cpu_rate_cap(100.0)

        self._initialized = True

    def step(self, telemetry_data: dict) -> dict:
        """Called every 1 second with latest telemetry dict.
        Returns controller decision dict.
        """
        now = telemetry_data.get("time", time.time())
        temp_c = float(telemetry_data.get("temp_c", 0.0))

        self.temp_history.append((now, temp_c))

        slope = 0.0
        predicted_temp = temp_c

        # Compute slope over recent history (deg C per second)
        if len(self.temp_history) >= 2:
            times = np.array([pt[0] for pt in self.temp_history])
            temps = np.array([pt[1] for pt in self.temp_history])
            t_rel = times - times[0]
            if t_rel[-1] > 0:
                # Simple linear regression slope
                coeffs = np.polyfit(t_rel, temps, 1)
                slope = float(coeffs[0])
            predicted_temp = temp_c + slope * 15.0

        decision_cap = self.current_cap

        if self.mode == "M0":
            # Do nothing
            decision_cap = 100.0

        elif self.mode == "M1":
            # Handled on_start (low priority), cap remains 100%
            decision_cap = 100.0

        elif self.mode == "M2":
            # Fixed threshold:
            # If temp >= T_hot: set hard cap to 40%
            # If temp <= T_hot - 4: remove cap (100%)
            # Otherwise maintain current cap (hysteresis gap)
            if temp_c >= self.T_hot:
                decision_cap = 40.0
            elif temp_c <= (self.T_hot - 4.0):
                decision_cap = 100.0
            else:
                decision_cap = self.current_cap

        elif self.mode == "M3":
            # Trend-aware controller:
            # Ladder: 100 -> 80 -> 60 -> 40 -> 25
            curr_idx = self._get_ladder_index(self.current_cap)

            if predicted_temp >= self.T_hot:
                # Lower cap by one step (increase throttling)
                new_idx = min(len(self.CAP_LADDER) - 1, curr_idx + 1)
                decision_cap = float(self.CAP_LADDER[new_idx])
            elif predicted_temp <= self.T_warn and slope <= 0.0:
                # Raise cap by one step (reduce throttling)
                new_idx = max(0, curr_idx - 1)
                decision_cap = float(self.CAP_LADDER[new_idx])
            else:
                decision_cap = self.current_cap

            # Never go below 25%
            decision_cap = max(25.0, decision_cap)

        # Apply cap to Job Object if changed
        if decision_cap != self.current_cap:
            if self.job_object:
                self.job_object.set_cpu_rate_cap(decision_cap)
            self.current_cap = decision_cap

        # Measure controller's own overhead (guarded against sub-second delta spikes)
        now_wall = time.time()
        if (now_wall - self._last_cpu_time) >= 0.2:
            ctrl_cpu = self.current_proc.cpu_percent(interval=None)
            self._last_cpu_time = now_wall
            self._last_ctrl_cpu = ctrl_cpu
        else:
            ctrl_cpu = self._last_ctrl_cpu

        record = {
            "timestamp": now,
            "mode": self.mode,
            "temp_c": temp_c,
            "slope": round(slope, 4),
            "predicted_temp": round(predicted_temp, 2),
            "current_cap": round(self.current_cap, 1),
            "controller_cpu_percent": round(ctrl_cpu, 2),
        }
        self.log_records.append(record)
        return record

    def _get_ladder_index(self, cap: float) -> int:
        """Finds closest index in CAP_LADDER."""
        for idx, val in enumerate(self.CAP_LADDER):
            if cap >= val:
                return idx
        return len(self.CAP_LADDER) - 1

    def close(self):
        """Flushes log records to CSV."""
        if self.log_csv and self.log_records:
            os.makedirs(os.path.dirname(os.path.abspath(self.log_csv)) or ".", exist_ok=True)
            df = pd.DataFrame(self.log_records)
            df.to_csv(self.log_csv, index=False)


def test_controller():
    """Validates M0, M1, M2, and M3 logic against mock telemetry stream."""
    print("Testing WorkloadController state transitions...")
    cfg = {"T_warn": 70.0, "T_hot": 72.0, "cooldown_target": 67.8}
    job = WindowsJobObject()
    bg = BackgroundJob(n_workers=1, total_work_chunks=1, chunk_size=1000)

    # Test M2 hysteresis
    m2 = WorkloadController("M2", job, bg, cfg)
    m2.on_start()
    r1 = m2.step({"time": 1.0, "temp_c": 72.5})
    assert r1["current_cap"] == 40.0, "M2 failed to cap at T_hot"
    r2 = m2.step({"time": 2.0, "temp_c": 70.0}) # Inside 4C gap
    assert r2["current_cap"] == 40.0, "M2 failed hysteresis retention"
    r3 = m2.step({"time": 3.0, "temp_c": 67.5}) # Cool down <= T_hot - 4 (68 C)
    assert r3["current_cap"] == 100.0, "M2 failed to release cap"
    print("-> M2 logic passed.")

    # Test M3 trend-awareness
    m3 = WorkloadController("M3", job, bg, cfg)
    m3.on_start()
    # Rapid heating slope: 65, 68 -> slope = 3 C/s -> predicted = 68 + 45 = 113 > 72
    m3.step({"time": 1.0, "temp_c": 65.0})
    r_step2 = m3.step({"time": 2.0, "temp_c": 68.0})
    assert r_step2["current_cap"] == 80.0, f"M3 expected cap step down to 80, got {r_step2['current_cap']}"
    r_step3 = m3.step({"time": 3.0, "temp_c": 71.0})
    assert r_step3["current_cap"] == 60.0, f"M3 expected cap step down to 60, got {r_step3['current_cap']}"
    print("-> M3 logic passed.")
    job.close()
    print("All controller tests passed!")


if __name__ == "__main__":
    test_controller()
