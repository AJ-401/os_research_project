"""
test_thermal_suite.py - End-to-end verification test suite for Thermal-Aware Workload Controller.
"""

import os
import json
import pytest
import psutil
import pandas as pd
from thermal_ctrl.telemetry import TelemetryReader, get_active_power_plan
from thermal_ctrl.workloads import WindowsJobObject, BackgroundJob, ForegroundProbe
from thermal_ctrl.controller import WorkloadController


def test_telemetry_reader():
    """Verify telemetry initializes and returns valid sensor metrics."""
    reader = TelemetryReader(interval_s=1.0)
    data = reader.read()
    reader.close()

    assert "temp_c" in data
    assert "freq_mhz" in data
    assert "perf_percent" in data
    assert "cpu_util_percent" in data
    assert data["temp_c"] > 0.0, f"Expected positive temperature, got {data['temp_c']}"
    assert data["freq_mhz"] > 0.0, f"Expected positive frequency, got {data['freq_mhz']}"


def test_job_object_cpu_rate_cap():
    """Verify Windows Job Object CPU rate control API."""
    job = WindowsJobObject()
    # Test setting 40% cap
    assert job.set_cpu_rate_cap(40.0) is True
    # Test setting 100% (uncapped)
    assert job.set_cpu_rate_cap(100.0) is True
    # Test setting 25% cap
    assert job.set_cpu_rate_cap(25.0) is True
    job.close()


def test_controller_modes_logic():
    """Verify controller transitions for M0, M1, M2, and M3."""
    cfg = {"T_warn": 70.0, "T_hot": 72.0, "cooldown_target": 67.8}
    job = WindowsJobObject()
    bg = BackgroundJob(n_workers=1, total_work_chunks=1, chunk_size=1000)

    # M2: Fixed threshold + hysteresis
    m2 = WorkloadController("M2", job, bg, cfg)
    m2.on_start()
    r1 = m2.step({"time": 1.0, "temp_c": 72.5})
    assert r1["current_cap"] == 40.0
    r2 = m2.step({"time": 2.0, "temp_c": 70.0}) # In 4C hysteresis band
    assert r2["current_cap"] == 40.0
    r3 = m2.step({"time": 3.0, "temp_c": 67.5}) # Cools down below 68 C
    assert r3["current_cap"] == 100.0

    # M3: Trend-aware slope ladder
    m3 = WorkloadController("M3", job, bg, cfg)
    m3.on_start()
    m3.step({"time": 1.0, "temp_c": 65.0})
    r_m3_1 = m3.step({"time": 2.0, "temp_c": 68.0})
    assert r_m3_1["current_cap"] == 80.0
    r_m3_2 = m3.step({"time": 3.0, "temp_c": 71.0})
    assert r_m3_2["current_cap"] == 60.0

    job.close()


def test_config_json_validity():
    """Verify config.json exists and adheres to PRD requirements."""
    assert os.path.exists("config.json"), "config.json missing"
    with open("config.json", "r") as f:
        cfg = json.load(f)
    for key in ["T_warn", "T_hot", "cooldown_target", "heavy_workers", "power_plan"]:
        assert key in cfg, f"Key {key} missing from config.json"
    assert cfg["T_hot"] > cfg["T_warn"]
    assert cfg["heavy_workers"] > 0


def test_experiment_results_generated():
    """Verify summary and analysis artifacts exist."""
    assert os.path.exists("results/summary.csv"), "summary.csv missing"
    assert os.path.exists("results/modes_table.csv"), "modes_table.csv missing"
    assert os.path.exists("results/phase1_table.csv"), "phase1_table.csv missing"
    assert os.path.exists("results/calibration.png"), "calibration.png missing"
    assert os.path.exists("results/tradeoff.png"), "tradeoff.png missing"
    assert os.path.exists("results/temp_vs_time.png"), "temp_vs_time.png missing"
