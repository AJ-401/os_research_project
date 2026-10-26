"""
telemetry.py - Hardware sensor monitoring for thermal, frequency, and CPU metrics on Windows.
Reads Performance Counters (PDH) and psutil metrics.
"""

import os
import sys
import time
import subprocess
import ctypes
from ctypes import byref, c_uint, Structure
import psutil

try:
    import win32pdh
except ImportError:
    win32pdh = None


class _NvmlUtilization(Structure):
    _fields_ = [("gpu", c_uint), ("memory", c_uint)]


class TelemetryReader:
    """Reads temperature, CPU frequency, performance limit, CPU usage, and GPU metrics."""

    def __init__(self, interval_s: float = 1.0):
        self.interval_s = interval_s
        self.h_query = None
        self.temp_counter = None
        self.high_prec_temp_counter = None
        self.freq_counter = None
        self.perf_counter = None
        self.limit_counter = None
        self.limit_flags_counter = None
        self.temp_path = None
        self.high_prec_path = None
        self.freq_path = None
        self.perf_path = None
        self.limit_path = None
        self.limit_flags_path = None
        self.temp_resolution = 1.0
        self.is_high_precision = False
        self.sensor_status = {}

        # GPU Telemetry State (NVML)
        self.gpu_available = False
        self.nvml_handle = None
        self.gpu_device = None
        self.gpu_name = None

        self._init_counters()
        self._init_gpu()

    def _init_counters(self):
        if win32pdh is None:
            raise RuntimeError("win32pdh (pywin32) is required for Windows PDH telemetry.")

        self.h_query = win32pdh.OpenQuery()

        # 1. Discover Thermal Zone
        tz_instance = None
        try:
            _, tz_instances = win32pdh.EnumObjectItems(None, None, "Thermal Zone Information", -1)
            if tz_instances:
                tz_instance = tz_instances[0]
        except Exception as e:
            self.sensor_status["thermal_zone_enum_error"] = str(e)

        if tz_instance:
            # Check High Precision Temperature first
            try:
                hp_path = f"\\Thermal Zone Information({tz_instance})\\High Precision Temperature"
                self.high_prec_temp_counter = win32pdh.AddCounter(self.h_query, hp_path)
                self.high_prec_path = hp_path
                self.is_high_precision = True
                self.temp_resolution = 0.1
                self.sensor_status["temperature_sensor"] = f"PDH: {hp_path} (High Precision, 0.1 C)"
            except Exception:
                self.high_prec_temp_counter = None

            # Also add standard Temperature counter as fallback/comparison
            try:
                std_path = f"\\Thermal Zone Information({tz_instance})\\Temperature"
                self.temp_counter = win32pdh.AddCounter(self.h_query, std_path)
                self.temp_path = std_path
                if not self.is_high_precision:
                    self.temp_resolution = 1.0
                    self.sensor_status["temperature_sensor"] = f"PDH: {std_path} (1.0 C resolution)"
            except Exception as e:
                if not self.is_high_precision:
                    self.sensor_status["temperature_sensor"] = f"Failed to attach: {e}"
        else:
            self.sensor_status["temperature_sensor"] = "No Thermal Zone instance found"

        # 2. Discover Processor Information counters (_Total)
        proc_instances = []
        try:
            _, proc_instances = win32pdh.EnumObjectItems(None, None, "Processor Information", -1)
        except Exception:
            pass

        target_total = "_Total"
        if "_Total" not in proc_instances and "0,_Total" in proc_instances:
            target_total = "0,_Total"

        # Frequency
        try:
            f_path = f"\\Processor Information({target_total})\\Processor Frequency"
            self.freq_counter = win32pdh.AddCounter(self.h_query, f_path)
            self.freq_path = f_path
            self.sensor_status["frequency_sensor"] = f"PDH: {f_path}"
        except Exception as e:
            self.sensor_status["frequency_sensor"] = f"Failed: {e}"

        # % Processor Performance
        try:
            p_path = f"\\Processor Information({target_total})\\% Processor Performance"
            self.perf_counter = win32pdh.AddCounter(self.h_query, p_path)
            self.perf_path = p_path
            self.sensor_status["perf_percent_sensor"] = f"PDH: {p_path}"
        except Exception as e:
            self.sensor_status["perf_percent_sensor"] = f"Failed: {e}"

        # % Performance Limit
        try:
            lim_path = f"\\Processor Information({target_total})\\% Performance Limit"
            self.limit_counter = win32pdh.AddCounter(self.h_query, lim_path)
            self.limit_path = lim_path
            self.sensor_status["perf_limit_sensor"] = f"PDH: {lim_path}"
        except Exception as e:
            self.sensor_status["perf_limit_sensor"] = f"Failed: {e}"

        # Performance Limit Flags
        try:
            flags_path = f"\\Processor Information({target_total})\\Performance Limit Flags"
            self.limit_flags_counter = win32pdh.AddCounter(self.h_query, flags_path)
            self.limit_flags_path = flags_path
            self.sensor_status["perf_limit_flags_sensor"] = f"PDH: {flags_path}"
        except Exception as e:
            self.sensor_status["perf_limit_flags_sensor"] = f"Failed: {e}"

        # First collect call primes the PDH difference counters
        try:
            win32pdh.CollectQueryData(self.h_query)
        except Exception:
            pass

    def _init_gpu(self):
        """Initializes GPU telemetry via NVML (ctypes) with graceful fallback."""
        nvml_paths = [
            r"C:\Windows\System32\nvml.dll",
            r"C:\Program Files\NVIDIA Corporation\NVSMI\nvml.dll",
        ]
        dll_path = None
        for p in nvml_paths:
            if os.path.exists(p):
                dll_path = p
                break

        if dll_path:
            try:
                self.nvml_handle = ctypes.WinDLL(dll_path)
                res = self.nvml_handle.nvmlInit_v2()
                if res == 0:
                    dev = ctypes.c_void_p()
                    if self.nvml_handle.nvmlDeviceGetHandleByIndex_v2(0, byref(dev)) == 0:
                        self.gpu_device = dev
                        self.gpu_available = True
                        # Get GPU Name
                        name_buf = ctypes.create_string_buffer(96)
                        if self.nvml_handle.nvmlDeviceGetName(dev, name_buf, 96) == 0:
                            self.gpu_name = name_buf.value.decode("utf-8", errors="ignore")
                        else:
                            self.gpu_name = "NVIDIA GPU"
                        self.sensor_status["gpu_sensor"] = f"NVML: {self.gpu_name}"
            except Exception as e:
                self.sensor_status["gpu_sensor"] = f"NVML init failed: {e}"

        if not self.gpu_available:
            # Check if nvidia-smi exists as CLI fallback
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    stderr=subprocess.DEVNULL,
                    text=True,
                )
                self.gpu_name = out.strip().split("\n")[0]
                self.gpu_available = True
                self.sensor_status["gpu_sensor"] = f"nvidia-smi: {self.gpu_name}"
            except Exception:
                self.gpu_available = False
                self.sensor_status["gpu_sensor"] = "None detected (CPU-only system)"

    def _read_gpu(self) -> tuple:
        """Reads GPU (temp_c, util_percent, power_w). Returns (None, None, None) if unavailable."""
        if not self.gpu_available:
            return None, None, None

        if self.nvml_handle and self.gpu_device:
            try:
                temp = c_uint()
                self.nvml_handle.nvmlDeviceGetTemperature(self.gpu_device, 0, byref(temp))
                gpu_temp = float(temp.value)

                util = _NvmlUtilization()
                self.nvml_handle.nvmlDeviceGetUtilizationRates(self.gpu_device, byref(util))
                gpu_util = float(util.gpu)

                power = c_uint()
                self.nvml_handle.nvmlDeviceGetPowerUsage(self.gpu_device, byref(power))
                gpu_power = round(power.value / 1000.0, 2)

                return gpu_temp, gpu_util, gpu_power
            except Exception:
                pass

        # Subprocess fallback if NVML direct C call failed
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,power.draw", "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            parts = [p.strip() for p in out.strip().split(",")]
            gpu_temp = float(parts[0]) if len(parts) > 0 else None
            gpu_util = float(parts[1]) if len(parts) > 1 else None
            gpu_power = float(parts[2]) if len(parts) > 2 else None
            return gpu_temp, gpu_util, gpu_power
        except Exception:
            return None, None, None

    def read(self) -> dict:
        """Collect and return a telemetry dictionary:
        time, temp_c, freq_mhz, perf_percent, perf_limit_flag, cpu_util_percent,
        gpu_temp_c, gpu_util_percent, gpu_power_w
        """
        now = time.time()

        # Update PDH query
        if self.h_query:
            try:
                win32pdh.CollectQueryData(self.h_query)
            except Exception:
                pass

        # Temperature
        temp_c = 0.0
        if self.is_high_precision and self.high_prec_temp_counter:
            try:
                val = win32pdh.GetFormattedCounterValue(self.high_prec_temp_counter, win32pdh.PDH_FMT_DOUBLE)[1]
                temp_c = round((val / 10.0) - 273.15, 2)
            except Exception:
                temp_c = 0.0
        elif self.temp_counter:
            try:
                val = win32pdh.GetFormattedCounterValue(self.temp_counter, win32pdh.PDH_FMT_DOUBLE)[1]
                temp_c = round(val - 273.15, 2)
            except Exception:
                temp_c = 0.0

        # Frequency
        freq_mhz = 0.0
        if self.freq_counter:
            try:
                freq_mhz = round(win32pdh.GetFormattedCounterValue(self.freq_counter, win32pdh.PDH_FMT_DOUBLE)[1], 1)
            except Exception:
                pass
        if freq_mhz <= 0.0:
            # Fallback to psutil cpu_freq if PDH returns 0
            cur = psutil.cpu_freq()
            if cur:
                freq_mhz = round(cur.current, 1)

        # % Processor Performance
        perf_percent = 100.0
        if self.perf_counter:
            try:
                perf_percent = round(win32pdh.GetFormattedCounterValue(self.perf_counter, win32pdh.PDH_FMT_DOUBLE)[1], 2)
            except Exception:
                pass

        # Performance Limit Flag / Throttle reason
        perf_limit_flag = 0
        if self.limit_flags_counter:
            try:
                perf_limit_flag = int(win32pdh.GetFormattedCounterValue(self.limit_flags_counter, win32pdh.PDH_FMT_LONG)[1])
            except Exception:
                pass
        if perf_limit_flag == 0 and self.limit_counter:
            try:
                # If % Performance Limit is less than 100, hardware is limiting speed
                lim_pct = win32pdh.GetFormattedCounterValue(self.limit_counter, win32pdh.PDH_FMT_DOUBLE)[1]
                if lim_pct < 99.0:
                    perf_limit_flag = 1
            except Exception:
                pass

        # CPU Utilization %
        cpu_util_percent = psutil.cpu_percent(interval=None)

        # GPU Metrics
        gpu_temp_c, gpu_util_percent, gpu_power_w = self._read_gpu()

        return {
            "time": now,
            "temp_c": temp_c,
            "freq_mhz": freq_mhz,
            "perf_percent": perf_percent,
            "perf_limit_flag": perf_limit_flag,
            "cpu_util_percent": cpu_util_percent,
            "gpu_temp_c": gpu_temp_c,
            "gpu_util_percent": gpu_util_percent,
            "gpu_power_w": gpu_power_w,
        }

    def close(self):
        if self.h_query:
            try:
                win32pdh.CloseQuery(self.h_query)
            except Exception:
                pass
            self.h_query = None

        if self.nvml_handle:
            try:
                self.nvml_handle.nvmlShutdown()
            except Exception:
                pass
            self.nvml_handle = None

    def __del__(self):
        self.close()


def get_active_power_plan() -> str:
    """Returns the current Windows active power scheme name."""
    try:
        out = subprocess.check_output(["powercfg", "/getactivescheme"], text=True)
        return out.strip()
    except Exception as e:
        return f"Unknown ({e})"


def write_telemetry_report(reader: TelemetryReader, sample_readings: list, report_path: str = "telemetry_report.txt"):
    """Generates the telemetry report detailing available sensors, resolution, and baseline values."""
    power_plan = get_active_power_plan()
    cpu_count_logical = psutil.cpu_count(logical=True)
    cpu_count_physical = psutil.cpu_count(logical=False)

    lines = [
        "============================================================",
        "THERMAL & WORKLOAD TELEMETRY REPORT",
        "============================================================",
        f"Generated At: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Operating System: {sys.platform} (Windows)",
        f"CPU Architecture: {os.environ.get('PROCESSOR_IDENTIFIER', 'Intel/AMD x64')}",
        f"CPU Logical Cores: {cpu_count_logical}",
        f"CPU Physical Cores: {cpu_count_physical}",
        f"Active Power Plan: {power_plan}",
        "",
        "------------------------------------------------------------",
        "1. DETECTED HARDWARE SENSORS & COUNTER PATHS",
        "------------------------------------------------------------",
        f"Temperature Sensor: {reader.sensor_status.get('temperature_sensor', 'None')}",
        f"Temperature High Precision: {'Yes (0.1 C)' if reader.is_high_precision else 'No'}",
        f"Temperature Resolution: {reader.temp_resolution} C",
        f"Frequency Sensor: {reader.sensor_status.get('frequency_sensor', 'None')}",
        f"Processor Performance Sensor: {reader.sensor_status.get('perf_percent_sensor', 'None')}",
        f"Performance Limit Sensor: {reader.sensor_status.get('perf_limit_sensor', 'None')}",
        f"Performance Limit Flags Sensor: {reader.sensor_status.get('perf_limit_flags_sensor', 'None')}",
        f"GPU Sensor: {reader.sensor_status.get('gpu_sensor', 'None')}",
        "CPU Utilization Sensor: psutil.cpu_percent (interval=None)",
        "",
        "------------------------------------------------------------",
        "2. TELEMETRY CONFIGURATION",
        "------------------------------------------------------------",
        f"Sampling Interval: {reader.interval_s:.1f} s",
        f"Smallest Temperature Step (Resolution): {reader.temp_resolution:.2f} C",
        f"GPU Monitoring: {'Online (' + str(reader.gpu_name) + ')' if reader.gpu_available else 'Not present / CPU-only'}",
        "Status: Online and operational",
        "",
        "------------------------------------------------------------",
        "3. 10-SECOND BASELINE READINGS SAMPLE",
        "------------------------------------------------------------",
        "Time (s) | CPU (C) | Freq (MHz) | Perf (%) | CPU (%) | GPU (C) | GPU (%) | GPU (W)",
        "---------+---------+------------+----------+---------+---------+---------+--------",
    ]

    for i, r in enumerate(sample_readings):
        gpu_t_str = f"{r['gpu_temp_c']:7.1f}" if r.get('gpu_temp_c') is not None else "    N/A"
        gpu_u_str = f"{r['gpu_util_percent']:7.1f}" if r.get('gpu_util_percent') is not None else "    N/A"
        gpu_p_str = f"{r['gpu_power_w']:7.1f}" if r.get('gpu_power_w') is not None else "    N/A"
        lines.append(
            f"{i+1:7d}  | {r['temp_c']:7.2f} | {r['freq_mhz']:10.1f} | {r['perf_percent']:8.1f} | {r['cpu_util_percent']:7.1f} | {gpu_t_str} | {gpu_u_str} | {gpu_p_str}"
        )

    lines.extend([
        "============================================================",
        "END OF TELEMETRY REPORT",
        "============================================================",
    ])

    report_text = "\n".join(lines) + "\n"

    # Write to target path
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    # If executed in thermal_ctrl or root, ensure both have a copy
    alt_paths = ["telemetry_report.txt", os.path.join("thermal_ctrl", "telemetry_report.txt")]
    for p in alt_paths:
        try:
            if os.path.abspath(p) != os.path.abspath(report_path):
                target_dir = os.path.dirname(p)
                if target_dir:
                    os.makedirs(target_dir, exist_ok=True)
                with open(p, "w", encoding="utf-8") as f:
                    f.write(report_text)
        except Exception:
            pass

    return report_text


def main():
    print("============================================================")
    print("INITIALIZING HARDWARE TELEMETRY SENSORS (CPU + GPU)")
    print("============================================================")
    reader = TelemetryReader(interval_s=1.0)
    print(f"Active CPU Temp Sensor: {reader.sensor_status.get('temperature_sensor')}")
    print(f"CPU Temp Resolution:   {reader.temp_resolution} C")
    print(f"Active GPU Sensor:      {reader.sensor_status.get('gpu_sensor')}")
    print(f"Active Power Plan:      {get_active_power_plan()}")
    print("\nStarting 10-second live telemetry stream (1 reading/second):")
    print(f"{'Time':<6} | {'CPU (C)':<8} | {'Freq (MHz)':<10} | {'Perf (%)':<9} | {'CPU (%)':<8} | {'GPU (C)':<8} | {'GPU (%)':<8} | {'GPU (W)':<8}")
    print("-" * 84)

    sample_readings = []
    # Prime initial psutil cpu_percent
    psutil.cpu_percent(interval=None)

    for i in range(10):
        time.sleep(1.0)
        data = reader.read()
        sample_readings.append(data)
        elapsed = i + 1
        gt = f"{data['gpu_temp_c']:6.1f}" if data.get('gpu_temp_c') is not None else "   N/A"
        gu = f"{data['gpu_util_percent']:6.1f}" if data.get('gpu_util_percent') is not None else "   N/A"
        gp = f"{data['gpu_power_w']:6.2f}" if data.get('gpu_power_w') is not None else "   N/A"
        print(
            f"{elapsed:<6} | {data['temp_c']:<8.2f} | {data['freq_mhz']:<10.1f} | {data['perf_percent']:<9.1f} | {data['cpu_util_percent']:<8.1f} | {gt:<8} | {gu:<8} | {gp:<8}"
        )

    reader.close()

    report_content = write_telemetry_report(reader, sample_readings, "telemetry_report.txt")
    print("\nSaved unified telemetry report to: telemetry_report.txt")
    print("============================================================\n")
    return report_content


if __name__ == "__main__":
    main()
