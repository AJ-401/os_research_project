Listed directory results_m6
Viewed campaign_meta.json:1-45

### Gravitas Policy Record

```json
{
  "mode": "plan",
  "planning": "deep",
  "context_depth": "system-wide",
  "verification_depth": "impact-and-academic-rigor",
  "allowed_write_scope": ["thermal_ctrl/*"],
  "delegation": "none",
  "reason_codes": [
    "academic-workflow-standardization",
    "dual-mode-gpu-pipeline",
    "m3-m6-controller-justification"
  ]
}
```

---

### 1. Cross-Verification of Your Proposed Workflow

You proposed:
$$\text{Telemetry for both CPU & GPU works} \longrightarrow \text{Calibrating the device} \longrightarrow \text{Testing our controllers}$$

**Is this correct?**
Yes, it is the correct core foundation. However, to publish this in an **academic OS/Systems paper** and cleanly defend `M3` and `M6`, **two essential stages are missing**:

```
Your Proposal:
[ Telemetry ] ──────────────────> [ Calibration ] ─────────────────────────> [ Controller Tests ]

Paper-Grade Systems Pipeline (Recommended):
[ Phase 0: Telemetry & Sensors ]
       │
       ▼
[ Phase 1: Hardware Calibration ] (Dual: Standalone CPU + Cross-Loaded)
       │
       ▼
[ Phase 2: Failure Characterization ] (The "Proof of Contention & Throttling")
       │  ├── Case A: Foreground alone (uncontended baseline)
       │  ├── Case B: Shared heatsink cross-throttling (Thermal & Power coupling)
       │  └── Case C: Core contention (Scheduler starvation)
       ▼
[ Phase 3: Randomized Control Campaign ] (M0, M1, M2, M3, M6 with thermal cooldown gating)
       │
       ▼
[ Phase 4: Pareto Trade-off & Paper Artifacts ] (QoS preservation vs Batch throughput)
```

#### What Was Missing and Why It Matters for Your Paper:

1. **Phase 2: Characterization / Failure Mode Isolation:**
   - In systems research, reviewers will ask: _"How do you know the LLM slowed down because of thermal/power throttling rather than CPU core contention?"_
   - Phase 2 runs 3 quick diagnostic cases before the controllers to decouple **core contention** from **shared thermal/power limits**. This directly provides the theoretical justification for why **M6 (Affinity + Power Cap)** is necessary.
2. **Inter-Run Thermal Cooldown Gating:**
   - Without an explicit cooldown gate between tests, Run #2 inherits residual heat from Run #1, invalidating comparisons. Every run must be gated on $T_{CPU} \le T_{cooldown}$ and $T_{GPU} \le T_{cooldown}$.
3. **Dual Pipeline for Calibration vs Evaluation:**
   - As you noted: **Calibration must remain zero-dependency** (a new laptop without Ollama should still be able to calibrate). The heavy LLM workload is an _evaluation benchmark_, not a prerequisite for basic calibration.

---

### 2. The Implementation Plan

#### Phase 0: Unified CPU + GPU Telemetry Engine

- **Target File:** [thermal_ctrl/telemetry.py](file:///d:/OS%20Project/Thermal_Awareness/thermal_ctrl/telemetry.py)
- **Goal:** Enable 1 Hz synchronized sampling of both CPU and GPU metrics without hanging or blocking.
- **Specifications:**
  - **CPU Metrics:** `temp_c`, `freq_mhz`, `perf_percent`, `perf_limit_flag`, `cpu_util_percent`.
  - **GPU Metrics:** Detects if an NVIDIA GPU is present via `C:\Windows\System32\nvidia-smi.exe`. If present, queries:
    - `gpu_temp_c` (temperature in °C)
    - `gpu_util_percent` (GPU compute core load %)
    - `gpu_power_w` (board power draw in Watts)
  - **Graceful Fallback:** If no dedicated GPU exists, logs `gpu_available: false` and proceeds normally.

#### Phase 1: Portable Calibration Pipeline ([calibrate.py](file:///d:/OS%20Project/Thermal_Awareness/thermal_ctrl/calibrate.py))

- **Requirement:** Zero external downloads. No Ollama, no PyTorch, no multi-gigabyte models.
- **Mechanism:**
  - Measures idle baseline for both CPU and GPU ($T_{cpu\_idle}, T_{gpu\_idle}, \text{Perf}_{baseline}$).
  - Runs the step ramp of 32 matrix workers.
  - Detects throttle onset:
    - **Thermal Knee:** Temperature $> 75^\circ\text{C}$ with frequency drop.
    - **Power Knee:** `perf_percent` drops $\ge 8\%$ while temperatures are still moderate.
  - Outputs calibrated thresholds into [config.json](file:///d:/OS%20Project/Thermal_Awareness/config.json):
    - `T_warn`, `T_hot`, `cooldown_target`
    - `perf_warn_threshold_percent`, `perf_hot_threshold_percent`
    - `throttle_type` (`"power"`, `"thermal"`, or `"both"`)

#### Phase 2 & 3: Workload Engine & Dual LLM Probe ([workloads.py](file:///d:/OS%20Project/Thermal_Awareness/thermal_ctrl/workloads.py))

We introduce `LLMStreamProbe` in addition to the standard micro-probe:

1. **Ollama Real-World Mode (Active on your machine):**
   - Connects to `http://localhost:11434/api/generate`.
   - Uses your installed `qwen2.5:7b-instruct` or `llama3.1:latest`.
   - Sends continuous prompt streams requesting long generative text (e.g. 512 tokens).
   - Samples streaming chunks to record:
     - `tokens_per_second` (eval rate)
     - `time_to_first_token_ms` (TTFT)
     - `inter_token_latency_p50`, `p95`, `p99` (ms)
2. **Synthetic GPU Tensor Burner (Portable Pipeline Mode):**
   - For a new laptop without Ollama installed, a standalone GPU matrix-multiplication worker (using pure Python with standard Windows DirectX/DirectML or OpenCL/CUDA API via ctypes) that draws equivalent GPU wattage (~80W) to simulate the exact thermal footprint of an LLM.

#### Phase 4: Controller Verification Campaign ([campaign.py](file:///d:/OS%20Project/Thermal_Awareness/thermal_ctrl/campaign.py))

We evaluate the policies that matter for your paper:

| Policy | Technique                                       | How It Fixes Throttling                                                                                                                                          |
| :----- | :---------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **M0** | Baseline (No Control)                           | Background burns 100% on all 32 cores. Demonstrates system collapse.                                                                                             |
| **M1** | Windows `BELOW_NORMAL` Priority                 | Shows why OS priority classes fail when the bottleneck is shared hardware wattage.                                                                               |
| **M2** | Fixed Thermal Threshold (40% cap at $T_{hot}$)  | Reactive step control.                                                                                                                                           |
| **M3** | **Trend-Aware Cap Ladder**                      | Predicts thermal slope ($dT/dt$) and dials back CPU before $T_{hot}$ is reached.                                                                                 |
| **M6** | **Hybrid: Core Placement + Delivered Perf Cap** | **Pinning:** Background off CPUs 0 & 1 $\to$ guarantees LLM host thread never starves.<br/>**Power Cap:** Steps cap down whenever `perf_percent` drops $< 95\%$. |

Each test runs for **120 seconds**, preceded by an automatic cooldown loop back to `cooldown_target`.

#### Phase 5: Academic Reporting & Visualization ([analyze.py](file:///d:/OS%20Project/Thermal_Awareness/thermal_ctrl/analyze.py))

Generates publishable figures:

1. **Figure 1 (LLM Responsiveness CDF / Boxplot):** Inter-token latency and tokens/sec under M0 vs M3 vs M6.
2. **Figure 2 (Thermal & Power Co-Evolution):** Dual-axis timeline showing CPU Temp, GPU Temp, CPU Perf %, and Background Job Object Cap.
3. **Figure 3 (Pareto Frontier):** LLM p95 latency (ms) vs. Background Compute Throughput. Shows M6 sitting at the optimal Pareto frontier.

---

### Step-by-Step Execution Sequence

1. **Step 1:** Enhance [telemetry.py](file:///d:/OS%20Project/Thermal_Awareness/thermal_ctrl/telemetry.py) to read GPU metrics via `nvidia-smi` alongside CPU PDH counters.
2. **Step 2:** Add `LLMStreamProbe` to [workloads.py](file:///d:/OS%20Project/Thermal_Awareness/thermal_ctrl/workloads.py) (supporting both Ollama local models and synthetic GPU burner).
3. **Step 3:** Run `calibrate.py` to produce a fresh [config.json](file:///d:/OS%20Project/Thermal_Awareness/config.json) with dual power & thermal thresholds.
4. **Step 4:** Run a focused campaign (`M0`, `M3`, `M6`) with the LLM probe.
5. **Step 5:** Generate the summary graphs and analysis tables for your paper.

Shall we begin with **Step 1** (GPU telemetry integration in `telemetry.py`)?
