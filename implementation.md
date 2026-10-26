# Implementation Plan: Cross-Loaded Thermal & Power Awareness Pipeline

## Executive Overview
This implementation plan establishes a reproducible, paper-grade evaluation pipeline for the **Thermal-Aware Background Workload Controller**. It integrates GPU telemetry, cross-loading (local LLM on NVIDIA RTX 5060 + heavy CPU batch workers), calibrated hardware thresholds, and policy comparisons to scientifically verify **M3** (thermal trend ladder) and **M6** (hybrid core placement + power/perf cap).

---

## 1. Hardware Context & Workload Scaling

### Is 32 Workers Enough for a 24-Core / 32-Thread Beast?
- **Current Limitation:** In previous tests, 32 workers were doing tiny $256 \times 256$ matrix multiplications ($512\text{ KB}$ per matrix). Because this fits entirely in L2/L3 cache, the memory subsystem and memory controller slept, drawing only ~45W (within the laptop's passive fan dissipation headroom at 65°C).
- **The Upgrade Needed:**
  1. **AVX Vector Saturation ($1024 \times 1024$ chunks):** Scaling matrix dimensions from $256$ to $1024$ forces continuous FMA (Fused Multiply-Add) vector unit execution, saturating the L3 cache and memory buses to push the CPU package to its maximum PL2/PL1 limit.
  2. **Over-subscription Testing:** Evaluating at $N = 32$ (100% capacity) and $N = 48$ ($1.5\times$ over-subscription) to test worst-case thread scheduling contention.
  3. **Combined CPU + GPU Cross-Loading:** The true bottleneck occurs when the **RTX 5060 draws ~85W–100W** running Ollama (`qwen2.5:7b-instruct` or `llama3.1`) while the CPU pulls ~55W+. Squeezing **>140W total thermal dissipation** into a 20mm chassis pushes heatsink temperatures to $85^\circ\text{C}-95^\circ\text{C}+$, forcing thermal and power limits to trigger.

---

## 2. Phase-by-Phase Checklist

### [x] Phase 0: Unified Telemetry Engine (CPU + GPU)
- [x] Update `thermal_ctrl/telemetry.py` to add non-blocking GPU metric sampling:
  - `gpu_temp_c`, `gpu_util_percent`, `gpu_power_w` via NVML (ctypes) and `nvidia-smi` fallback.
  - Graceful fallback for non-NVIDIA / integrated-only machines (`gpu_available: false`).
- [x] Unit verification: Verified live 10s stream captures both CPU (Temp, Freq, Perf%) and GPU (Temp, Util%, Watts) concurrently.

### [ ] Phase 1: Device Calibration Pipeline (Zero-Dependency)
- [ ] Ensure `thermal_ctrl/calibrate.py` remains 100% portable:
  - Requires **no external downloads** (no Ollama models required to calibrate a new laptop).
  - Samples idle baselines ($T_{cpu\_idle}, T_{gpu\_idle}, \text{Perf}_{baseline}$).
  - Ramps workers (1 to 32) using upgraded high-intensity matrix math.
  - Automatically identifies throttle type (`power`, `thermal`, `both`, or `none`).
  - Writes dual thresholds to `config.json` (`T_warn`, `T_hot`, `perf_warn_threshold_percent`, `perf_hot_threshold_percent`).
  - Generates 3-panel plot (`results/calibration.png`).
- [ ] Verification: Run calibration test and verify `config.json` schema and realistic values.

### [ ] Phase 2: Dual-Mode Foreground Workload Engine
- [ ] Create `LLMStreamProbe` in `thermal_ctrl/workloads.py`:
  - **Mode A (Ollama Real-World):** Connects to `http://localhost:11434/api/generate`. Streams prompts to `qwen2.5:7b-instruct` or `llama3.1:latest`.
    - Measures: `tokens_per_second` (eval rate), `time_to_first_token_ms` (TTFT), and inter-token latency percentiles (`p50`, `p95`, `p99`).
  - **Mode B (Zero-Dependency Synthetic GPU Burner):** Pure Python fallback generating high-throughput tensor operations on the GPU to draw equivalent ~85W wattage for systems without Ollama.
- [ ] Verification: Test `LLMStreamProbe` streams tokens, records millisecond timestamps, and reports metrics cleanly.

### [ ] Phase 3: Failure Mode Characterization (Phase-1 Experiment)
- [ ] Update `run_experiment.py` / `campaign.py` to execute a 3-way isolation test:
  - **Case A:** LLM alone (uncontended baseline).
  - **Case B:** LLM + Background CPU on remote cores (thermal & power cross-loading only, zero core contention).
  - **Case C:** LLM + Background CPU sharing core 0 (thermal & power + core contention).
- [ ] Verification: Produce `results_llm/phase1_table.csv` demonstrating the isolated contribution of thermal vs scheduler contention.

### [ ] Phase 4: Controller Evaluation Campaign
- [ ] Update `thermal_ctrl/campaign.py` to execute randomized trials with **inter-run cooldown gating**:
  - Gated on: $T_{CPU} \le \text{cooldown\_target}$ and $T_{GPU} \le 50^\circ\text{C}$.
  - Policies to compare:
    - **M0:** Baseline (uncontrolled 32 workers).
    - **M1:** Windows `BELOW_NORMAL` priority.
    - **M2:** Static 40% cap at $T_{hot}$.
    - **M3:** Trend-aware cap ladder ($dT/dt$).
    - **M6:** Hybrid: Placement (pinned off cores 0 & 1) + Power/perf cap.
- [ ] Verification: Run campaign across M0, M3, M6 with full telemetry, probe, and controller logs.

### [ ] Phase 5: Academic Reporting & Paper Artifacts
- [ ] Update `thermal_ctrl/analyze.py` to produce publishable figures:
  - **Figure 1:** Token Generation Latency CDF & Jitter Comparison (M0 vs M3 vs M6).
  - **Figure 2:** Time-series of CPU/GPU Temps, Delivered Perf %, and Job Object Cap.
  - **Figure 3:** Pareto Trade-off Frontier (LLM p95 latency vs. Background Compute Throughput).
  - **Summary Table:** Mean tokens/sec, p95 token delay, max CPU temp, and background work done.

---

## 3. Acceptance Criteria
- [ ] **Sensor non-blocking invariant:** GPU telemetry read must take $< 15\text{ms}$ or run in a background daemon thread to avoid distorting CPU sampling intervals.
- [ ] **Portability invariant:** Running on a laptop without NVIDIA GPU or without Ollama must gracefully run CPU-only baseline tests without crashing.
- [ ] **Evidence of M3/M6 Efficacy:**
  - M3 must show lower peak temperature than M0.
  - M6 must show significantly lower p95/p99 token latency and higher tokens/sec than M0 under identical heavy cross-load.
