import os, sys, time, csv
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def burn(stop_at):
    import numpy as np
    a = np.random.rand(256, 256)
    b = np.random.rand(256, 256)
    while time.time() < stop_at:
        a = a @ b
        a /= a.max()


if __name__ == "__main__":
    from telemetry import TelemetryReader
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    n = os.cpu_count()
    stop_at = time.time() + duration
    procs = [mp.Process(target=burn, args=(stop_at,)) for _ in range(n)]
    for p in procs:
        p.start()
    tel = TelemetryReader(interval_s=1.0)
    rows = []
    try:
        while time.time() < stop_at:
            t0 = time.time()
            d = tel.read()
            rows.append(d)
            if len(rows) % 10 == 0:
                print(f"{len(rows):4d}s temp {d.get('temp_c')} perf {d.get('perf_percent')} "
                      f"limit {d.get('perf_limit_flag')}")
            time.sleep(max(0, 1 - (time.time() - t0)))
    finally:
        for p in procs:
            p.terminate()
        tel.close()
        with open("stress_telemetry.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        temps = [r["temp_c"] for r in rows]
        print("max temp", max(temps), "| min perf", min(r["perf_percent"] for r in rows),
              "| limit flag ever on:", any(r["perf_limit_flag"] > 0 for r in rows))