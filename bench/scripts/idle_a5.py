"""Summarise the idle-state columns recorded before each A5 row."""
import collections
import csv
import sys

rows = list(csv.DictReader(open(sys.argv[1])))
for dev in ("GPU", "CPU"):
    rr = [r for r in rows if r["device"] == dev]
    if not rr:
        continue
    util = collections.Counter(r["gpu_util_pct_before"] for r in rr)
    mem = [float(r["gpu_mem_mib_before"] or 0) for r in rr]
    la = [float(r["loadavg1_before"] or 0) for r in rr]
    print(f"{dev}: rows={len(rr)} hosts={set(r['host'] for r in rr)} gpu_util%={dict(util)} "
          f"gpu_mem_MiB[min,max]=[{min(mem):.0f},{max(mem):.0f}] loadavg1[min,max]=[{min(la):.2f},{max(la):.2f}]")
