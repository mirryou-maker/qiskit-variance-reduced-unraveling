"""Print per-shot cost (ms) of every method/device/config from an A5 CSV."""
import csv
import sys

rows = list(csv.DictReader(open(sys.argv[1])))
d = {}
for r in rows:
    d[(int(r["n"]), float(r["gt"]), r["method"], r["device"], r["config"])] = float(r["t_per_shot_s"]) * 1e3
cols = [("std", "GPU", "batched"), ("std", "GPU", "plain"), ("proj", "GPU", "plain"),
        ("std", "CPU", "plain"), ("proj", "CPU", "plain")]
print(" n   gt  | " + " ".join(f"{m}-{dv}-{c}"[:13].rjust(13) for m, dv, c in cols))
for n in sorted({k[0] for k in d}):
    for g in sorted({k[1] for k in d}):
        cells = [(f"{d[(n, g) + c]:13.4f}" if (n, g) + c in d else "            -") for c in cols]
        print(f"{n:2d} {g:5}  | " + " ".join(cells))
