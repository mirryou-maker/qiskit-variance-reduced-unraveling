"""E9c (docs/plan_B_characterization_2026-11-06.md, "E9c pre-registration"): extend sparse truncated Pauli propagation
to larger M on the E9a groups that were classified 'needs trajectories' (TN), so the TN fraction can be reported as a
function of the M cap instead of at the single cap M <= 1e5.

Groups: E9a TN groups (bench/results/b_e9a_summary.csv, TN == True, not deterministic) at n in --n (default 12 20 24).
Rows (ledger CSV, resumable; same fields as b_e9a.csv plus status):
  kind=pp  M, value, peak, wall_s (GPU synced), status ok | oom   (after an oom the larger M of that group are skipped)
Work split: --part / --nparts over configs (deterministic round robin on the cost-sorted list).

E9d mode ("E9d pre-registration"): --summary bench/results/b_e9c_summary.csv --tncol TN3000000 --M 100000000 --big
uses pauli_prop.propagate_big (chunked expansion, host-RAM buckets; identical truncation) and adds the column uniq
(largest number of distinct terms after a merge, before truncation).
"""
import argparse
import csv
import datetime as dt
import os
import socket
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from src.noise import pauli_prop as P  # noqa: E402
from src.noise import traj_kraus as K  # noqa: E402
from src.noise.traj_ref import GPU, xp  # noqa: E402

FIELDS = ["channel", "family", "variant", "n", "L", "gt", "obs", "kind", "M", "value", "peak", "wall_s", "status",
          "host", "gpu", "gpu_other", "load1", "timestamp"]


def gpu_other():
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout
    except Exception as e:  # noqa: BLE001
        return f"err:{e}"
    me = str(os.getpid())
    others = [l.strip() for l in out.splitlines() if l.strip() and l.split(",")[0].strip() != me]
    return ";".join(others) if others else "none"


def gpu_name():
    if not GPU:
        return "cpu"
    try:
        return xp.cuda.runtime.getDeviceProperties(xp.cuda.Device().id)["name"].decode()
    except Exception:  # noqa: BLE001
        return "gpu"


def free():
    if GPU:
        xp.get_default_memory_pool().free_all_blocks()


def groups(summary, ns, tncol="TN"):
    cfg = {}
    for r in csv.DictReader(open(summary)):
        if r["deterministic"] == "True" or r[tncol] not in ("True", "1", "1.0") or int(r["n"]) not in ns:
            continue
        k = (r["channel"], r["family"], r["variant"], int(r["n"]), int(r["L"]), float(r["gt"]))
        cfg.setdefault(k, []).append(r["obs"])
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=os.path.join(ROOT, "bench/results/b_e9a_summary.csv"))
    ap.add_argument("--n", type=int, nargs="+", default=[12, 20, 24])
    ap.add_argument("--M", type=int, nargs="+", default=[3_000_000, 10_000_000, 30_000_000])
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="first k configs (smoke)")
    ap.add_argument("--out", default=os.path.join(ROOT, "bench/results/b_e9c.csv"))
    ap.add_argument("--tncol", default="TN", help="summary column marking the groups to run")
    ap.add_argument("--big", action="store_true", help="use propagate_big (E9d)")
    ap.add_argument("--chunk", type=int, default=10_000_000)
    ap.add_argument("--nb", type=int, default=64)
    a = ap.parse_args()
    if a.big:
        FIELDS.insert(FIELDS.index("peak") + 1, "uniq")
    cfg = groups(a.summary, set(a.n), a.tncol)
    order = sorted(cfg, key=lambda k: -(k[4] * len(cfg[k]) * (2 if k[3] >= 20 else 1)))
    todo = order[a.part::a.nparts][:a.limit]
    done, oom = set(), set()
    if os.path.exists(a.out):
        for r in csv.DictReader(open(a.out)):
            g = (r["channel"], r["family"], r["variant"], int(r["n"]), int(r["L"]), float(r["gt"]), r["obs"])
            done.add(g + (int(r["M"]),))
            if r["status"] == "oom":
                oom.add(g)
    new = not os.path.exists(a.out)
    f = open(a.out, "a", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new:
        w.writeheader()
    host, gname = socket.gethostname(), gpu_name()
    print(f"part {a.part}/{a.nparts}: {len(todo)} configs, "
          f"{sum(len(cfg[k]) for k in todo)} groups on {gname}", flush=True)
    for k in todo:
        ch, fam, var, n, L, gt = k
        ops = K.build(fam, var, n, L, gt, ch)
        tables = {}
        t0 = time.time()
        base = dict(channel=ch, family=fam, variant=var, n=n, L=L, gt=gt, host=host, gpu=gname,
                    gpu_other=gpu_other(), load1=round(os.getloadavg()[0], 2) if hasattr(os, "getloadavg") else "")
        for o in cfg[k]:
            g = k + (o,)
            for m in sorted(a.M):
                if g + (m,) in done or g in oom:
                    continue
                free()
                P.sync()
                t1 = time.perf_counter()
                try:
                    st = {}
                    if a.big:
                        v, peak = P.propagate_big(ops, n, o, m, tables, chunk=a.chunk, nb=a.nb, stats=st)
                    else:
                        v, peak = P.propagate(ops, n, o, m, tables)
                    P.sync()
                    row = dict(base, obs=o, kind="pp", M=m, value=v, peak=peak,
                               wall_s=round(time.perf_counter() - t1, 3), status="ok")
                    if a.big:
                        row["uniq"] = st.get("peak_unique", "")
                except Exception as e:  # noqa: BLE001
                    if ("OutOfMemory" not in type(e).__name__ and "out of memory" not in str(e).lower()
                            and not isinstance(e, MemoryError)):
                        raise
                    free()
                    oom.add(g)
                    row = dict(base, obs=o, kind="pp", M=m, value="", peak="",
                               wall_s=round(time.perf_counter() - t1, 3), status="oom")
                row["timestamp"] = dt.datetime.now().isoformat(timespec="seconds")
                w.writerow(row)
                f.flush()
        print(f"{dt.datetime.now().isoformat(timespec='seconds')} {ch}_{fam}_{var}_n{n}_L{L}_gt{gt} "
              f"obs={','.join(cfg[k])} ({time.time() - t0:.0f}s) other={base['gpu_other']}", flush=True)
    f.close()


if __name__ == "__main__":
    main()
