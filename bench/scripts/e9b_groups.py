"""E9b group list: every non-deterministic E9a TN group at n = 20 (pre-registered; no selection)."""
import os

import pandas as pd

R = os.path.join(os.path.dirname(__file__), "..", "..", "bench/results")
S = pd.read_csv(os.path.join(R, "b_e9a_summary.csv"))
g = S[(~S.deterministic) & (S.n == 20) & S.TN][["channel", "family", "variant", "n", "L", "gt", "obs", "traj_mean",
                                                 "var_std"]]
g.to_csv(os.path.join(R, "b_e9b_groups.csv"), index=False)
print(len(g), "groups,", len(g.groupby(["channel", "family", "variant", "L", "gt"])), "configs")
