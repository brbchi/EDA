"""
PHASE 2 -- k-Sweep: Find Optimal k for kNN Graph
Evaluates k in [8, 12, 16, 20, 24, 30] using a lightweight
2-layer EdgeConv (64 hidden units) with 5-fold drillhole-level CV.

Rules:
  - torch imported BEFORE pandas (Windows DLL constraint)
  - Folds split at DRILLHOLE level, stratified by Z_g_earth quartile
  - Only ss_encountered==1 holes in train/val splits
  - ALL 84990 intervals are graph nodes; ss_encountered==0 nodes
    pass messages only -- excluded from loss
  - Targets (depth_at_HW, depth_at_fw) standardised to zero-mean /
    unit-variance during training; inverse-transformed for evaluation
  - Huber loss (delta=1.0) weighted by interval length
  - Adam lr=0.001, max 100 epochs, early stopping patience=10

Outputs:
  outputs/k_sweep_summary.csv
  outputs/k_sweep_results.png
"""

import os, sys, random, warnings

# torch BEFORE pandas -- required on this system (Windows DLL order)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import EdgeConv

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

ROOT    = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(OUT_DIR, "df_model.csv"))
print(f"Loaded df_model: {df.shape}")

NODE_FEAT_COLS = [
    "X", "Y", "Z_g_earth", "MID_DEPTH", "Z_mid",
    "interval_length", "depth_ratio", "is_SS", "ss_encountered"
]

# ── 2. Standardise targets (fit only on ss_encountered==1 rows) ───────────────
enc1_mask = df["ss_encountered"].values == 1
raw_y     = df[["depth_at_HW", "depth_at_fw"]].fillna(0).values.astype(np.float32)

tgt_scaler = StandardScaler()
tgt_scaler.fit(raw_y[enc1_mask])          # fit on labelled holes only

y_scaled   = tgt_scaler.transform(raw_y)  # apply to all rows (enc0 rows unused)
y_tensor   = torch.tensor(y_scaled, dtype=torch.float32)

# Original-scale targets kept for evaluation
y_orig     = raw_y.copy()

print(f"  Target means (orig) : HW={raw_y[enc1_mask,0].mean():.1f}m  "
      f"FW={raw_y[enc1_mask,1].mean():.1f}m")
print(f"  Target stds  (orig) : HW={raw_y[enc1_mask,0].std():.1f}m  "
      f"FW={raw_y[enc1_mask,1].std():.1f}m")

# Node features
X_feat   = torch.tensor(df[NODE_FEAT_COLS].values, dtype=torch.float32)

# Interval-length weights (normalised)
raw_len  = (df["Depth_To"] - df["Depth_From"]).values.astype(np.float32)
wt       = torch.tensor(raw_len / raw_len.mean(), dtype=torch.float32)

# 3-D spatial coords for kNN (scaled X, Y, Z_mid)
xyz      = df[["X", "Y", "Z_mid"]].values.astype(np.float32)
hole_ids = df["Drillhole"].values

print(f"  Node features : {X_feat.shape}")
print(f"  Targets       : {y_tensor.shape}  (standardised)")
print(f"  ss_enc==1 rows: {enc1_mask.sum()}")

# ── 3. 5-fold drillhole-level CV stratified by Z_g_earth ─────────────────────
hole_df = (
    df[df["ss_encountered"] == 1]
    .groupby("Drillhole")
    .agg(Z_g=("Z_g_earth_orig", "first"))
    .reset_index()
)
hole_df["strat"] = pd.qcut(hole_df["Z_g"], q=4, labels=False, duplicates="drop")

skf     = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
folds   = list(skf.split(hole_df["Drillhole"], hole_df["strat"]))
fold_of = {}
for fi, (_, vi) in enumerate(folds):
    for h in hole_df.iloc[vi]["Drillhole"]:
        fold_of[h] = fi

interval_fold = np.array([fold_of.get(h, -1) for h in hole_ids])

print(f"\n5-fold CV over {len(hole_df)} ss_encountered=1 holes")
for f in range(5):
    nh = sum(1 for _, fv in fold_of.items() if fv == f)
    ni = (interval_fold == f).sum()
    print(f"  Fold {f}: {nh} holes, {ni} intervals")

# ── 4. Graph builder ──────────────────────────────────────────────────────────

def build_graph(k: int) -> Data:
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree",
                            n_jobs=-1).fit(xyz)
    dists, idxs = nbrs.kneighbors(xyz)
    dists = dists[:, 1:]; idxs = idxs[:, 1:]

    N   = len(xyz)
    src = np.repeat(np.arange(N), k)
    dst = idxs.ravel()
    ew  = (1.0 / (dists.ravel() + 1e-8)).astype(np.float32)

    return Data(
        x          = X_feat,
        edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long),
        edge_attr  = torch.tensor(ew, dtype=torch.float32),
        y          = y_tensor,          # standardised targets
    )

# ── 5. Lightweight GNN for sweep (64 hidden units) ───────────────────────────

class SweepGNN(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.conv1 = EdgeConv(nn.Sequential(
            nn.Linear(2 * in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),     nn.ReLU(),
        ), aggr="mean")
        self.conv2 = EdgeConv(nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),     nn.ReLU(),
        ), aggr="mean")
        self.drop = nn.Dropout(0.2)
        self.head = nn.Linear(hidden, 2)

    def forward(self, x, edge_index):
        h = self.drop(F.relu(self.conv1(x, edge_index)))
        h = self.drop(F.relu(self.conv2(h, edge_index)))
        return self.head(h)

# ── 6. Weighted Huber loss (on standardised scale, delta=1.0) ─────────────────

def huber_wt(pred, target, weight, delta=1.0):
    e = pred - target
    l = torch.where(e.abs() <= delta,
                    0.5 * e ** 2,
                    delta * (e.abs() - 0.5 * delta))
    return (l * weight.unsqueeze(1)).mean()

# ── 7. Train one fold ─────────────────────────────────────────────────────────

def train_fold(graph, tr_mask, vl_mask):
    torch.manual_seed(SEED)
    model = SweepGNN(in_dim=len(NODE_FEAT_COLS))
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_loss, best_w, wait = float("inf"), None, 0

    for ep in range(100):
        model.train()
        opt.zero_grad()
        pred = model(graph.x, graph.edge_index)
        loss = huber_wt(pred[tr_mask], graph.y[tr_mask], wt[tr_mask])
        loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            vl = huber_wt(
                model(graph.x, graph.edge_index)[vl_mask],
                graph.y[vl_mask], wt[vl_mask]
            ).item()

        if vl < best_loss - 1e-6:
            best_loss = vl
            best_w    = {k: v.clone() for k, v in model.state_dict().items()}
            wait      = 0
        else:
            wait += 1
            if wait >= 10:
                break

    model.load_state_dict(best_w)
    return model, ep + 1

# ── 8. Evaluate one fold (inverse-transform -> original metres) ───────────────

def eval_fold(model, graph, vl_mask):
    model.eval()
    with torch.no_grad():
        p_scaled = model(graph.x, graph.edge_index).numpy()

    # Inverse-transform to original depth scale
    p_orig = tgt_scaler.inverse_transform(p_scaled)

    idx  = vl_mask.numpy()
    hl   = pd.DataFrame({
        "Drillhole": hole_ids[idx],
        "p_hw": p_orig[idx, 0], "p_fw": p_orig[idx, 1],
        "t_hw": y_orig[idx, 0], "t_fw": y_orig[idx, 1],
    }).groupby("Drillhole").mean()

    m = {}
    for tgt, col in [("HW", "hw"), ("FW", "fw")]:
        pv, tv = hl[f"p_{col}"].values, hl[f"t_{col}"].values
        m[f"R2_{tgt}"]   = r2_score(tv, pv)
        m[f"RMSE_{tgt}"] = float(np.sqrt(mean_squared_error(tv, pv)))
        m[f"MAE_{tgt}"]  = float(mean_absolute_error(tv, pv))
    return m

# ── 9. k-sweep ────────────────────────────────────────────────────────────────
K_VALUES = [8, 12, 16, 20, 24, 30]
rows     = []

for k in K_VALUES:
    print(f"\n{'='*55}\n  k = {k}\n{'='*55}")
    g = build_graph(k)
    print(f"  Graph: {g.num_nodes} nodes, {g.num_edges} edges")

    fold_rows = []
    for fold in range(5):
        val_h  = [h for h, fv in fold_of.items() if fv == fold]
        vl_msk = torch.tensor(
            np.isin(hole_ids, val_h) & enc1_mask, dtype=torch.bool)
        tr_msk = torch.tensor(
            (~np.isin(hole_ids, val_h)) & enc1_mask, dtype=torch.bool)

        model, n_ep = train_fold(g, tr_msk, vl_msk)
        m = eval_fold(model, g, vl_msk)
        m.update({"fold": fold, "k": k, "epochs": n_ep})
        fold_rows.append(m)
        print(f"  Fold {fold}: R2_HW={m['R2_HW']:.3f}  RMSE_HW={m['RMSE_HW']:.1f}m  "
              f"R2_FW={m['R2_FW']:.3f}  RMSE_FW={m['RMSE_FW']:.1f}m  (ep={n_ep})")

    fd   = pd.DataFrame(fold_rows)
    mean = fd[["R2_HW","RMSE_HW","MAE_HW","R2_FW","RMSE_FW","MAE_FW"]].mean()
    mean["k"] = k; mean["fold"] = "mean"; mean["epochs"] = fd["epochs"].mean()
    rows.append(mean.to_dict())
    print(f"  MEAN: R2_HW={mean['R2_HW']:.3f}  RMSE_HW={mean['RMSE_HW']:.1f}m  "
          f"R2_FW={mean['R2_FW']:.3f}  RMSE_FW={mean['RMSE_FW']:.1f}m")

# ── 10. Summary table + optimal k ────────────────────────────────────────────
res = pd.DataFrame(rows)[
    ["k","R2_HW","RMSE_HW","MAE_HW","R2_FW","RMSE_FW","MAE_FW","epochs"]
].round(4)
res["RMSE_combined"] = res["RMSE_HW"] + res["RMSE_FW"]
opt_k = int(res.loc[res["RMSE_combined"].idxmin(), "k"])

res.to_csv(os.path.join(OUT_DIR, "k_sweep_summary.csv"), index=False)

print("\n" + "="*65)
print("  k-SWEEP SUMMARY  (5-fold CV mean, original depth scale)")
print("="*65)
print(res.drop(columns="RMSE_combined").to_string(index=False))
print(f"\n  >>> Optimal k = {opt_k}  (lowest RMSE_HW + RMSE_FW)")

# ── 11. Plot ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("k-Sweep: 5-Fold CV Performance  (2-layer EdgeConv, 64 hidden units)",
             fontsize=11, fontweight="bold")

for ax, (col, lbl) in zip(axes, [
    ("RMSE_HW", "RMSE  HW Depth (m)"),
    ("RMSE_FW", "RMSE  FW Depth (m)"),
    ("R2_HW",   "R2  HW Depth"),
]):
    ax.plot(res["k"], res[col], "o-", color="#2563eb", lw=2, ms=7)
    ax.axvline(opt_k, color="#dc2626", ls="--", lw=1.5, label=f"Optimal k={opt_k}")
    ax.set_xlabel("k  (neighbours)"); ax.set_ylabel(lbl); ax.set_title(lbl)
    ax.set_xticks(K_VALUES); ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "k_sweep_results.png"), dpi=150, bbox_inches="tight")
plt.close()

print(f"\nSaved: outputs/k_sweep_summary.csv")
print(f"Saved: outputs/k_sweep_results.png")
print("\n=== PHASE 2 COMPLETE ===")
print(f"Use k = {opt_k} in Phase 3.")
