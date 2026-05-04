"""
PHASE 2b -- k-Sweep: Drillhole-Level Nodes + Full Interval Encoding

Each drillhole is a single graph node whose features combine three groups:
  1. Spatial/geometric  : X, Y, Z_g_earth (scaled), hole_length          [4]
  2. Lithological summary: n_intervals, ss_proportion, n_ss_intervals,
                           mean_interval_length, std_interval_length,
                           ss_encountered                                  [6]
  3. Sequence embedding  : BiLSTM(in=5, hidden=32, bidir) over depth-
                           sorted intervals -> 64-dim                     [64]
  Total node dim: 74

  Graph  : kNN on (X_orig, Y_orig, Z_g_earth_orig) collar coords
  GNN    : 2-layer EdgeConv (128 hidden) + LayerNorm + residual (paper-aligned)
  k vals : [4, 6, 8, 10, 12, 16]

Rules:
  - Folds split at drillhole level, stratified by Z_g_earth quartile
  - Only ss_encountered==1 holes contribute to loss
  - ss_encountered==0 holes remain as graph nodes (message passing only)
  - Huber loss (delta=1.0), Adam lr=0.001 (β1=0.9, β2=0.99), max 150 epochs, patience=15
  - Targets standardised during training, inverse-transformed for eval

Outputs:
  outputs/k_sweep_b_summary.csv
  outputs/k_sweep_b_results.png
"""

import os, random, warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import EdgeConv
from torch.nn.utils.rnn import pack_padded_sequence

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

# ── 1. Load interval data ─────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(OUT_DIR, "df_model.csv"))
print(f"Loaded df_model: {df.shape}")

# ── 2. Per-interval features (interval-specific only) ─────────────────────────
# X, Y, Z_g_earth are collar-level (identical for all intervals in a hole)
# ss_encountered is hole-level → excluded from interval features
INTERVAL_FEAT_COLS = ["MID_DEPTH", "Z_mid", "interval_length", "depth_ratio", "is_SS"]
N_INTERVAL_FEAT    = len(INTERVAL_FEAT_COLS)

# ── 3. Build per-hole records (sequences + summary stats) ────────────────────
hole_records = []
for hole, grp in df.groupby("Drillhole"):
    grp     = grp.sort_values("Depth_From")
    raw_il  = (grp["Depth_To"] - grp["Depth_From"]).values.astype(np.float32)
    hole_records.append({
        # identifiers / targets
        "Drillhole"      : hole,
        "X_orig"         : grp["X_orig"].iloc[0],
        "Y_orig"         : grp["Y_orig"].iloc[0],
        "Z_g_earth_orig" : grp["Z_g_earth_orig"].iloc[0],
        "ss_encountered" : int(grp["ss_encountered"].iloc[0]),
        "depth_at_HW"    : grp["depth_at_HW"].iloc[0],
        "depth_at_fw"    : grp["depth_at_fw"].iloc[0],
        # ── group 1: spatial/geometric (phase-1 scaled) ──────────────────
        "X"              : grp["X"].iloc[0],
        "Y"              : grp["Y"].iloc[0],
        "Z_g_earth"      : grp["Z_g_earth"].iloc[0],
        "hole_length"    : float(grp["Depth_To"].max() - grp["Depth_From"].min()),
        # ── group 2: lithological profile summary ────────────────────────
        "n_intervals"       : len(grp),
        "ss_proportion"     : float(grp["is_SS"].mean()),
        "n_ss_intervals"    : int(grp["is_SS"].sum()),
        "mean_interval_len" : float(raw_il.mean()),
        "std_interval_len"  : float(raw_il.std()) if len(raw_il) > 1 else 0.0,
        # ── group 3: sequence (for BiLSTM) ───────────────────────────────
        "seq"            : grp[INTERVAL_FEAT_COLS].values.astype(np.float32),
    })

sequences = [r.pop("seq") for r in hole_records]
hole_df   = pd.DataFrame(hole_records)
N_HOLES   = len(hole_df)

seq_lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long)
print(f"Total drillholes : {N_HOLES}")
print(f"  ss_encountered=1 : {(hole_df['ss_encountered']==1).sum()}")
print(f"  ss_encountered=0 : {(hole_df['ss_encountered']==0).sum()}")
print(f"Sequence lengths : min={seq_lengths.min().item()}  "
      f"max={seq_lengths.max().item()}  mean={seq_lengths.float().mean().item():.1f}")

# Pad sequences -> [N_HOLES, max_len, N_INTERVAL_FEAT]
max_len    = int(seq_lengths.max().item())
seq_padded = torch.zeros(N_HOLES, max_len, N_INTERVAL_FEAT)
for i, s in enumerate(sequences):
    seq_padded[i, :len(s)] = torch.tensor(s)

# ── 4. Standardise targets (fit on ss_encountered==1 holes only) ──────────────
enc1_mask = hole_df["ss_encountered"].values == 1
raw_y     = hole_df[["depth_at_HW", "depth_at_fw"]].fillna(0).values.astype(np.float32)

tgt_scaler = StandardScaler()
tgt_scaler.fit(raw_y[enc1_mask])
y_tensor = torch.tensor(tgt_scaler.transform(raw_y), dtype=torch.float32)
y_orig   = raw_y.copy()

print(f"\nTarget means (orig) : HW={raw_y[enc1_mask,0].mean():.1f}m  "
      f"FW={raw_y[enc1_mask,1].mean():.1f}m")
print(f"Target stds  (orig) : HW={raw_y[enc1_mask,0].std():.1f}m  "
      f"FW={raw_y[enc1_mask,1].std():.1f}m")

# ── 5. Assemble & scale hole-level scalar features ───────────────────────────
# Group 1: X, Y, Z_g_earth already standardised by phase-1 scaler
# Group 2: hole_length, n_intervals, ss_proportion, n_ss_intervals,
#          mean_interval_len, std_interval_len  -> standardise here
SCALE_SUMMARY = ["hole_length", "n_intervals", "ss_proportion",
                 "n_ss_intervals", "mean_interval_len", "std_interval_len"]
summary_arr = hole_df[SCALE_SUMMARY].values.astype(np.float32)
summary_scaler = StandardScaler()
summary_scaled = summary_scaler.fit_transform(summary_arr)

# ss_encountered kept as-is (binary)
hole_feats = torch.tensor(
    np.hstack([
        hole_df[["X", "Y", "Z_g_earth"]].values.astype(np.float32),  # group 1 (3)
        summary_scaled,                                                # group 2 (6)
        hole_df[["ss_encountered"]].values.astype(np.float32),        # flag   (1)
    ]),
    dtype=torch.float32
)
print(f"Node feature dim : {hole_feats.shape[1]} scalar + 64 BiLSTM = "
      f"{hole_feats.shape[1] + 64} total")

# ── 6. Collar spatial coords for kNN graph construction ──────────────────────
collar_xyz = hole_df[["X_orig", "Y_orig", "Z_g_earth_orig"]].values.astype(np.float32)

# ── 7. 5-fold CV stratified by Z_g_earth quartile ─────────────────────────────
enc1_df = hole_df[enc1_mask].copy().reset_index(drop=True)
enc1_df["strat"] = pd.qcut(enc1_df["Z_g_earth_orig"], q=4, labels=False, duplicates="drop")

skf     = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
folds   = list(skf.split(enc1_df["Drillhole"], enc1_df["strat"]))
fold_of = {}
for fi, (_, vi) in enumerate(folds):
    for h in enc1_df.iloc[vi]["Drillhole"]:
        fold_of[h] = fi

hole_fold = np.array([fold_of.get(h, -1) for h in hole_df["Drillhole"]])

print(f"\n5-fold CV over {enc1_mask.sum()} ss_encountered=1 holes")
for f in range(5):
    print(f"  Fold {f}: {(hole_fold == f).sum()} holes")

# ── 8. Models ─────────────────────────────────────────────────────────────────

class IntervalEncoder(nn.Module):
    """BiLSTM over sorted interval sequence -> 64-dim hole embedding."""
    def __init__(self, in_dim: int, hidden: int = 32):
        super().__init__()
        self.lstm    = nn.LSTM(in_dim, hidden, batch_first=True, bidirectional=True)
        self.out_dim = hidden * 2

    def forward(self, x_padded, lengths):
        packed = pack_padded_sequence(x_padded, lengths.cpu(),
                                      batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return torch.cat([h[0], h[1]], dim=-1)  # [N, 64]


LSTM_DIM   = 64
HOLE_EXTRA = 10  # X, Y, Z_g_earth (3) + summary stats (6) + ss_encountered (1)
NODE_DIM   = LSTM_DIM + HOLE_EXTRA  # 74

class DrillholeGNN(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.encoder = IntervalEncoder(N_INTERVAL_FEAT)
        self.conv1   = EdgeConv(nn.Sequential(
            nn.Linear(2 * NODE_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),       nn.ReLU(),
        ), aggr="mean")
        self.norm1   = nn.LayerNorm(hidden)
        self.proj    = nn.Linear(NODE_DIM, hidden)   # residual projection
        self.conv2   = EdgeConv(nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),     nn.ReLU(),
        ), aggr="mean")
        self.norm2   = nn.LayerNorm(hidden)
        self.drop    = nn.Dropout(0.2)
        self.head    = nn.Linear(hidden, 2)

    def forward(self, seq_pad, seq_len, h_feats, edge_index):
        emb = self.encoder(seq_pad, seq_len)
        x   = torch.cat([emb, h_feats], dim=-1)          # [N, 74]
        h1  = self.drop(self.norm1(self.conv1(x, edge_index)))
        h1  = F.relu(h1 + self.proj(x))                   # residual
        h2  = self.drop(self.norm2(self.conv2(h1, edge_index)))
        h2  = F.relu(h2)
        return self.head(h2)

# ── 9. Loss ───────────────────────────────────────────────────────────────────

def huber_masked(pred, target, mask, delta=1.0):
    e = pred[mask] - target[mask]
    l = torch.where(e.abs() <= delta,
                    0.5 * e ** 2,
                    delta * (e.abs() - 0.5 * delta))
    return l.mean()

# ── 10. Graph builder ─────────────────────────────────────────────────────────

def build_graph(k: int) -> torch.Tensor:
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree",
                            n_jobs=-1).fit(collar_xyz)
    _, idxs = nbrs.kneighbors(collar_xyz)
    idxs = idxs[:, 1:]
    src  = np.repeat(np.arange(N_HOLES), k)
    dst  = idxs.ravel()
    return torch.tensor(np.stack([src, dst]), dtype=torch.long)

# ── 11. Train one fold ────────────────────────────────────────────────────────

def train_fold(edge_index, tr_mask, vl_mask):
    torch.manual_seed(SEED)
    model = DrillholeGNN()
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99), eps=1e-8)
    best_loss, best_w, wait = float("inf"), None, 0

    for ep in range(150):
        model.train()
        opt.zero_grad()
        pred = model(seq_padded, seq_lengths, hole_feats, edge_index)
        huber_masked(pred, y_tensor, tr_mask).backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            vl = huber_masked(
                model(seq_padded, seq_lengths, hole_feats, edge_index),
                y_tensor, vl_mask
            ).item()

        if vl < best_loss - 1e-6:
            best_loss = vl
            best_w    = {key: val.clone() for key, val in model.state_dict().items()}
            wait      = 0
        else:
            wait += 1
            if wait >= 15:
                break

    model.load_state_dict(best_w)
    return model, ep + 1

# ── 12. Evaluate one fold ─────────────────────────────────────────────────────

def eval_fold(model, edge_index, vl_mask):
    model.eval()
    with torch.no_grad():
        p_scaled = model(seq_padded, seq_lengths, hole_feats, edge_index).numpy()

    p_orig = tgt_scaler.inverse_transform(p_scaled)
    idx    = vl_mask.numpy()
    m      = {}
    for tgt, col in [("HW", 0), ("FW", 1)]:
        pv, tv = p_orig[idx, col], y_orig[idx, col]
        m[f"R2_{tgt}"]   = r2_score(tv, pv)
        m[f"RMSE_{tgt}"] = float(np.sqrt(mean_squared_error(tv, pv)))
        m[f"MAE_{tgt}"]  = float(mean_absolute_error(tv, pv))
    return m

# ── 13. k-sweep ───────────────────────────────────────────────────────────────
K_VALUES = [4, 6, 8, 10, 12, 16]
rows     = []

for k in K_VALUES:
    print(f"\n{'='*55}\n  k = {k}\n{'='*55}")
    edge_index = build_graph(k)
    print(f"  Graph: {N_HOLES} nodes, {edge_index.shape[1]} edges")

    fold_rows = []
    for fold in range(5):
        vl_msk = torch.tensor((hole_fold == fold)  & enc1_mask, dtype=torch.bool)
        tr_msk = torch.tensor((hole_fold != fold)  & enc1_mask, dtype=torch.bool)

        model, n_ep = train_fold(edge_index, tr_msk, vl_msk)
        m = eval_fold(model, edge_index, vl_msk)
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

# ── 14. Summary + optimal k ───────────────────────────────────────────────────
res = pd.DataFrame(rows)[
    ["k","R2_HW","RMSE_HW","MAE_HW","R2_FW","RMSE_FW","MAE_FW","epochs"]
].round(4)
res["RMSE_combined"] = res["RMSE_HW"] + res["RMSE_FW"]
opt_k = int(res.loc[res["RMSE_combined"].idxmin(), "k"])

res.to_csv(os.path.join(OUT_DIR, "k_sweep_b_summary.csv"), index=False)

print("\n" + "="*65)
print("  k-SWEEP B SUMMARY  (drillhole nodes | spatial+profile+BiLSTM, 5-fold CV mean)")
print("="*65)
print(res.drop(columns="RMSE_combined").to_string(index=False))
print(f"\n  >>> Optimal k = {opt_k}  (lowest RMSE_HW + RMSE_FW)")

# ── 15. Plot ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle(
    "k-Sweep B: Drillhole Nodes | Spatial + Profile Summary + BiLSTM  (5-Fold CV)",
    fontsize=11, fontweight="bold"
)

for ax, (col, lbl) in zip(axes, [
    ("RMSE_HW", "RMSE  HW Depth (m)"),
    ("RMSE_FW", "RMSE  FW Depth (m)"),
    ("R2_HW",   "R²  HW Depth"),
]):
    ax.plot(res["k"], res[col], "o-", color="#2563eb", lw=2, ms=7)
    ax.axvline(opt_k, color="#dc2626", ls="--", lw=1.5, label=f"Optimal k={opt_k}")
    ax.set_xlabel("k  (neighbours)"); ax.set_ylabel(lbl); ax.set_title(lbl)
    ax.set_xticks(K_VALUES); ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "k_sweep_b_results.png"), dpi=150, bbox_inches="tight")
plt.close()

print(f"\nSaved: outputs/k_sweep_b_summary.csv")
print(f"Saved: outputs/k_sweep_b_results.png")
print("\n=== PHASE 2b COMPLETE ===")
print(f"Use k = {opt_k} in Phase 3b.")
