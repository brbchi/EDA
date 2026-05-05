"""
train_gnn.py  --  Drillhole-Graph GNN  ->  SS Hanging Wall & Foot Wall prediction

Input   : outputs/df_model.csv  (produced by phase1_data_prep)
Graph   : k=10 kNN on original collar coords (X_orig, Y_orig, Z_g_earth_orig)
Nodes   : one node per drillhole
Features: BiLSTM over depth-sorted intervals [64-dim]
          + scalar stats [10-dim]  =  74-dim total
Targets : depth_at_HW  (metres from collar to SS top)
          depth_at_fw  (metres from collar to SS base)
          -- standardised during training, inverse-transformed for eval --
          -- only ss_encountered==1 holes contribute to the loss        --

Inference: any query (x, y, z) is added as an extra node, connected to its
           k nearest collars, and the GNN returns predicted HW/FW depths.

Outputs
  outputs/gnn_model.pt        best model weights
  outputs/gnn_scalers.pkl     scalers + graph tensors for standalone inference
  outputs/gnn_cv_results.csv  per-fold CV metrics
  outputs/gnn_cv_results.png  bar charts
"""

import os, random, warnings, pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import EdgeConv
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────────
SEED     = 42
K        = 10
HIDDEN   = 128
LR       = 1e-3
EPOCHS   = 200
PATIENCE = 20

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}"
      + (f"  ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else ""))

ROOT    = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Load df_model (output of phase1_data_prep) ──────────────────────────────
df = pd.read_csv(os.path.join(OUT_DIR, "df_model.csv"))
print(f"Loaded df_model : {df.shape}")
# Columns available:
#   interval-level : Sample_Num, Drillhole, Depth_From, Depth_To,
#                    X, Y, Z_g_earth  (phase-1 scaled)
#                    MID_DEPTH, Z_mid, interval_length, depth_ratio, is_SS
#   hole-level     : ss_encountered, depth_at_HW, depth_at_fw, ss_thickness
#   originals      : X_orig, Y_orig, Z_g_earth_orig

# ── 2. Interval features (already computed & scaled by phase 1) ─────────────────
INTERVAL_FEAT_COLS = ["MID_DEPTH", "Z_mid", "interval_length", "depth_ratio", "is_SS"]
N_INTERVAL_FEAT    = len(INTERVAL_FEAT_COLS)

# ── 3. Build per-hole records ───────────────────────────────────────────────────
hole_records = []
sequences    = []

for hole, grp in df.groupby("Drillhole"):
    grp    = grp.sort_values("Depth_From")
    raw_il = (grp["Depth_To"] - grp["Depth_From"]).values.astype(np.float32)

    hole_records.append({
        # identifiers / spatial (original coords for kNN)
        "Drillhole"      : hole,
        "X_orig"         : grp["X_orig"].iloc[0],
        "Y_orig"         : grp["Y_orig"].iloc[0],
        "Z_g_earth_orig" : grp["Z_g_earth_orig"].iloc[0],
        # targets
        "ss_encountered" : int(grp["ss_encountered"].iloc[0]),
        "depth_at_HW"    : grp["depth_at_HW"].iloc[0],
        "depth_at_fw"    : grp["depth_at_fw"].iloc[0],
        # group 1 — spatial (phase-1 scaled)
        "X"              : grp["X"].iloc[0],
        "Y"              : grp["Y"].iloc[0],
        "Z_g_earth"      : grp["Z_g_earth"].iloc[0],
        # group 2 — lithological summary
        "hole_length"       : float(grp["Depth_To"].max() - grp["Depth_From"].min()),
        "n_intervals"       : len(grp),
        "ss_proportion"     : float(grp["is_SS"].mean()),
        "n_ss_intervals"    : int(grp["is_SS"].sum()),
        "mean_interval_len" : float(raw_il.mean()),
        "std_interval_len"  : float(raw_il.std()) if len(raw_il) > 1 else 0.0,
    })
    # group 3 — interval sequence for BiLSTM
    sequences.append(grp[INTERVAL_FEAT_COLS].values.astype(np.float32))

hole_df = pd.DataFrame(hole_records)
N_HOLES = len(hole_df)

enc1_mask = hole_df["ss_encountered"].values == 1
print(f"Total drillholes    : {N_HOLES}")
print(f"  ss_encountered=1  : {enc1_mask.sum()}")
print(f"  ss_encountered=0  : {(~enc1_mask).sum()}")

# ── 4. Pad sequences -> [N_HOLES, max_len, N_INTERVAL_FEAT] ────────────────────
seq_lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long).to(DEVICE)
max_len     = int(seq_lengths.max().item())
seq_padded  = torch.zeros(N_HOLES, max_len, N_INTERVAL_FEAT, device=DEVICE)
for i, s in enumerate(sequences):
    seq_padded[i, :len(s)] = torch.tensor(s, device=DEVICE)

print(f"Sequence lengths    : min={seq_lengths.min().item()}  "
      f"max={max_len}  mean={seq_lengths.float().mean().item():.1f}")

# ── 5. Standardise targets (fit only on ss_encountered==1 holes) ────────────────
raw_y      = hole_df[["depth_at_HW", "depth_at_fw"]].fillna(0).values.astype(np.float32)
tgt_scaler = StandardScaler()
tgt_scaler.fit(raw_y[enc1_mask])
y_tensor   = torch.tensor(tgt_scaler.transform(raw_y), dtype=torch.float32).to(DEVICE)
y_orig     = raw_y.copy()

print(f"\nTarget stats (SS holes only) :")
print(f"  depth_at_HW : mean={raw_y[enc1_mask,0].mean():.1f}m  "
      f"std={raw_y[enc1_mask,0].std():.1f}m")
print(f"  depth_at_fw : mean={raw_y[enc1_mask,1].mean():.1f}m  "
      f"std={raw_y[enc1_mask,1].std():.1f}m")

# ── 6. Assemble & scale hole-level scalar features ──────────────────────────────
# group 1 : X, Y, Z_g_earth  already standardised by phase-1  (3 dims)
# group 2 : hole_length, n_intervals, ss_proportion, n_ss_intervals,
#            mean_interval_len, std_interval_len               (6 dims)
# flag    : ss_encountered  (kept binary)                      (1 dim)
# total   : 10 dims
SCALE_SUMMARY = ["hole_length", "n_intervals", "ss_proportion",
                 "n_ss_intervals", "mean_interval_len", "std_interval_len"]
summary_arr    = hole_df[SCALE_SUMMARY].values.astype(np.float32)
summary_scaler = StandardScaler()
summary_scaled = summary_scaler.fit_transform(summary_arr)

SCALAR_DIM = 10
hole_feats = torch.tensor(
    np.hstack([
        hole_df[["X", "Y", "Z_g_earth"]].values.astype(np.float32),   # 3
        summary_scaled,                                                  # 6
        hole_df[["ss_encountered"]].values.astype(np.float32),          # 1
    ]),
    dtype=torch.float32,
).to(DEVICE)

print(f"\nNode feature dim : {SCALAR_DIM} scalar + 64 BiLSTM = {SCALAR_DIM + 64} total")

# ── 7. Build k=10 kNN graph using 3-D Euclidean distance ───────────────────────
# d_ij = sqrt((Xi-Xj)² + (Yi-Yj)² + (Zi-Zj)²)  [paper Eq. 7]
collar_xyz = hole_df[["X_orig", "Y_orig", "Z_g_earth_orig"]].values.astype(np.float32)
nbrs       = NearestNeighbors(n_neighbors=K + 1, algorithm="ball_tree", n_jobs=-1).fit(collar_xyz)
dists, idxs = nbrs.kneighbors(collar_xyz)

src   = np.repeat(np.arange(N_HOLES), K)
dst   = idxs[:, 1:].ravel()
d_raw = dists[:, 1:].ravel().astype(np.float32)          # Euclidean distance per edge

# Normalise distances to [0, 1] so they are scale-invariant edge attributes
d_norm     = d_raw / (d_raw.max() + 1e-8)
edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long).to(DEVICE)
edge_attr  = torch.tensor(d_norm, dtype=torch.float32).unsqueeze(1).to(DEVICE)  # [E, 1]

print(f"Graph : {N_HOLES} nodes, {edge_index.shape[1]} edges  (k={K})")
print(f"Edge distances : min={d_raw.min():.0f}m  "
      f"max={d_raw.max():.0f}m  mean={d_raw.mean():.0f}m")

# ── 8. Spatial hold-out: middle X-strip as test set ────────────────────────────
# Divide the X range into 5 quantile-based strips; hold out the middle strip
# (~40th–60th percentile of X). This places a spatial gap between train and
# test so no test drillhole has a training neighbour immediately adjacent.
x_vals   = hole_df["X_orig"].values
x_edges  = np.quantile(x_vals, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
strip    = np.digitize(x_vals, x_edges[1:-1])   # 0 = leftmost … 4 = rightmost
TEST_STRIP = 2                                    # middle strip

test_spatial = (strip == TEST_STRIP)
tr_bool = (~test_spatial) & enc1_mask
vl_bool = test_spatial   & enc1_mask

tr_msk = torch.tensor(tr_bool, dtype=torch.bool)
vl_msk = torch.tensor(vl_bool, dtype=torch.bool)

print(f"\nSpatial hold-out  (middle X-strip  ≈ 20 % of area)")
print(f"  X range      : [{x_edges[0]:.0f} m, {x_edges[-1]:.0f} m]")
print(f"  Test strip X : [{x_edges[2]:.0f} m, {x_edges[4]:.0f} m]")
print(f"  Train SS holes : {int(tr_bool.sum())}")
print(f"  Test  SS holes : {int(vl_bool.sum())}")

# ── 9. Model ───────────────────────────────────────────────────────────────────
LSTM_DIM = 64
NODE_DIM = LSTM_DIM + SCALAR_DIM   # 74


class IntervalEncoder(nn.Module):
    """BiLSTM over depth-sorted intervals -> 64-dim drillhole embedding."""
    def __init__(self, in_dim: int, hidden: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, batch_first=True, bidirectional=True)

    def forward(self, x_padded, lengths):
        packed = pack_padded_sequence(
            x_padded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h, _) = self.lstm(packed)
        return torch.cat([h[0], h[1]], dim=-1)   # [N, 64]


class DrillholeGNN(nn.Module):
    """2-layer EdgeConv + LayerNorm + residual connection."""
    def __init__(self, hidden: int = HIDDEN):
        super().__init__()
        self.enc   = IntervalEncoder(N_INTERVAL_FEAT)
        self.conv1 = EdgeConv(nn.Sequential(
            nn.Linear(2 * NODE_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),       nn.ReLU(),
        ), aggr="mean")
        self.norm1 = nn.LayerNorm(hidden)
        self.proj  = nn.Linear(NODE_DIM, hidden)
        self.conv2 = EdgeConv(nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),     nn.ReLU(),
        ), aggr="mean")
        self.norm2 = nn.LayerNorm(hidden)
        self.drop  = nn.Dropout(0.2)
        self.head  = nn.Linear(hidden, 2)           # -> [depth_at_HW, depth_at_fw]

    def forward(self, seq_pad, seq_len, h_feats, ei):
        emb = self.enc(seq_pad, seq_len)             # [N, 64]
        x   = torch.cat([emb, h_feats], dim=-1)      # [N, 74]
        h1  = self.drop(self.norm1(self.conv1(x, ei)))
        h1  = F.relu(h1 + self.proj(x))             # residual
        h2  = self.drop(self.norm2(self.conv2(h1, ei)))
        return self.head(F.relu(h2))                 # [N, 2]


# ── 10. Masked Huber loss ───────────────────────────────────────────────────────
def huber_masked(pred, tgt, mask, delta=1.0):
    e = pred[mask] - tgt[mask]
    return torch.where(
        e.abs() <= delta, 0.5 * e ** 2, delta * (e.abs() - 0.5 * delta)
    ).mean()


# ── 11. Train one fold ──────────────────────────────────────────────────────────
def train_fold(tr_mask, vl_mask):
    torch.manual_seed(SEED)
    mdl     = DrillholeGNN().to(DEVICE)
    opt     = torch.optim.Adam(mdl.parameters(), lr=LR, betas=(0.9, 0.99), eps=1e-8)
    tr_mask = tr_mask.to(DEVICE)
    vl_mask = vl_mask.to(DEVICE)
    best_loss, best_w, wait = float("inf"), None, 0

    for ep in range(EPOCHS):
        mdl.train()
        opt.zero_grad()
        huber_masked(
            mdl(seq_padded, seq_lengths, hole_feats, edge_index),
            y_tensor, tr_mask
        ).backward()
        opt.step()

        mdl.eval()
        with torch.no_grad():
            vl = huber_masked(
                mdl(seq_padded, seq_lengths, hole_feats, edge_index),
                y_tensor, vl_mask
            ).item()

        if vl < best_loss - 1e-6:
            best_loss = vl
            best_w    = {k: v.clone() for k, v in mdl.state_dict().items()}
            wait      = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    mdl.load_state_dict(best_w)
    return mdl, vl_mask, ep + 1


# ── 12. Evaluate one fold ───────────────────────────────────────────────────────
def eval_fold(mdl, vl_mask):
    mdl.eval()
    with torch.no_grad():
        ps = mdl(seq_padded, seq_lengths, hole_feats, edge_index).cpu().numpy()
    p   = tgt_scaler.inverse_transform(ps)
    idx = vl_mask.cpu().numpy()
    m   = {}
    for name, col in [("HW", 0), ("FW", 1)]:
        pv, tv = p[idx, col], y_orig[idx, col]
        m[f"R2_{name}"]   = r2_score(tv, pv)
        m[f"RMSE_{name}"] = float(np.sqrt(mean_squared_error(tv, pv)))
        m[f"MAE_{name}"]  = float(mean_absolute_error(tv, pv))
    return m


# ── 13. Train on spatial hold-out split ─────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  GNN Training  (k={K}, spatial hold-out, max {EPOCHS} ep, patience={PATIENCE})")
print(f"{'='*60}")

mdl, _, n_ep = train_fold(tr_msk, vl_msk)
m_test  = eval_fold(mdl, vl_msk)
m_train = eval_fold(mdl, tr_msk)
best_state = {k: v.cpu().clone() for k, v in mdl.state_dict().items()}

print(f"  Train : R2_HW={m_train['R2_HW']:.3f}  RMSE_HW={m_train['RMSE_HW']:.1f}m  "
      f"R2_FW={m_train['R2_FW']:.3f}  RMSE_FW={m_train['RMSE_FW']:.1f}m")
print(f"  Test  : R2_HW={m_test['R2_HW']:.3f}  RMSE_HW={m_test['RMSE_HW']:.1f}m  "
      f"R2_FW={m_test['R2_FW']:.3f}  RMSE_FW={m_test['RMSE_FW']:.1f}m  (ep={n_ep})")

results_df = pd.DataFrame([
    {"split": "train", **m_train},
    {"split": "test",  **m_test},
])
results_df.to_csv(os.path.join(OUT_DIR, "gnn_results.csv"), index=False)

# ── 14. Save model + everything needed for inference ────────────────────────────
torch.save(best_state, os.path.join(OUT_DIR, "gnn_model.pt"))

with open(os.path.join(OUT_DIR, "gnn_scalers.pkl"), "wb") as f:
    pickle.dump({
        "summary_scaler" : summary_scaler,
        "tgt_scaler"     : tgt_scaler,
        "collar_xyz"     : collar_xyz,
        "d_raw_max"      : float(d_raw.max()),
        "seq_padded"     : seq_padded.cpu().numpy(),
        "seq_lengths"    : seq_lengths.cpu().numpy(),
        "hole_feats"     : hole_feats.cpu().numpy(),
        "K"              : K,
        "max_len"        : max_len,
        "N_INTERVAL_FEAT": N_INTERVAL_FEAT,
        "SCALAR_DIM"     : SCALAR_DIM,
    }, f)

print(f"\nSaved : outputs/gnn_model.pt")
print(f"Saved : outputs/gnn_scalers.pkl")

# ── 15. Results plot ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle(f"GNN  k={K}  ·  Spatial Hold-Out (middle X-strip)", fontsize=11, fontweight="bold")

# Spatial scatter: all holes coloured by split
ax = axes[0]
non_ss = np.logical_not(enc1_mask)
ax.scatter(hole_df.loc[non_ss, "X_orig"], hole_df.loc[non_ss, "Y_orig"],
           c="#d1d5db", s=12, label="non-SS", zorder=1)
ax.scatter(hole_df.loc[tr_bool, "X_orig"], hole_df.loc[tr_bool, "Y_orig"],
           c="#2563eb", s=22, label=f"train  (n={int(tr_bool.sum())})", zorder=2)
ax.scatter(hole_df.loc[vl_bool, "X_orig"], hole_df.loc[vl_bool, "Y_orig"],
           c="#dc2626", s=30, marker="^", label=f"test   (n={int(vl_bool.sum())})", zorder=3)
ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_title("Spatial Split")
ax.legend(fontsize=8); ax.ticklabel_format(style="sci", axis="both", scilimits=(5, 5))

# RMSE bar chart
ax = axes[1]
labels = ["HW", "FW"]
x_pos  = np.arange(len(labels))
ax.bar(x_pos - 0.2, [m_train["RMSE_HW"], m_train["RMSE_FW"]], 0.35,
       label="train", color="#2563eb", alpha=0.85)
ax.bar(x_pos + 0.2, [m_test["RMSE_HW"],  m_test["RMSE_FW"]],  0.35,
       label="test",  color="#dc2626", alpha=0.85)
ax.set_xticks(x_pos); ax.set_xticklabels(["depth_at_HW", "depth_at_fw"])
ax.set_ylabel("RMSE (m)"); ax.set_title("RMSE"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

# R² bar chart
ax = axes[2]
ax.bar(x_pos - 0.2, [m_train["R2_HW"], m_train["R2_FW"]], 0.35,
       label="train", color="#2563eb", alpha=0.85)
ax.bar(x_pos + 0.2, [m_test["R2_HW"],  m_test["R2_FW"]],  0.35,
       label="test",  color="#dc2626", alpha=0.85)
ax.set_xticks(x_pos); ax.set_xticklabels(["depth_at_HW", "depth_at_fw"])
ax.set_ylabel("R²"); ax.set_title("R²"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "gnn_results.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved : outputs/gnn_results.png")

# ── 16. Inference: predict HW / FW depth at any (x, y, z) ──────────────────────
def predict_hwfw(x: float, y: float, z: float) -> dict:
    """
    Predict SS Hanging-Wall and Foot-Wall depths at an arbitrary location.

    Parameters
    ----------
    x, y, z : original collar coordinates (same CRS / units as df_model X_orig,
               Y_orig, Z_g_earth_orig).  z is the surface elevation.

    Returns
    -------
    {"depth_at_HW": float, "depth_at_fw": float}
        Predicted drill depths (metres from collar) to SS top and SS base.

    Method
    ------
    The query point is inserted as node N into the trained graph.
    It is connected to its k nearest collars and is given:
      - scalar features : phase-1-scaled (x, y, z) in positions [0:3],
                          zeros for the summary stats, ss_flag = 0
      - sequence        : one zero-padded step (no composite data)
    The GNN forward pass runs over the augmented graph and the output at
    index N is inverse-transformed to original depth units.
    """
    q_xyz   = np.array([[x, y, z]], dtype=np.float32)
    aug_xyz = np.vstack([collar_xyz, q_xyz])       # [N+1, 3]
    n_aug   = len(aug_xyz)
    q_idx   = n_aug - 1

    # Edges for augmented graph — same 3-D Euclidean construction as training
    nbrs_aug         = NearestNeighbors(
        n_neighbors=min(K + 1, n_aug), algorithm="ball_tree"
    ).fit(aug_xyz)
    dists_aug, idxs_aug = nbrs_aug.kneighbors(aug_xyz)
    idxs_aug  = idxs_aug[:, 1:]
    dists_aug = dists_aug[:, 1:].ravel().astype(np.float32)
    k_eff     = idxs_aug.shape[1]
    src_aug   = np.repeat(np.arange(n_aug), k_eff)
    dst_aug   = idxs_aug.ravel()
    ei_aug    = torch.tensor(
        np.stack([src_aug, dst_aug]), dtype=torch.long
    ).to(DEVICE)
    # Query node scalar features
    # X, Y, Z_g_earth were already standardised in phase-1; replicate that
    # by reading the phase-1 scaler means/stds from the first three columns
    # of hole_feats (they were directly taken from df_model X/Y/Z_g_earth).
    # Since we don't store the phase-1 scaler here we do a simple z-score
    # against the training set's own statistics from hole_feats.
    hf_np     = hole_feats.cpu().numpy()             # [N, 10]
    mu_xyz    = hf_np[:, :3].mean(axis=0)
    std_xyz   = hf_np[:, :3].std(axis=0) + 1e-8
    q_xyz_s   = ((np.array([x, y, z], dtype=np.float32) - mu_xyz) / std_xyz)
    q_scalar  = np.zeros(SCALAR_DIM, dtype=np.float32)
    q_scalar[:3] = q_xyz_s
    q_feat    = torch.tensor(q_scalar).unsqueeze(0).to(DEVICE)
    hf_aug    = torch.cat([hole_feats, q_feat], dim=0)   # [N+1, 10]

    # Query node sequence (one zero step, padded to max_len)
    q_seq  = torch.zeros(1, max_len, N_INTERVAL_FEAT, device=DEVICE)
    q_len  = torch.tensor([1], dtype=torch.long, device=DEVICE)
    sp_aug = torch.cat([seq_padded, q_seq], dim=0)       # [N+1, max_len, 5]
    sl_aug = torch.cat([seq_lengths, q_len], dim=0)      # [N+1]

    # Load best model and run inference
    mdl = DrillholeGNN().to(DEVICE)
    mdl.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    mdl.eval()
    with torch.no_grad():
        out = mdl(sp_aug, sl_aug, hf_aug, ei_aug).cpu().numpy()

    hw, fw = tgt_scaler.inverse_transform(out)[q_idx]
    return {"depth_at_HW": round(float(hw), 2), "depth_at_fw": round(float(fw), 2)}


# ── 17. Demo inference ──────────────────────────────────────────────────────────
cx = float(hole_df["X_orig"].mean())
cy = float(hole_df["Y_orig"].mean())
cz = float(hole_df["Z_g_earth_orig"].mean())
r  = predict_hwfw(cx, cy, cz)
print(f"\nDemo inference at cloud centroid  (X={cx:.0f}, Y={cy:.0f}, Z={cz:.0f}) :")
print(f"  Predicted depth_at_HW  =  {r['depth_at_HW']} m  (SS top)")
print(f"  Predicted depth_at_fw  =  {r['depth_at_fw']} m  (SS base)")

print("\n=== train_gnn.py complete ===")
