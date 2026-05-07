"""
train_gnn.py  --  EDA/zelev  (Z-elevation targets, Z-conditioned GNN)
BiLSTM + EdgeConv GNN predicting SS layer Z-elevation contacts.

Key design: every collar node carries its own Z_HW / Z_FW as features
(scaled, same space as targets).  During training a random 40 % of the
training SS holes have their Z features masked to 0; the GNN must then
predict those values by spatially propagating context from the unmasked
neighbours.  Test holes are *always* masked so they never leak into the
loss.  During inference (predict_grid.py) all 122 known drillhole Z
values are visible — the model interpolates to undrilled query points.

SCALAR_DIM = 12
  [0:3]  X, Y, Z_g_earth (scaled)
  [3:9]  hole-level summary stats (scaled)
  [9]    ss_encountered (0/1)
  [10]   Z_HW_scaled  (0 if masked or non-SS)
  [11]   Z_FW_scaled  (0 if masked or non-SS)
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
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

SEED      = 42
K         = 10
HIDDEN    = 128
LR        = 1e-3
EPOCHS    = 250
PATIENCE  = 25
MASK_FRAC = 0.40   # fraction of train-SS holes masked per step

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}"
      + (f"  ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else ""))

ROOT    = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Load df_model ──────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(OUT_DIR, "df_model.csv"))
print(f"Loaded df_model : {df.shape}")

# ── 2. Interval feature columns ───────────────────────────────────────────────
INTERVAL_FEAT_COLS = ["MID_DEPTH", "Z_mid", "interval_length", "depth_ratio", "is_SS"]
N_INTERVAL_FEAT    = len(INTERVAL_FEAT_COLS)

# ── 3. Build per-hole records ─────────────────────────────────────────────────
hole_records, sequences = [], []

for hole, grp in df.groupby("Drillhole"):
    grp    = grp.sort_values("Depth_From")
    raw_il = (grp["Depth_To"] - grp["Depth_From"]).values.astype(np.float32)
    hole_records.append({
        "Drillhole"         : hole,
        "X_orig"            : grp["X_orig"].iloc[0],
        "Y_orig"            : grp["Y_orig"].iloc[0],
        "Z_g_earth_orig"    : grp["Z_g_earth_orig"].iloc[0],
        "ss_encountered"    : int(grp["ss_encountered"].iloc[0]),
        "Z_HW"              : grp["Z_HW"].iloc[0],
        "Z_FW"              : grp["Z_FW"].iloc[0],
        "X"                 : grp["X"].iloc[0],
        "Y"                 : grp["Y"].iloc[0],
        "Z_g_earth"         : grp["Z_g_earth"].iloc[0],
        "hole_length"       : float(grp["Depth_To"].max() - grp["Depth_From"].min()),
        "n_intervals"       : len(grp),
        "ss_proportion"     : float(grp["is_SS"].mean()),
        "n_ss_intervals"    : int(grp["is_SS"].sum()),
        "mean_interval_len" : float(raw_il.mean()),
        "std_interval_len"  : float(raw_il.std()) if len(raw_il) > 1 else 0.0,
    })
    sequences.append(grp[INTERVAL_FEAT_COLS].values.astype(np.float32))

hole_df = pd.DataFrame(hole_records)
N_HOLES = len(hole_df)

enc1_mask = hole_df["ss_encountered"].values == 1
print(f"Total drillholes   : {N_HOLES}")
print(f"  ss_encountered=1 : {enc1_mask.sum()}")
print(f"  ss_encountered=0 : {(~enc1_mask).sum()}")

# ── 4. Pad sequences ──────────────────────────────────────────────────────────
seq_lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long).to(DEVICE)
max_len     = int(seq_lengths.max().item())
seq_padded  = torch.zeros(N_HOLES, max_len, N_INTERVAL_FEAT, device=DEVICE)
for i, s in enumerate(sequences):
    seq_padded[i, :len(s)] = torch.tensor(s, device=DEVICE)

# ── 5. Standardise Z targets (fit only on SS holes) ───────────────────────────
raw_y      = hole_df[["Z_HW", "Z_FW"]].fillna(0).values.astype(np.float32)
tgt_scaler = StandardScaler()
tgt_scaler.fit(raw_y[enc1_mask])
y_tensor   = torch.tensor(tgt_scaler.transform(raw_y), dtype=torch.float32).to(DEVICE)

print(f"\nTarget stats (SS holes, metres asl):")
print(f"  Z_HW : mean={raw_y[enc1_mask,0].mean():.1f}  std={raw_y[enc1_mask,0].std():.1f}  "
      f"range=[{raw_y[enc1_mask,0].min():.1f}, {raw_y[enc1_mask,0].max():.1f}]")
print(f"  Z_FW : mean={raw_y[enc1_mask,1].mean():.1f}  std={raw_y[enc1_mask,1].std():.1f}  "
      f"range=[{raw_y[enc1_mask,1].min():.1f}, {raw_y[enc1_mask,1].max():.1f}]")

# ── 6. Hole-level scalar features (SCALAR_DIM = 12) ───────────────────────────
SCALE_SUMMARY  = ["hole_length", "n_intervals", "ss_proportion",
                  "n_ss_intervals", "mean_interval_len", "std_interval_len"]
summary_arr    = hole_df[SCALE_SUMMARY].values.astype(np.float32)
summary_scaler = StandardScaler()
summary_scaled = summary_scaler.fit_transform(summary_arr)

# Scaled Z features: 0 for non-SS, true scaled value for SS holes
y_scaled_full    = tgt_scaler.transform(raw_y)   # (N_HOLES, 2) — raw_y=0 for non-SS
z_feat_all       = y_scaled_full.astype(np.float32)
z_feat_all[~enc1_mask] = 0.0                      # non-SS holes: no Z context

SCALAR_DIM = 12
hole_feats_np = np.hstack([
    hole_df[["X", "Y", "Z_g_earth"]].values.astype(np.float32),  # 3
    summary_scaled.astype(np.float32),                             # 6
    hole_df[["ss_encountered"]].values.astype(np.float32),         # 1
    z_feat_all,                                                     # 2  <- Z_HW, Z_FW
])  # shape: (N_HOLES, 12)

hole_feats = torch.tensor(hole_feats_np, dtype=torch.float32).to(DEVICE)
print(f"\nNode feature dim : {SCALAR_DIM} scalar + 64 BiLSTM = {SCALAR_DIM + 64} total")

# ── 7. kNN graph ──────────────────────────────────────────────────────────────
collar_xyz  = hole_df[["X_orig", "Y_orig", "Z_g_earth_orig"]].values.astype(np.float32)
nbrs        = NearestNeighbors(n_neighbors=K + 1, algorithm="ball_tree", n_jobs=-1)
nbrs.fit(collar_xyz)
dists, idxs = nbrs.kneighbors(collar_xyz)

src    = np.repeat(np.arange(N_HOLES), K)
dst    = idxs[:, 1:].ravel()
d_raw  = dists[:, 1:].ravel().astype(np.float32)

edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long).to(DEVICE)
print(f"Graph : {N_HOLES} nodes, {edge_index.shape[1]} edges  (k={K})")

# ── 8. Spatial hold-out: middle X-strip (40th-60th pct) ───────────────────────
x_vals  = hole_df["X_orig"].values
x_lo    = np.quantile(x_vals, 0.40)
x_hi    = np.quantile(x_vals, 0.60)
in_strip = (x_vals >= x_lo) & (x_vals <= x_hi)

tr_bool  = (~in_strip) & enc1_mask
vl_bool  = in_strip    & enc1_mask

tr_ss_idx = np.where(tr_bool)[0]
vl_ss_idx = np.where(vl_bool)[0]
print(f"\nSpatial hold-out (middle 20% X-strip)")
print(f"  Train SS holes : {len(tr_ss_idx)}")
print(f"  Test  SS holes : {len(vl_ss_idx)}")

# ── 9. Model ──────────────────────────────────────────────────────────────────
LSTM_DIM = 64
NODE_DIM = LSTM_DIM + SCALAR_DIM   # 64 + 12 = 76


class IntervalEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, batch_first=True, bidirectional=True)

    def forward(self, x_pad, lengths):
        packed = pack_padded_sequence(x_pad, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return torch.cat([h[0], h[1]], dim=-1)   # (N, 64)


class DrillholeGNN(nn.Module):
    def __init__(self, hidden: int = HIDDEN):
        super().__init__()
        self.enc   = IntervalEncoder(N_INTERVAL_FEAT)
        self.conv1 = EdgeConv(nn.Sequential(
            nn.Linear(2 * NODE_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),       nn.ReLU(),
        ), aggr="max")
        self.norm1 = nn.LayerNorm(hidden)
        self.proj  = nn.Linear(NODE_DIM, hidden)
        self.conv2 = EdgeConv(nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),     nn.ReLU(),
        ), aggr="max")
        self.norm2 = nn.LayerNorm(hidden)
        self.drop  = nn.Dropout(0.2)
        self.head  = nn.Linear(hidden, 2)   # -> [Z_HW, Z_FW]

    def forward(self, seq_pad, seq_len, h_feats, ei):
        emb = self.enc(seq_pad, seq_len)
        x   = torch.cat([emb, h_feats], dim=-1)
        h1  = self.drop(self.norm1(self.conv1(x, ei)))
        h1  = F.relu(h1 + self.proj(x))
        h2  = self.drop(self.norm2(self.conv2(h1, ei)))
        return self.head(F.relu(h2))


model     = DrillholeGNN().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, "min", factor=0.5, patience=10, min_lr=1e-5)
mse = nn.MSELoss()

# ── 10. Training loop (random Z-masking) ──────────────────────────────────────
print(f"\nTraining  epochs={EPOCHS}  patience={PATIENCE}  mask_frac={MASK_FRAC}")
best_val   = float("inf")
best_state = None
no_improve = 0
n_mask     = max(1, round(MASK_FRAC * len(tr_ss_idx)))

for epoch in range(1, EPOCHS + 1):
    # ---- train step ----
    model.train()
    h = hole_feats.clone()
    # always mask test holes (they must not see own Z during training)
    h[vl_ss_idx, -2:] = 0.0
    # randomly mask n_mask training holes → model must predict from neighbours
    masked = np.random.choice(tr_ss_idx, n_mask, replace=False)
    h[masked, -2:] = 0.0

    optimizer.zero_grad()
    out  = model(seq_padded, seq_lengths, h, edge_index)
    loss = mse(out[masked], y_tensor[masked])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    # ---- val step: train holes unmasked → full spatial context for test holes ----
    model.eval()
    with torch.no_grad():
        h_v = hole_feats.clone()
        h_v[vl_ss_idx, -2:] = 0.0          # mask test holes only
        out_v    = model(seq_padded, seq_lengths, h_v, edge_index)
        val_loss = mse(out_v[vl_ss_idx], y_tensor[vl_ss_idx]).item()

    scheduler.step(val_loss)

    if val_loss < best_val:
        best_val   = val_loss
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"  Early stop at epoch {epoch}")
            break

    if epoch % 25 == 0 or epoch == 1:
        print(f"  Epoch {epoch:>3}  train_loss={loss.item():.4f}  "
              f"val_loss={val_loss:.4f}  best={best_val:.4f}")

# ── 11. Evaluate best model ───────────────────────────────────────────────────
model.load_state_dict(best_state)
model.eval()

with torch.no_grad():
    # Training evaluation: mask each train hole, all others unmasked
    pred_tr = np.zeros((len(tr_ss_idx), 2), dtype=np.float32)
    for li, gi in enumerate(tr_ss_idx):
        h_e = hole_feats.clone()
        h_e[gi, -2:] = 0.0
        out_e = model(seq_padded, seq_lengths, h_e, edge_index)
        pred_tr[li] = out_e[gi].cpu().numpy()

    # Test evaluation: all train holes unmasked, test holes masked
    h_v   = hole_feats.clone()
    h_v[vl_ss_idx, -2:] = 0.0
    out_v = model(seq_padded, seq_lengths, h_v, edge_index).cpu().numpy()
    pred_vl = out_v[vl_ss_idx]

true_tr = y_tensor[tr_ss_idx].cpu().numpy()
true_vl = y_tensor[vl_ss_idx].cpu().numpy()

pred_tr_m = tgt_scaler.inverse_transform(pred_tr)
pred_vl_m = tgt_scaler.inverse_transform(pred_vl)
true_tr_m = tgt_scaler.inverse_transform(true_tr)
true_vl_m = tgt_scaler.inverse_transform(true_vl)

print("\n--- Train (leave-one-out) ---")
for j, tgt in enumerate(["Z_HW", "Z_FW"]):
    r2   = r2_score(true_tr_m[:, j], pred_tr_m[:, j])
    rmse = np.sqrt(mean_squared_error(true_tr_m[:, j], pred_tr_m[:, j]))
    print(f"  {tgt}: R2={r2:.3f}  RMSE={rmse:.1f}m")

print("--- Test (spatial hold-out, train holes unmasked) ---")
for j, tgt in enumerate(["Z_HW", "Z_FW"]):
    r2   = r2_score(true_vl_m[:, j], pred_vl_m[:, j])
    rmse = np.sqrt(mean_squared_error(true_vl_m[:, j], pred_vl_m[:, j]))
    print(f"  {tgt}: R2={r2:.3f}  RMSE={rmse:.1f}m")

# Prediction range on held-out set
print(f"\nTest predictions (metres asl):")
print(f"  Z_HW: mean={pred_vl_m[:,0].mean():.1f}  std={pred_vl_m[:,0].std():.1f}  "
      f"range=[{pred_vl_m[:,0].min():.1f}, {pred_vl_m[:,0].max():.1f}]")
print(f"  Z_FW: mean={pred_vl_m[:,1].mean():.1f}  std={pred_vl_m[:,1].std():.1f}  "
      f"range=[{pred_vl_m[:,1].min():.1f}, {pred_vl_m[:,1].max():.1f}]")

# ── 12. Save ──────────────────────────────────────────────────────────────────
torch.save(best_state, os.path.join(OUT_DIR, "gnn_model.pt"))

with open(os.path.join(OUT_DIR, "gnn_scalers.pkl"), "wb") as f:
    pickle.dump({
        "summary_scaler" : summary_scaler,
        "tgt_scaler"     : tgt_scaler,
        "collar_xyz"     : collar_xyz,
        "seq_padded"     : seq_padded.cpu().numpy(),
        "seq_lengths"    : seq_lengths.cpu().numpy(),
        "hole_feats"     : hole_feats.cpu().numpy(),   # unmasked — full Z context
        "K"              : K,
        "max_len"        : max_len,
        "N_INTERVAL_FEAT": N_INTERVAL_FEAT,
        "SCALAR_DIM"     : SCALAR_DIM,
    }, f)

print(f"\nSaved: outputs/gnn_model.pt")
print(f"Saved: outputs/gnn_scalers.pkl  (SCALAR_DIM={SCALAR_DIM})")

# ── 13. Results plot ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f"GNN Z-Elevation — k={K} — Spatial Hold-Out", fontsize=12)

for j, (tgt, ax) in enumerate(zip(["Z_HW", "Z_FW"], axes)):
    vmin = min(true_vl_m[:, j].min(), pred_vl_m[:, j].min())
    vmax = max(true_vl_m[:, j].max(), pred_vl_m[:, j].max())
    ax.scatter(true_vl_m[:, j], pred_vl_m[:, j], alpha=0.7, edgecolors="k", linewidths=0.4)
    ax.plot([vmin, vmax], [vmin, vmax], "r--", lw=1)
    r2   = r2_score(true_vl_m[:, j], pred_vl_m[:, j])
    rmse = np.sqrt(mean_squared_error(true_vl_m[:, j], pred_vl_m[:, j]))
    ax.set_xlabel(f"True {tgt} (m asl)")
    ax.set_ylabel(f"Predicted {tgt} (m asl)")
    ax.set_title(f"{tgt}  R²={r2:.3f}  RMSE={rmse:.1f}m")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "gnn_results.png"), dpi=150)
print("Saved: outputs/gnn_results.png")
print("\n=== TRAINING COMPLETE ===")
