"""
predict_grid.py  --  EDA/ss3d  (3-D indicator GNN inference)

For every (X, Y) point in a dense 2-D grid, the GNN predicts P(is_SS)
at a stack of Z levels.  Z_HW and Z_FW are then derived as:

  Z_HW = highest Z where P(is_SS) > THRESHOLD   (top of SS layer)
  Z_FW = lowest  Z where P(is_SS) > THRESHOLD   (base of SS layer)

Inference is fast because h1_train was pre-computed during training and
stored in gnn_scalers.pkl.  For each query batch we only run the MLP
weights — no full graph rebuild required.

Output: outputs/predicted_grid.csv
  Columns: Predicted_X, Predicted_Y, Z_HW_elev, Z_FW_elev
"""

import os, time, pickle, warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import EdgeConv
import numpy as np
import pandas as pd
import joblib
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

THRESHOLD  = 0.5      # P(is_SS) cut-off for contact extraction
BATCH_XY   = 300      # XY points per inference batch
GRID_X     = 200      # XY grid resolution
GRID_Y     = 160
Z_MIN, Z_MAX, Z_STEP = -760, 430, 12   # elevation levels to sample (m asl)

ROOT = os.path.dirname(os.path.abspath(__file__))
EDA  = os.path.join(ROOT, "..")
OUT  = os.path.join(ROOT, "outputs")

# ── 1. Load artefacts ─────────────────────────────────────────────────────────
with open(os.path.join(OUT, "gnn_scalers.pkl"), "rb") as f:
    sc = pickle.load(f)

x_train   = torch.tensor(sc["x_train"],   dtype=torch.float32)   # (N_TRAIN, 6)
xyz_train = sc["xyz_train"]                                        # (N_TRAIN, 3)
h1_train  = torch.tensor(sc["h1_train"],  dtype=torch.float32)   # (N_TRAIN, HIDDEN)
K         = sc["K"]
NODE_DIM  = sc["NODE_DIM"]
HIDDEN    = sc["HIDDEN"]

feat_scaler = joblib.load(os.path.join(OUT, "feature_scaler.joblib"))
N_TRAIN = len(x_train)

print(f"Training intervals : {N_TRAIN:,}  |  K={K}  |  NODE_DIM={NODE_DIM}  |  HIDDEN={HIDDEN}")

# ── 2. Reconstruct model and extract MLP weights ───────────────────────────────
class IndicatorGNN(nn.Module):
    def __init__(self, in_dim=NODE_DIM, hidden=HIDDEN):
        super().__init__()
        self.conv1 = EdgeConv(nn.Sequential(
            nn.Linear(2 * in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),     nn.ReLU(),
        ), aggr="max")
        self.norm1 = nn.LayerNorm(hidden)
        self.proj  = nn.Linear(in_dim, hidden)
        self.conv2 = EdgeConv(nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),     nn.ReLU(),
        ), aggr="max")
        self.norm2 = nn.LayerNorm(hidden)
        self.drop  = nn.Dropout(0.2)
        self.head  = nn.Linear(hidden, 1)

    def forward(self, x, ei):
        h1 = self.drop(self.norm1(self.conv1(x, ei)))
        h1 = F.relu(h1 + self.proj(x))
        h2 = self.drop(self.norm2(self.conv2(h1, ei)))
        return self.head(F.relu(h2)).squeeze(-1)

model = IndicatorGNN()
model.load_state_dict(torch.load(os.path.join(OUT, "gnn_model.pt"), map_location="cpu"))
model.eval()

mlp1 = model.conv1.nn   # MLP inside EdgeConv layer 1: 12-dim → HIDDEN
mlp2 = model.conv2.nn   # MLP inside EdgeConv layer 2: 2*HIDDEN-dim → HIDDEN
norm1, proj = model.norm1, model.proj
norm2       = model.norm2
head        = model.head
print("Model loaded.")

# ── 3. Build kNN index on training intervals (3-D) ────────────────────────────
print(f"Building 3-D kNN index on {N_TRAIN:,} training nodes ...")
nbrs = NearestNeighbors(n_neighbors=K, algorithm="ball_tree", n_jobs=-1)
nbrs.fit(xyz_train)
print("kNN index ready.")

# ── 4. Z levels and XY query grid ─────────────────────────────────────────────
z_levels = np.arange(Z_MIN, Z_MAX + Z_STEP, Z_STEP, dtype=np.float32)
N_Z      = len(z_levels)
print(f"Z levels   : {z_levels[0]:.0f} to {z_levels[-1]:.0f} m  step={Z_STEP}m  ({N_Z} levels)")

# XY extent from Simulated file
sim = pd.read_csv(os.path.join(EDA, "Simulated_with_Predicted_XY_Enhanced.csv"))
x_min, x_max = sim["Predicted_X"].min(), sim["Predicted_X"].max()
y_min, y_max = sim["Predicted_Y"].min(), sim["Predicted_Y"].max()

xi = np.linspace(x_min, x_max, GRID_X, dtype=np.float32)
yi = np.linspace(y_min, y_max, GRID_Y, dtype=np.float32)
XX, YY = np.meshgrid(xi, yi)
xy_grid = np.column_stack([XX.ravel(), YY.ravel()])   # (GRID_X*GRID_Y, 2)
N_XY    = len(xy_grid)
print(f"XY grid    : {GRID_X}x{GRID_Y} = {N_XY:,} points")
print(f"Total query nodes : {N_XY:,} x {N_Z} = {N_XY*N_Z:,}")

hw_out = np.full(N_XY, np.nan, dtype=np.float32)
fw_out = np.full(N_XY, np.nan, dtype=np.float32)

# ── 5. Vectorised inference ───────────────────────────────────────────────────
# For a batch of (X,Y) points expanded over all Z levels:
#   Layer 1: h1_q = relu( norm1( max_j MLP1([x_q||x_train[j]-x_q]) ) + proj(x_q) )
#   Layer 2: h2_q = norm2( max_j MLP2([h1_q||h1_train[j]-h1_q]) )
#   out_q   = head( relu(h2_q) )
#
# h1_train is pre-computed — no full graph rebuild needed.

def predict_batch(xy_batch):
    """
    xy_batch : (B, 2) XY positions
    Returns  : probs (B, N_Z) — P(is_SS) at each Z level
    """
    B  = len(xy_batch)
    nq = B * N_Z

    # Expand XY over Z levels
    xy_rep = np.repeat(xy_batch, N_Z, axis=0)           # (nq, 2)
    z_rep  = np.tile(z_levels, B).reshape(-1, 1)        # (nq, 1)
    xyz_q  = np.hstack([xy_rep, z_rep])                  # (nq, 3) in original units

    # Scale X, Y, Z_mid via the fitted scaler; set derived features
    # (MID_DEPTH, interval_length, depth_ratio) to 0 IN SCALED SPACE = training mean.
    # Setting raw=0 would give (0-mean)/std which is far out of distribution.
    raw_q  = np.hstack([xyz_q, np.zeros((nq, 3), dtype=np.float32)])
    x_q_np = feat_scaler.transform(raw_q).astype(np.float32)
    x_q_np[:, 3:] = 0.0   # force unknown features to scaled mean

    # kNN: find K nearest training intervals in 3-D for each query
    _, idx = nbrs.kneighbors(xyz_q)    # (nq, K)
    idx_t  = torch.tensor(idx, dtype=torch.long)    # (nq, K)

    x_q    = torch.tensor(x_q_np, dtype=torch.float32)  # (nq, 6)

    with torch.no_grad():
        # ── Layer 1 ──────────────────────────────────────────────────────────
        x_q_exp  = x_q.unsqueeze(1).expand(-1, K, -1)       # (nq, K, 6)
        x_t_nbrs = x_train[idx_t]                            # (nq, K, 6)
        ef1      = torch.cat([x_q_exp, x_t_nbrs - x_q_exp], dim=-1)   # (nq, K, 12)
        h1_edge  = mlp1(ef1.view(-1, 2 * NODE_DIM)).view(nq, K, HIDDEN)
        h1_agg   = h1_edge.max(dim=1).values                 # (nq, HIDDEN)
        h1_q     = F.relu(norm1(h1_agg) + proj(x_q))        # (nq, HIDDEN)

        # ── Layer 2 ──────────────────────────────────────────────────────────
        h1_q_exp  = h1_q.unsqueeze(1).expand(-1, K, -1)      # (nq, K, HIDDEN)
        h1_t_nbrs = h1_train[idx_t]                           # (nq, K, HIDDEN)
        ef2       = torch.cat([h1_q_exp, h1_t_nbrs - h1_q_exp], dim=-1)  # (nq, K, 2H)
        h2_edge   = mlp2(ef2.view(-1, 2 * HIDDEN)).view(nq, K, HIDDEN)
        h2_agg    = h2_edge.max(dim=1).values                 # (nq, HIDDEN)
        h2_q      = norm2(h2_agg)                             # (nq, HIDDEN)

        # ── Head ──────────────────────────────────────────────────────────────
        logits = head(F.relu(h2_q)).squeeze(-1)               # (nq,)
        probs  = torch.sigmoid(logits).numpy()                 # (nq,)

    return probs.reshape(B, N_Z)   # (B, N_Z)


t0       = time.time()
n_batches = (N_XY + BATCH_XY - 1) // BATCH_XY

for bi in range(n_batches):
    start = bi * BATCH_XY
    end   = min(start + BATCH_XY, N_XY)
    probs = predict_batch(xy_grid[start:end])   # (batch, N_Z)

    for j, prob_profile in enumerate(probs):
        ss_idx = np.where(prob_profile > THRESHOLD)[0]
        if len(ss_idx) > 0:
            hw_out[start + j] = z_levels[ss_idx[-1]]   # highest Z = HW (top)
            fw_out[start + j] = z_levels[ss_idx[0]]    # lowest  Z = FW (base)

    if bi % 50 == 0 or bi == n_batches - 1:
        elapsed = time.time() - t0
        eta     = elapsed / (bi + 1) * (n_batches - bi - 1)
        valid   = (~np.isnan(hw_out[:end])).sum()
        print(f"  [{bi+1:>4}/{n_batches}]  {end:>6,}/{N_XY:,} pts  "
              f"SS found={valid:,}  elapsed={elapsed/60:.1f}min  ETA={eta/60:.1f}min")

# ── 6. Save ───────────────────────────────────────────────────────────────────
out_df = pd.DataFrame({
    "Predicted_X" : xy_grid[:, 0],
    "Predicted_Y" : xy_grid[:, 1],
    "Z_HW_elev"   : hw_out,
    "Z_FW_elev"   : fw_out,
})
out_path = os.path.join(OUT, "predicted_grid.csv")
out_df.to_csv(out_path, index=False)

valid_hw = hw_out[~np.isnan(hw_out)]
valid_fw = fw_out[~np.isnan(fw_out)]
elapsed  = time.time() - t0
print(f"\nDone in {elapsed/60:.1f} min  ->  {out_path}")
print(f"Grid points with SS detected : {len(valid_hw):,} / {N_XY:,}")
if len(valid_hw) > 0:
    print(f"  Z_HW_elev : mean={valid_hw.mean():.1f}  std={valid_hw.std():.1f}  "
          f"range=[{valid_hw.min():.1f}, {valid_hw.max():.1f}]")
    print(f"  Z_FW_elev : mean={valid_fw.mean():.1f}  std={valid_fw.std():.1f}  "
          f"range=[{valid_fw.min():.1f}, {valid_fw.max():.1f}]")
    print(f"  Thickness : mean={(valid_hw-valid_fw).mean():.1f}  "
          f"std={(valid_hw-valid_fw).std():.1f}")
else:
    print("  WARNING: no SS detected — check Z range or threshold")
