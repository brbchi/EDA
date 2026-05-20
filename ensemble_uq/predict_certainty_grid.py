"""
predict_certainty_grid.py  --  EDA/ensemble_uq

3-D certainty volume from the Deep Ensemble GNN.

Instead of a single HW/FW contact, every voxel in the 3-D grid
receives two scores derived from the M ensemble members:

  certainty_pct   0–100 %  — mean P(is_SS) across M models
                             100 = all models agree it IS SS
                               0 = all models agree it is NOT SS
                              50 = models are split / genuinely uncertain

  spread          0–0.5    — std of P(is_SS) across models
                             high spread = boundary region, uncertain location

These let you visualise not just WHERE the SS layer likely is, but
HOW CONFIDENT the model is at each location.

Output:
  outputs/certainty_grid_3d.csv
    columns: X, Y, Z, certainty_pct, spread, is_SS_pred

  outputs/certainty_grid_contacts.csv   (optional contact summary)
    columns: X, Y, Z_HW_mean, Z_HW_spread, Z_FW_mean, Z_FW_spread
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── Grid settings ─────────────────────────────────────────────────────────────
THRESHOLD  = 0.5       # certainty threshold for contact extraction
BATCH_XY   = 200       # XY points per batch
GRID_X     = 200
GRID_Y     = 160
Z_MIN, Z_MAX, Z_STEP = -760, 430, 12

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE     = os.path.dirname(os.path.abspath(__file__))
SS3D     = os.path.join(HERE, "..", "ss3d")
SS3D_OUT = os.path.join(SS3D, "outputs")
OUT_DIR  = os.path.join(HERE, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Load ensemble artefacts ────────────────────────────────────────────────
print("Loading ensemble artefacts ...")
with open(os.path.join(OUT_DIR, "gnn_ensemble_scalers.pkl"), "rb") as f:
    sc = pickle.load(f)

x_train      = torch.tensor(sc["x_train"],   dtype=torch.float32)   # (N, 6)
xyz_train    = sc["xyz_train"]                                        # (N, 3)
h1_train_ens = [torch.tensor(h, dtype=torch.float32)
                for h in sc["h1_train_ens"]]                          # list[M] (N, H)
K            = sc["K"]
NODE_DIM     = sc["NODE_DIM"]
HIDDEN       = sc["HIDDEN"]
M_ENSEMBLE   = sc["M_ENSEMBLE"]
N_TRAIN      = len(x_train)

feat_scaler = joblib.load(os.path.join(SS3D_OUT, "feature_scaler.joblib"))
print(f"Ensemble M={M_ENSEMBLE}  Training nodes={N_TRAIN:,}  K={K}")

# ── 2. Reconstruct M models ───────────────────────────────────────────────────
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

all_states = torch.load(os.path.join(OUT_DIR, "gnn_ensemble.pt"), map_location="cpu")
models = []
for state in all_states:
    m = IndicatorGNN()
    m.load_state_dict(state)
    m.eval()
    models.append(m)
print(f"{len(models)} models loaded.")

# Extract per-model MLP weights for fast vectorised inference
def extract_weights(model):
    return {
        "mlp1"  : model.conv1.nn,
        "mlp2"  : model.conv2.nn,
        "norm1" : model.norm1,
        "proj"  : model.proj,
        "norm2" : model.norm2,
        "head"  : model.head,
    }

weights = [extract_weights(m) for m in models]

# ── 3. Build kNN index on training intervals ──────────────────────────────────
print(f"Building kNN index on {N_TRAIN:,} training nodes ...")
nbrs = NearestNeighbors(n_neighbors=K, algorithm="ball_tree", n_jobs=-1)
nbrs.fit(xyz_train)

# ── 4. Z levels and XY grid ───────────────────────────────────────────────────
z_levels = np.arange(Z_MIN, Z_MAX + Z_STEP, Z_STEP, dtype=np.float32)
N_Z      = len(z_levels)
print(f"Z levels: {z_levels[0]:.0f} to {z_levels[-1]:.0f} m  step={Z_STEP}m  ({N_Z} levels)")

sim = pd.read_csv(os.path.join(HERE, "..", "Simulated_with_Predicted_XY_Enhanced.csv"))
xi = np.linspace(sim["Predicted_X"].min(), sim["Predicted_X"].max(), GRID_X, dtype=np.float32)
yi = np.linspace(sim["Predicted_Y"].min(), sim["Predicted_Y"].max(), GRID_Y, dtype=np.float32)
XX, YY   = np.meshgrid(xi, yi)
xy_grid  = np.column_stack([XX.ravel(), YY.ravel()])
N_XY     = len(xy_grid)
print(f"Grid: {GRID_X}x{GRID_Y} = {N_XY:,} XY points  x  {N_Z} Z levels = {N_XY*N_Z:,} voxels")

# ── 5. Ensemble inference per batch ──────────────────────────────────────────
def run_one_model_batch(w, h1_train_m, x_q, idx_t, nq):
    """
    Fast inference for one ensemble member using pre-computed h1_train.
    Returns probabilities (nq,).
    """
    with torch.no_grad():
        # Layer 1
        x_q_exp  = x_q.unsqueeze(1).expand(-1, K, -1)
        x_t_nbrs = x_train[idx_t]
        ef1      = torch.cat([x_q_exp, x_t_nbrs - x_q_exp], dim=-1)
        h1_agg   = w["mlp1"](ef1.view(-1, 2*NODE_DIM)).view(nq, K, HIDDEN).max(dim=1).values
        h1_q     = F.relu(w["norm1"](h1_agg) + w["proj"](x_q))

        # Layer 2 — uses this model's h1_train
        h1_q_exp  = h1_q.unsqueeze(1).expand(-1, K, -1)
        h1_t_nbrs = h1_train_m[idx_t]
        ef2       = torch.cat([h1_q_exp, h1_t_nbrs - h1_q_exp], dim=-1)
        h2_agg    = w["mlp2"](ef2.view(-1, 2*HIDDEN)).view(nq, K, HIDDEN).max(dim=1).values
        h2_q      = w["norm2"](h2_agg)

        logits = w["head"](F.relu(h2_q)).squeeze(-1)
        return torch.sigmoid(logits).numpy()


def predict_certainty_batch(xy_batch):
    """
    xy_batch : (B, 2)
    Returns  : certainty_pct (B, N_Z), spread (B, N_Z)
    """
    B  = len(xy_batch)
    nq = B * N_Z

    xy_rep = np.repeat(xy_batch, N_Z, axis=0)
    z_rep  = np.tile(z_levels, B).reshape(-1, 1)
    xyz_q  = np.hstack([xy_rep, z_rep])

    raw_q  = np.hstack([xyz_q, np.zeros((nq, 3), dtype=np.float32)])
    x_q_np = feat_scaler.transform(raw_q).astype(np.float32)
    x_q_np[:, 3:] = 0.0   # set derived features to scaled mean

    _, idx     = nbrs.kneighbors(xyz_q)
    idx_t      = torch.tensor(idx, dtype=torch.long)
    x_q        = torch.tensor(x_q_np, dtype=torch.float32)

    # Run all M models, collect (M, nq) probability array
    member_probs = np.stack([
        run_one_model_batch(weights[m], h1_train_ens[m], x_q, idx_t, nq)
        for m in range(M_ENSEMBLE)
    ])  # (M, nq)

    certainty_pct = member_probs.mean(axis=0) * 100    # (nq,)  0–100 %
    spread        = member_probs.std(axis=0)            # (nq,)

    return (certainty_pct.reshape(B, N_Z),
            spread.reshape(B, N_Z),
            member_probs.reshape(M_ENSEMBLE, B, N_Z))


# ── 6. Run grid inference — save full 3D voxel output ────────────────────────
print(f"\nRunning ensemble inference over {N_XY:,} x {N_Z} = {N_XY*N_Z:,} voxels ...")
t0 = time.time()
n_batches = (N_XY + BATCH_XY - 1) // BATCH_XY

# Pre-allocate output arrays  (N_XY, N_Z)
cert_grid   = np.zeros((N_XY, N_Z), dtype=np.float32)
spread_grid = np.zeros((N_XY, N_Z), dtype=np.float32)

# Per-model contact elevation — used to compute Z_HW_std in metres
hw_per_model = np.full((M_ENSEMBLE, N_XY), np.nan, np.float32)
fw_per_model = np.full((M_ENSEMBLE, N_XY), np.nan, np.float32)

for bi in range(n_batches):
    start = bi * BATCH_XY
    end   = min(start + BATCH_XY, N_XY)
    B = end - start
    cert_batch, spread_batch, member_batch = predict_certainty_batch(xy_grid[start:end])
    cert_grid[start:end]   = cert_batch
    spread_grid[start:end] = spread_batch

    # Per-model contact extraction (vectorised)
    above     = member_batch > THRESHOLD                        # (M, B, N_Z) bool
    any_above = above.any(axis=2)                               # (M, B)
    hw_idx    = (N_Z - 1) - above[:, :, ::-1].argmax(axis=2)  # last True in Z
    fw_idx    = above.argmax(axis=2)                            # first True in Z
    hw_per_model[:, start:end] = np.where(any_above, z_levels[hw_idx], np.nan)
    fw_per_model[:, start:end] = np.where(any_above, z_levels[fw_idx], np.nan)

    if bi % 40 == 0 or bi == n_batches - 1:
        elapsed = time.time() - t0
        eta     = elapsed / (bi + 1) * (n_batches - bi - 1)
        print(f"  [{bi+1:>4}/{n_batches}]  {end:>6,}/{N_XY:,} XY pts"
              f"  elapsed={elapsed/60:.1f}min  ETA={eta/60:.1f}min")

# ── 7. Save 3D voxel CSV ──────────────────────────────────────────────────────
print("\nBuilding 3D voxel dataframe ...")
xs = np.repeat(xy_grid[:, 0], N_Z)
ys = np.repeat(xy_grid[:, 1], N_Z)
zs = np.tile(z_levels, N_XY)

voxel_df = pd.DataFrame({
    "X"              : xs,
    "Y"              : ys,
    "Z"              : zs,
    "certainty_pct"  : cert_grid.ravel(),
    "spread"         : spread_grid.ravel(),
    "is_SS_pred"     : (cert_grid.ravel() > THRESHOLD * 100).astype(np.uint8),
})
voxel_path = os.path.join(OUT_DIR, "certainty_grid_3d.csv")
voxel_df.to_csv(voxel_path, index=False)
print(f"Saved: {voxel_path}  ({len(voxel_df):,} voxels)")

# ── 8. Contact summary: HW / FW from certainty threshold ─────────────────────
hw_mean = np.full(N_XY, np.nan, np.float32)
fw_mean = np.full(N_XY, np.nan, np.float32)
hw_spread = np.full(N_XY, np.nan, np.float32)
fw_spread = np.full(N_XY, np.nan, np.float32)

for i in range(N_XY):
    ss_idx = np.where(cert_grid[i] > THRESHOLD * 100)[0]
    if len(ss_idx) > 0:
        hw_mean[i]   = z_levels[ss_idx[-1]]         # top of SS
        fw_mean[i]   = z_levels[ss_idx[0]]          # base of SS
        hw_spread[i] = spread_grid[i, ss_idx[-1]]   # uncertainty at HW contact
        fw_spread[i] = spread_grid[i, ss_idx[0]]    # uncertainty at FW contact

# Elevation std across 100 models (metres) — used for the ±2σ visual envelope
Z_HW_std = np.nanstd(hw_per_model, axis=0)
Z_FW_std = np.nanstd(fw_per_model, axis=0)

contacts_df = pd.DataFrame({
    "X"          : xy_grid[:, 0],
    "Y"          : xy_grid[:, 1],
    "Z_HW"       : hw_mean,
    "Z_HW_spread": hw_spread,
    "Z_HW_std"   : Z_HW_std,
    "Z_FW"       : fw_mean,
    "Z_FW_spread": fw_spread,
    "Z_FW_std"   : Z_FW_std,
})
contacts_path = os.path.join(OUT_DIR, "certainty_grid_contacts.csv")
contacts_df.to_csv(contacts_path, index=False)
print(f"Saved: {contacts_path}")

valid = ~np.isnan(hw_mean)
print(f"\nGrid points with SS detected : {valid.sum():,} / {N_XY:,}")
if valid.sum() > 0:
    print(f"  Z_HW : mean={hw_mean[valid].mean():.1f}  spread={hw_spread[valid].mean():.3f}  elev_std={Z_HW_std[valid].mean():.1f}m")
    print(f"  Z_FW : mean={fw_mean[valid].mean():.1f}  spread={fw_spread[valid].mean():.3f}  elev_std={Z_FW_std[valid].mean():.1f}m")
    thick = hw_mean[valid] - fw_mean[valid]
    print(f"  Thickness : mean={thick.mean():.1f}  std={thick.std():.1f}")

# ── 9. 2D certainty slice plot (mid Z) ───────────────────────────────────────
mid_z_idx = N_Z // 2
cert_slice = cert_grid[:, mid_z_idx].reshape(GRID_Y, GRID_X)
sprd_slice = spread_grid[:, mid_z_idx].reshape(GRID_Y, GRID_X)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
im1 = axes[0].imshow(cert_slice, origin="lower", cmap="RdYlGn",
                     vmin=0, vmax=100,
                     extent=[xi[0], xi[-1], yi[0], yi[-1]], aspect="auto")
plt.colorbar(im1, ax=axes[0], label="Certainty P̄(is_SS)  [%]")
axes[0].set_title(f"Certainty at Z={z_levels[mid_z_idx]:.0f} m  [M={M_ENSEMBLE}]")
axes[0].set_xlabel("X (m)"); axes[0].set_ylabel("Y (m)")

im2 = axes[1].imshow(sprd_slice, origin="lower", cmap="plasma",
                     extent=[xi[0], xi[-1], yi[0], yi[-1]], aspect="auto")
plt.colorbar(im2, ax=axes[1], label="Spread σ")
axes[1].set_title(f"Model spread at Z={z_levels[mid_z_idx]:.0f} m")
axes[1].set_xlabel("X (m)"); axes[1].set_ylabel("Y (m)")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "certainty_slice.png"), dpi=150)
print(f"Saved: ensemble_uq/outputs/certainty_slice.png")

elapsed = time.time() - t0
print(f"\nTotal time: {elapsed/60:.1f} min")
print("=== CERTAINTY GRID COMPLETE ===")
