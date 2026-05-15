"""
train_ensemble.py  --  EDA/ensemble_uq

Deep Ensemble Uncertainty Quantification for the SS-layer GNN.

Theory:
  Train M independent GNN models with different random seeds.
  Each model starts from a different random initialisation and
  therefore converges to a different loss basin — this is the
  source of *epistemic* (model) uncertainty.

  By the Law of Large Numbers:
    certainty(voxel) = (1/M) Σ p_m(voxel)  →  E[P(SS | voxel)]  as M → ∞

  The running-convergence plot shows exactly when M is "enough"
  (the estimate stops changing — typically M = 5–7 suffices).

Per-voxel outputs written to gnn_ensemble_scalers.pkl
  and used by predict_certainty_grid.py:
    p_mean_pct  certainty score 0–100 %  (fraction of models voting SS)
    p_std       spread / disagreement    (high = boundary is uncertain)
"""

import os, sys, random, warnings, pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import EdgeConv
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE    = os.path.dirname(os.path.abspath(__file__))
SS3D    = os.path.join(HERE, "..", "ss3d")          # existing ss3d outputs
SS3D_OUT = os.path.join(SS3D, "outputs")
OUT_DIR  = os.path.join(HERE, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Hyper-parameters ──────────────────────────────────────────────────────────
M_ENSEMBLE = 7
SEEDS      = [42, 123, 456, 789, 1234, 2024, 31415][:M_ENSEMBLE]
K          = 15
HIDDEN     = 64
LR         = 1e-3
EPOCHS     = 200
PATIENCE   = 20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}  |  M={M_ENSEMBLE} ensemble members")

# ── 1. Load intervals ─────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(SS3D_OUT, "df_model.csv"))
print(f"Intervals loaded : {df.shape[0]:,}")

NODE_FEAT_COLS = ["X", "Y", "Z_mid", "MID_DEPTH", "interval_length", "depth_ratio"]
NODE_DIM = len(NODE_FEAT_COLS)

x_np   = df[NODE_FEAT_COLS].values.astype(np.float32)
xyz_np = df[["X_orig", "Y_orig", "Z_orig"]].values.astype(np.float32)
y_np   = df["is_SS"].values.astype(np.float32)
N      = len(df)
print(f"SS fraction : {y_np.mean():.3f}  ({int(y_np.sum()):,} SS / {N:,} total)")

# ── 2. Spatial holdout ────────────────────────────────────────────────────────
hole_x     = df.groupby("Drillhole")["X_orig"].first()
x_lo       = np.quantile(hole_x.values, 0.40)
x_hi       = np.quantile(hole_x.values, 0.60)
test_holes = hole_x[(hole_x >= x_lo) & (hole_x <= x_hi)].index.tolist()
test_mask  = df["Drillhole"].isin(test_holes).values
tr_idx     = np.where(~test_mask)[0]
vl_idx     = np.where(test_mask)[0]
print(f"Spatial hold-out  train={len(tr_idx):,}  test={len(vl_idx):,}")

# ── 3. Build 3-D kNN graph — done ONCE, seed-independent ─────────────────────
print(f"\nBuilding k={K} graph ...")
nbrs_fit = NearestNeighbors(n_neighbors=K + 1, algorithm="ball_tree", n_jobs=-1)
nbrs_fit.fit(xyz_np)
_, idxs = nbrs_fit.kneighbors(xyz_np)
idxs = idxs[:, 1:]
src = np.repeat(np.arange(N), K)
dst = idxs.ravel()
edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long).to(DEVICE)
print(f"Graph: {N:,} nodes  {edge_index.shape[1]:,} edges")

x_t = torch.tensor(x_np, dtype=torch.float32).to(DEVICE)
y_t = torch.tensor(y_np, dtype=torch.float32).to(DEVICE)

# ── 4. Class weight ───────────────────────────────────────────────────────────
pos_weight = torch.tensor([(N - y_np.sum()) / y_np.sum()],
                          dtype=torch.float32).to(DEVICE)
print(f"pos_weight = {pos_weight.item():.2f}")

# ── 5. Model definition ───────────────────────────────────────────────────────
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

    def embeddings(self, x, ei):
        """Dropout-free forward for fast grid inference."""
        with torch.no_grad():
            h1 = self.norm1(self.conv1(x, ei))
            h1 = F.relu(h1 + self.proj(x))
            h2 = self.norm2(self.conv2(h1, ei))
        return h1, h2


bce  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
tr_t = torch.tensor(tr_idx, dtype=torch.long)
vl_t = torch.tensor(vl_idx, dtype=torch.long)

# ── 6. Deep Ensemble loop ─────────────────────────────────────────────────────
# all_probs[m, n] = P(is_SS | node n) from model m
all_probs    = np.zeros((M_ENSEMBLE, N), dtype=np.float32)
all_states   = []
h1_train_ens = []
h2_train_ens = []

print(f"\n{'='*60}")
print(f"  DEEP ENSEMBLE  M={M_ENSEMBLE}  seeds={SEEDS}")
print(f"{'='*60}")

for m, seed in enumerate(SEEDS):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    print(f"\n--- Model {m+1}/{M_ENSEMBLE}  seed={seed} ---")

    model     = IndicatorGNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=0.5, patience=8, min_lr=1e-5)

    best_val   = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        loss = bce(model(x_t, edge_index)[tr_t], y_t[tr_t])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = bce(model(x_t, edge_index)[vl_t], y_t[vl_t]).item()
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

        if epoch % 40 == 0 or epoch == 1:
            print(f"  epoch {epoch:>3}  train={loss.item():.4f}  val={val_loss:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        all_probs[m] = torch.sigmoid(model(x_t, edge_index)).cpu().numpy()
    all_states.append(best_state)

    auc = roc_auc_score(y_np[vl_idx], all_probs[m, vl_idx])
    f1  = f1_score(y_np[vl_idx], (all_probs[m, vl_idx] > 0.5).astype(int))
    print(f"  Individual  AUC={auc:.3f}  F1={f1:.3f}")

    h1, h2 = model.embeddings(x_t, edge_index)
    h1_train_ens.append(h1.cpu().numpy().astype(np.float32))
    h2_train_ens.append(h2.cpu().numpy().astype(np.float32))

# ── 7. Per-node certainty & spread ───────────────────────────────────────────
# certainty = mean P across models = fraction of "votes" for SS
# spread    = std across models     = how much models disagree
p_mean    = all_probs.mean(axis=0)           # (N,)  certainty 0–1
p_std     = all_probs.std(axis=0)            # (N,)  disagreement
p_vote    = (all_probs > 0.5).mean(axis=0)  # (N,)  fraction voting SS

pred_ens = (p_mean > 0.5).astype(int)

print(f"\n{'='*60}")
print("  ENSEMBLE RESULTS (certainty = mean of M predictions)")
print(f"{'='*60}")
for split, idx in [("Train", tr_idx), ("Test ", vl_idx)]:
    auc = roc_auc_score(y_np[idx], p_mean[idx])
    f1  = f1_score(y_np[idx], pred_ens[idx])
    acc = (pred_ens[idx] == y_np[idx]).mean()
    print(f"  {split}: AUC={auc:.3f}  F1={f1:.3f}  Acc={acc:.3f}"
          f"  avg_certainty={p_mean[idx].mean()*100:.1f}%"
          f"  avg_spread={p_std[idx].mean():.4f}")

# ── 8. Plots ──────────────────────────────────────────────────────────────────
# Running convergence: mean certainty and spread as M grows (LLN in action)
running_cert  = [all_probs[:m+1, vl_idx].mean(axis=0).mean() * 100
                 for m in range(M_ENSEMBLE)]
running_spread = [all_probs[:m+1, vl_idx].std(axis=0).mean()
                  for m in range(M_ENSEMBLE)]

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# (a) LLN convergence
ax  = axes[0]
ax2 = ax.twinx()
ax.plot(range(1, M_ENSEMBLE+1), running_cert,   "o-", color="steelblue", lw=2, label="Mean certainty %")
ax2.plot(range(1, M_ENSEMBLE+1), running_spread, "s--", color="coral",    lw=2, label="Spread σ")
ax.set_xlabel("Ensemble members M")
ax.set_ylabel("Mean certainty (%)", color="steelblue")
ax2.set_ylabel("Spread  σ(p)", color="coral")
ax.set_title("LLN convergence — estimate stabilises with M\n(shows when adding more models stops changing the answer)")
ax.set_xticks(range(1, M_ENSEMBLE+1))
ax.grid(True, alpha=0.3)
lines = ax.get_lines() + ax2.get_lines()
ax.legend(lines, [l.get_label() for l in lines], fontsize=8)

# (b) Certainty per test node 0–100 %
sc1 = axes[1].scatter(df.loc[vl_idx, "X_orig"], df.loc[vl_idx, "Z_orig"],
                      c=p_mean[vl_idx]*100, cmap="RdYlGn", s=3, vmin=0, vmax=100)
plt.colorbar(sc1, ax=axes[1], label="Certainty  P̄(is_SS)  [%]")
axes[1].set_xlabel("X (m)"); axes[1].set_ylabel("Z (m asl)")
axes[1].set_title(f"Per-interval certainty  [M={M_ENSEMBLE}]\n(green=confident SS, red=confident not-SS)")

# (c) Spread — where models disagree
sc2 = axes[2].scatter(df.loc[vl_idx, "X_orig"], df.loc[vl_idx, "Z_orig"],
                      c=p_std[vl_idx], cmap="plasma", s=3)
plt.colorbar(sc2, ax=axes[2], label="Spread σ  (model disagreement)")
axes[2].set_xlabel("X (m)"); axes[2].set_ylabel("Z (m asl)")
axes[2].set_title("Model disagreement\n(bright = uncertain boundary region)")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "ensemble_certainty.png"), dpi=150)
print(f"\nSaved: ensemble_uq/outputs/ensemble_certainty.png")

# ── 9. Save artefacts for predict_certainty_grid.py ──────────────────────────
torch.save(all_states, os.path.join(OUT_DIR, "gnn_ensemble.pt"))

with open(os.path.join(OUT_DIR, "gnn_ensemble_scalers.pkl"), "wb") as f:
    pickle.dump({
        "x_train"      : x_np,
        "xyz_train"    : xyz_np,
        "h1_train_ens" : h1_train_ens,   # list[M] of (N, HIDDEN)
        "h2_train_ens" : h2_train_ens,
        "K"            : K,
        "NODE_DIM"     : NODE_DIM,
        "HIDDEN"       : HIDDEN,
        "M_ENSEMBLE"   : M_ENSEMBLE,
        "SEEDS"        : SEEDS,
    }, f)

print(f"Saved: ensemble_uq/outputs/gnn_ensemble.pt  ({M_ENSEMBLE} model states)")
print(f"Saved: ensemble_uq/outputs/gnn_ensemble_scalers.pkl")
print("\n=== DEEP ENSEMBLE TRAINING COMPLETE ===")
