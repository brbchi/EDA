"""
train_ensemble.py  --  EDA/ensemble_uq

Deep Ensemble Uncertainty Quantification for the SS-layer GNN.

Theory:
  Train M=100 independent GNN models with different random seeds.
  Each model starts from a different random initialisation and
  therefore converges to a different loss basin.

  By the Law of Large Numbers:
    certainty(voxel) = (1/M) Σ p_m(voxel)  →  E[P(SS | voxel)]  as M → ∞

  With M=100, the estimate is well-converged and P10/P50/P90 percentiles
  across realizations are reliable — the standard requirement in
  geostatistical uncertainty reporting.

Per-voxel outputs (predict_certainty_grid.py):
    certainty_pct   0–100 %  mean P(SS) across 100 models
    spread          std across models
    p10 / p50 / p90 percentile realizations

Checkpointing:
  Each model is saved to outputs/checkpoints/model_{m:03d}.pt as it
  finishes.  If training is interrupted, re-running this script resumes
  automatically from the last completed model.
"""

import os, random, warnings, pickle
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
HERE     = os.path.dirname(os.path.abspath(__file__))
SS3D_OUT = os.path.join(HERE, "..", "ss3d", "outputs")
OUT_DIR  = os.path.join(HERE, "outputs")
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")
os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# ── Hyper-parameters ──────────────────────────────────────────────────────────
M_ENSEMBLE = 100

# 100 seeds generated deterministically from a master seed — reproducible
_rng  = np.random.default_rng(0)
SEEDS = _rng.integers(0, 1_000_000, size=M_ENSEMBLE).tolist()

K        = 15
HIDDEN   = 64
LR       = 1e-3
EPOCHS   = 200
PATIENCE = 20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}  |  M={M_ENSEMBLE} realizations")

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

# ── 3. Build 3-D kNN graph — done ONCE ───────────────────────────────────────
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

# ── 5. Model ──────────────────────────────────────────────────────────────────
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
        with torch.no_grad():
            h1 = self.norm1(self.conv1(x, ei))
            h1 = F.relu(h1 + self.proj(x))
            h2 = self.norm2(self.conv2(h1, ei))
        return h1, h2


bce  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
tr_t = torch.tensor(tr_idx, dtype=torch.long)
vl_t = torch.tensor(vl_idx, dtype=torch.long)

# ── 6. Deep Ensemble loop with checkpointing ──────────────────────────────────
# If a checkpoint exists for model m, it is loaded instead of re-trained.
# This lets you resume after an interruption.

all_probs    = np.zeros((M_ENSEMBLE, N), dtype=np.float32)
all_states   = [None] * M_ENSEMBLE
h1_train_ens = [None] * M_ENSEMBLE
h2_train_ens = [None] * M_ENSEMBLE

# How many are already done?
completed = [os.path.exists(os.path.join(CKPT_DIR, f"model_{m:03d}.pt"))
             for m in range(M_ENSEMBLE)]
n_done = sum(completed)
print(f"\n{'='*60}")
print(f"  DEEP ENSEMBLE  M={M_ENSEMBLE}  ({n_done} already checkpointed)")
print(f"{'='*60}")

for m, seed in enumerate(SEEDS):
    ckpt_path = os.path.join(CKPT_DIR, f"model_{m:03d}.pt")

    if completed[m]:
        # Load from checkpoint instead of retraining
        ckpt = torch.load(ckpt_path, map_location="cpu")
        all_states[m]   = ckpt["state_dict"]
        all_probs[m]    = ckpt["probs"]
        h1_train_ens[m] = ckpt["h1_train"]
        h2_train_ens[m] = ckpt["h2_train"]
        auc = roc_auc_score(y_np[vl_idx], all_probs[m, vl_idx])
        print(f"[{m+1:>3}/{M_ENSEMBLE}] loaded checkpoint  AUC={auc:.3f}")
        continue

    # ── Train this realization ────────────────────────────────────────────────
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    print(f"\n[{m+1:>3}/{M_ENSEMBLE}]  seed={seed}")

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
                print(f"  Early stop at epoch {epoch}  best_val={best_val:.4f}")
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        all_probs[m] = torch.sigmoid(model(x_t, edge_index)).cpu().numpy()
    all_states[m] = best_state

    auc = roc_auc_score(y_np[vl_idx], all_probs[m, vl_idx])
    f1  = f1_score(y_np[vl_idx], (all_probs[m, vl_idx] > 0.5).astype(int))
    print(f"  AUC={auc:.3f}  F1={f1:.3f}")

    h1, h2 = model.embeddings(x_t, edge_index)
    h1_train_ens[m] = h1.cpu().numpy().astype(np.float32)
    h2_train_ens[m] = h2.cpu().numpy().astype(np.float32)

    # Save checkpoint immediately so progress is never lost
    torch.save({
        "state_dict": best_state,
        "probs"     : all_probs[m],
        "h1_train"  : h1_train_ens[m],
        "h2_train"  : h2_train_ens[m],
        "seed"      : seed,
        "auc"       : auc,
    }, ckpt_path)

print(f"\nAll {M_ENSEMBLE} realizations complete.")

# ── 7. Aggregate across 100 realizations ─────────────────────────────────────
print(f"\n{'='*60}")
print("  ENSEMBLE AGGREGATE  (100 realizations)")
print(f"{'='*60}")

p_mean = all_probs.mean(axis=0)          # certainty
p_std  = all_probs.std(axis=0)           # spread
p_p10  = np.percentile(all_probs, 10, axis=0)
p_p50  = np.percentile(all_probs, 50, axis=0)
p_p90  = np.percentile(all_probs, 90, axis=0)

pred_ens = (p_mean > 0.5).astype(int)

for split, idx in [("Train", tr_idx), ("Test ", vl_idx)]:
    auc = roc_auc_score(y_np[idx], p_mean[idx])
    f1  = f1_score(y_np[idx], pred_ens[idx])
    print(f"  {split}: AUC={auc:.3f}  F1={f1:.3f}"
          f"  certainty={p_mean[idx].mean()*100:.1f}%"
          f"  spread={p_std[idx].mean():.4f}")

# ── 8. LLN convergence — show how mean and spread stabilise with M ────────────
# Sample M checkpoints [5,10,20,50,100] to trace the convergence
milestones     = [m for m in [5, 10, 20, 30, 50, 75, 100] if m <= M_ENSEMBLE]
running_mean   = [all_probs[:m, vl_idx].mean(axis=0).mean() * 100 for m in milestones]
running_spread = [all_probs[:m, vl_idx].std(axis=0).mean()        for m in milestones]

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

ax  = axes[0]
ax2 = ax.twinx()
ax.plot(milestones, running_mean,   "o-", color="steelblue", lw=2, label="Mean certainty %")
ax2.plot(milestones, running_spread, "s--", color="coral",   lw=2, label="Spread σ")
ax.set_xlabel("Realizations M")
ax.set_ylabel("Mean certainty (%)", color="steelblue")
ax2.set_ylabel("Spread  σ", color="coral")
ax.set_title(f"LLN convergence over {M_ENSEMBLE} realizations")
ax.grid(True, alpha=0.3)
lines = ax.get_lines() + ax2.get_lines()
ax.legend(lines, [l.get_label() for l in lines], fontsize=8)

sc1 = axes[1].scatter(df.loc[vl_idx, "X_orig"], df.loc[vl_idx, "Z_orig"],
                      c=p_mean[vl_idx]*100, cmap="RdYlGn", s=3, vmin=0, vmax=100)
plt.colorbar(sc1, ax=axes[1], label="Certainty P̄(is_SS) [%]")
axes[1].set_xlabel("X (m)"); axes[1].set_ylabel("Z (m asl)")
axes[1].set_title(f"Mean certainty  [{M_ENSEMBLE} realizations]")

sc2 = axes[2].scatter(df.loc[vl_idx, "X_orig"], df.loc[vl_idx, "Z_orig"],
                      c=(p_p90 - p_p10)[vl_idx], cmap="plasma", s=3)
plt.colorbar(sc2, ax=axes[2], label="P90 − P10  (uncertainty interval)")
axes[2].set_xlabel("X (m)"); axes[2].set_ylabel("Z (m asl)")
axes[2].set_title("P90 − P10 uncertainty interval")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "ensemble_certainty.png"), dpi=150)
print(f"\nSaved: ensemble_uq/outputs/ensemble_certainty.png")

# ── 9. PDF illustrations — 4 representative voxels ───────────────────────────
# Pick one node from each category in the test set:
#   (a) confident SS         p_mean > 0.80
#   (b) uncertain boundary   0.40 < p_mean < 0.60
#   (c) confident not-SS     p_mean < 0.20
#   (d) highest spread       most disagreement across models
def _gaussian_kde(data, xs):
    """Silverman's rule KDE — no scipy needed."""
    bw = 1.06 * data.std() * len(data) ** (-0.2)
    bw = max(bw, 1e-4)
    k  = np.exp(-0.5 * ((xs[:, None] - data[None, :]) / bw) ** 2)
    return k.sum(axis=1) / (len(data) * bw * np.sqrt(2 * np.pi))

def _pick(mask):
    """Return index of node with median spread among those matching mask."""
    candidates = vl_idx[mask]
    if len(candidates) == 0:
        return vl_idx[0]
    spreads = p_std[candidates]
    return candidates[np.argsort(spreads)[len(spreads) // 2]]

vl_mean  = p_mean[vl_idx]
vl_std   = p_std[vl_idx]

node_a = _pick(vl_mean > 0.80)                              # confident SS
node_b = _pick((vl_mean > 0.40) & (vl_mean < 0.60))        # uncertain
node_c = _pick(vl_mean < 0.20)                              # confident not-SS
node_d = vl_idx[np.argmax(vl_std)]                          # most uncertain

nodes  = [node_a, node_b, node_c, node_d]
titles = ["Confident SS\n(p̄ > 0.80)",
          "Uncertain boundary\n(0.40 < p̄ < 0.60)",
          "Confident not-SS\n(p̄ < 0.20)",
          "Maximum uncertainty\n(highest σ)"]
colors = ["#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]

fig, axes = plt.subplots(1, 4, figsize=(18, 4))
fig.suptitle(f"PDF of P(is_SS) across {M_ENSEMBLE} realizations — representative voxels",
             fontsize=12, fontweight="bold")

for ax, node, title, color in zip(axes, nodes, titles, colors):
    vals = all_probs[:, node]           # (100,) — one value per realization

    # Histogram (the 100 realizations)
    ax.hist(vals, bins=20, range=(0, 1), density=True,
            color=color, alpha=0.4, edgecolor="white", label="Realizations")

    # KDE — smooth PDF
    if vals.std() > 0.005:
        xs  = np.linspace(0, 1, 300)
        kde = _gaussian_kde(vals, xs)
        ax.plot(xs, kde, color=color, lw=2.5, label="PDF (KDE)")

    # P10 / P50 / P90 lines
    for pct, ls, lab in [(10, "--", "P10"), (50, "-", "P50"), (90, ":", "P90")]:
        v = np.percentile(vals, pct)
        ax.axvline(v, color="black", ls=ls, lw=1.2, alpha=0.8)
        ax.text(v + 0.01, ax.get_ylim()[1] * 0.95, lab,
                fontsize=7, va="top", color="black")

    ax.axvline(0.5, color="gray", lw=0.8, alpha=0.5)   # decision boundary
    ax.set_xlim(0, 1)
    ax.set_xlabel("P(is_SS)")
    ax.set_ylabel("Density")
    ax.set_title(f"{title}\np̄={vals.mean():.2f}  σ={vals.std():.3f}",
                 fontsize=9)
    ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "pdf_voxels.png"), dpi=150)
print("Saved: ensemble_uq/outputs/pdf_voxels.png")

# ── 10. Violin plot — PDF shape across 3 regions ─────────────────────────────
# Show the full distribution (not just mean/std) grouped by certainty level
high_mask = p_mean[vl_idx] > 0.65
low_mask  = p_mean[vl_idx] < 0.35
mid_mask  = ~high_mask & ~low_mask

groups = {
    f"Confident SS\n(n={high_mask.sum()})": all_probs[:, vl_idx[high_mask]].ravel(),
    f"Uncertain\n(n={mid_mask.sum()})":     all_probs[:, vl_idx[mid_mask]].ravel(),
    f"Confident not-SS\n(n={low_mask.sum()})": all_probs[:, vl_idx[low_mask]].ravel(),
}

fig2, ax = plt.subplots(figsize=(9, 5))
parts = ax.violinplot(list(groups.values()), positions=[1, 2, 3],
                      showmedians=True, showextrema=True)
violin_colors = ["#2ca02c", "#ff7f0e", "#d62728"]
for pc, col in zip(parts["bodies"], violin_colors):
    pc.set_facecolor(col); pc.set_alpha(0.6)
parts["cmedians"].set_color("black")
parts["cmaxes"].set_color("gray")
parts["cmins"].set_color("gray")
parts["cbars"].set_color("gray")

ax.set_xticks([1, 2, 3])
ax.set_xticklabels(list(groups.keys()), fontsize=9)
ax.set_ylabel("P(is_SS) across 100 realizations")
ax.set_ylim(0, 1)
ax.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.6, label="Decision boundary")
ax.set_title(f"Distribution of realizations by certainty region  [M={M_ENSEMBLE}]")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "pdf_violin.png"), dpi=150)
print("Saved: ensemble_uq/outputs/pdf_violin.png")

# ── 11. Save final artefacts ──────────────────────────────────────────────────
torch.save(all_states, os.path.join(OUT_DIR, "gnn_ensemble.pt"))

with open(os.path.join(OUT_DIR, "gnn_ensemble_scalers.pkl"), "wb") as f:
    pickle.dump({
        "x_train"      : x_np,
        "xyz_train"    : xyz_np,
        "h1_train_ens" : h1_train_ens,
        "h2_train_ens" : h2_train_ens,
        "K"            : K,
        "NODE_DIM"     : NODE_DIM,
        "HIDDEN"       : HIDDEN,
        "M_ENSEMBLE"   : M_ENSEMBLE,
        "SEEDS"        : SEEDS,
        "p_p10"        : p_p10,
        "p_p50"        : p_p50,
        "p_p90"        : p_p90,
    }, f)

print(f"Saved: gnn_ensemble.pt  ({M_ENSEMBLE} model states)")
print(f"Saved: gnn_ensemble_scalers.pkl")
print("\n=== 100-REALIZATION DEEP ENSEMBLE COMPLETE ===")
