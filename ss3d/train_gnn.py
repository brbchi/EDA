"""
train_gnn.py  --  EDA/ss3d  (3-D SS indicator GNN)

Each composite interval is a graph node.
Node features : [X, Y, Z_mid, MID_DEPTH, interval_length, depth_ratio]  (all scaled)
Graph         : K-nearest neighbours in 3-D (X_orig, Y_orig, Z_orig)
Target        : is_SS  (binary 0/1)
Loss          : weighted BCE  (handles class imbalance)


Pre-computed training embeddings h1_train and h2_train are saved in
gnn_scalers.pkl so that predict_grid.py can do fast inference without
rebuilding the full training graph for every query batch.
"""

import os, random, warnings, pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import EdgeConv
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

SEED    = 42
K       = 15       # 3-D neighbours
HIDDEN  = 64
LR      = 1e-3
EPOCHS  = 200
PATIENCE = 20

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

ROOT    = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "outputs")

# ── 1. Load intervals ─────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(OUT_DIR, "df_model.csv"))
print(f"Intervals loaded : {df.shape[0]:,}")

NODE_FEAT_COLS = ["X", "Y", "Z_mid", "MID_DEPTH", "interval_length", "depth_ratio"]
NODE_DIM = len(NODE_FEAT_COLS)

x_np  = df[NODE_FEAT_COLS].values.astype(np.float32)       # (N, 6) scaled
xyz_np = df[["X_orig", "Y_orig", "Z_orig"]].values.astype(np.float32)  # 3-D for kNN
y_np  = df["is_SS"].values.astype(np.float32)               # (N,)

N = len(df)
print(f"SS fraction : {y_np.mean():.3f}  ({int(y_np.sum()):,} SS)")

# ── 2. Spatial holdout — middle X-strip by hole ────────────────────────────────
hole_x = df.groupby("Drillhole")["X_orig"].first()
x_lo   = np.quantile(hole_x.values, 0.40)
x_hi   = np.quantile(hole_x.values, 0.60)
test_holes = hole_x[(hole_x >= x_lo) & (hole_x <= x_hi)].index.tolist()
test_mask  = df["Drillhole"].isin(test_holes).values   # per-interval

tr_idx = np.where(~test_mask)[0]
vl_idx = np.where(test_mask)[0]
print(f"\nSpatial hold-out (middle 20 % X-strip)")
print(f"  Train intervals : {len(tr_idx):,}  ({(~test_mask).mean()*100:.1f} %)")
print(f"  Test  intervals : {len(vl_idx):,}  ({test_mask.mean()*100:.1f} %)")

# ── 3. Build 3-D kNN graph ────────────────────────────────────────────────────
print(f"\nBuilding 3-D k={K} graph on {N:,} nodes ...")
nbrs = NearestNeighbors(n_neighbors=K + 1, algorithm="ball_tree", n_jobs=-1)
nbrs.fit(xyz_np)
_, idxs = nbrs.kneighbors(xyz_np)
idxs = idxs[:, 1:]  # exclude self

src = np.repeat(np.arange(N), K)
dst = idxs.ravel()
edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long).to(DEVICE)
print(f"Graph: {N:,} nodes, {edge_index.shape[1]:,} edges")

x_t = torch.tensor(x_np,  dtype=torch.float32).to(DEVICE)
y_t = torch.tensor(y_np,  dtype=torch.float32).to(DEVICE)

# ── 4. Class-imbalance weight ──────────────────────────────────────────────────
n_pos = y_np.sum()
n_neg = N - n_pos
pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(DEVICE)
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
        return self.head(F.relu(h2)).squeeze(-1)   # (N,) logits

    def embeddings(self, x, ei):
        """Return (h1, h2) without dropout — used for fast inference."""
        with torch.no_grad():
            h1 = self.norm1(self.conv1(x, ei))
            h1 = F.relu(h1 + self.proj(x))
            h2 = self.norm2(self.conv2(h1, ei))
        return h1, h2


model     = IndicatorGNN().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, "min", factor=0.5, patience=8, min_lr=1e-5)
bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

tr_t = torch.tensor(tr_idx, dtype=torch.long)
vl_t = torch.tensor(vl_idx, dtype=torch.long)

# ── 6. Training loop ──────────────────────────────────────────────────────────
print(f"\nTraining  epochs={EPOCHS}  patience={PATIENCE}")
best_val   = float("inf")
best_state = None
no_improve = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_grad()
    logits = model(x_t, edge_index)
    loss   = bce(logits[tr_t], y_t[tr_t])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    model.eval()
    with torch.no_grad():
        logits_v = model(x_t, edge_index)
        val_loss = bce(logits_v[vl_t], y_t[vl_t]).item()
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

    if epoch % 20 == 0 or epoch == 1:
        print(f"  Epoch {epoch:>3}  train={loss.item():.4f}  val={val_loss:.4f}  best={best_val:.4f}")

# ── 7. Evaluate ───────────────────────────────────────────────────────────────
model.load_state_dict(best_state)
model.eval()

with torch.no_grad():
    logits_all = model(x_t, edge_index).cpu().numpy()

probs_all = 1 / (1 + np.exp(-logits_all))
pred_all  = (probs_all > 0.5).astype(int)

for split, idx in [("Train", tr_idx), ("Test ", vl_idx)]:
    auc = roc_auc_score(y_np[idx], probs_all[idx])
    f1  = f1_score(y_np[idx], pred_all[idx])
    acc = (pred_all[idx] == y_np[idx]).mean()
    print(f"  {split}: AUC={auc:.3f}  F1={f1:.3f}  Acc={acc:.3f}")

# ── 8. Pre-compute training embeddings for fast inference ─────────────────────
print("\nPre-computing training embeddings (h1, h2) ...")
model.eval()
h1_train, h2_train = model.embeddings(x_t, edge_index)
h1_train = h1_train.cpu().numpy().astype(np.float32)
h2_train = h2_train.cpu().numpy().astype(np.float32)
print(f"  h1: {h1_train.shape}  h2: {h2_train.shape}")

# ── 9. Save ───────────────────────────────────────────────────────────────────
torch.save(best_state, os.path.join(OUT_DIR, "gnn_model.pt"))

with open(os.path.join(OUT_DIR, "gnn_scalers.pkl"), "wb") as f:
    pickle.dump({
        "x_train"   : x_np,        # (N, 6) scaled interval features
        "xyz_train" : xyz_np,       # (N, 3) 3-D positions for kNN
        "h1_train"  : h1_train,     # (N, HIDDEN) pre-computed layer-1 embeddings
        "h2_train"  : h2_train,     # (N, HIDDEN) pre-computed layer-2 embeddings
        "K"         : K,
        "NODE_DIM"  : NODE_DIM,
        "HIDDEN"    : HIDDEN,
    }, f)

print(f"Saved: outputs/gnn_model.pt")
print(f"Saved: outputs/gnn_scalers.pkl")

# ── 10. Plot ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
with torch.no_grad():
    p_all = torch.sigmoid(torch.tensor(logits_all)).numpy()
ax.scatter(df.loc[vl_idx, "X_orig"], df.loc[vl_idx, "Z_orig"],
           c=p_all[vl_idx], cmap="RdYlGn", s=1, vmin=0, vmax=1)
ax.set_xlabel("X (m)"); ax.set_ylabel("Z mid (m asl)")
ax.set_title("Test set — predicted P(is_SS)")
plt.colorbar(ax.collections[0], ax=ax, label="P(is_SS)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "gnn_results.png"), dpi=150)
print("Saved: outputs/gnn_results.png")
print("\n=== TRAINING COMPLETE ===")
