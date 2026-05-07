"""
predict_grid.py  --  EDA/zelev  (Z-conditioned GNN inference)
Batch GNN inference for simulated (X, Y) query points.

Collar nodes carry their true Z_HW / Z_FW in features (features [10,11]).
Query nodes have 0 for those features.  The GNN propagates known SS
intercept elevations spatially to predict undrilled locations.

IDW estimates the surface elevation at each query point so that the
position feature [X, Y, Z_surface] has the correct elevation context.

Output: outputs/predicted_grid.csv
  Columns: Predicted_X, Predicted_Y, Z_HW_elev, Z_FW_elev
"""

import os, time, pickle, warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import EdgeConv
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
import pandas as pd
import joblib
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

CHUNK = 500
ROOT  = os.path.dirname(os.path.abspath(__file__))
EDA   = os.path.join(ROOT, "..")
OUT   = os.path.join(ROOT, "outputs")

# ── 1. Load saved artefacts ───────────────────────────────────────────────────
with open(os.path.join(OUT, "gnn_scalers.pkl"), "rb") as f:
    sc = pickle.load(f)

collar_xyz      = sc["collar_xyz"]
hole_feats      = torch.tensor(sc["hole_feats"],   dtype=torch.float32)
seq_padded      = torch.tensor(sc["seq_padded"],   dtype=torch.float32)
seq_lengths     = torch.tensor(sc["seq_lengths"],  dtype=torch.long)
K               = sc["K"]
max_len         = sc["max_len"]
N_INTERVAL_FEAT = sc["N_INTERVAL_FEAT"]
SCALAR_DIM      = sc["SCALAR_DIM"]   # 12
tgt_scaler      = sc["tgt_scaler"]

feat_scaler = joblib.load(os.path.join(OUT, "feature_scaler.joblib"))
xyz_mean    = feat_scaler.mean_[:3].astype(np.float32)
xyz_scale   = feat_scaler.scale_[:3].astype(np.float32)

N_COLLARS = len(collar_xyz)
HIDDEN    = 128
NODE_DIM  = 64 + SCALAR_DIM   # 76

print(f"Collars: {N_COLLARS}  |  K={K}  |  SCALAR_DIM={SCALAR_DIM}  |  max_seq_len={max_len}")

# ── 2. IDW surface-Z estimator ────────────────────────────────────────────────
collar_raw    = pd.read_csv(os.path.join(EDA, "collar.csv"))
collar_raw.rename(columns={"Nom": "Drillhole"}, inplace=True)
collar_xy_idw = collar_raw[["X", "Y"]].values.astype(np.float32)
collar_z_idw  = collar_raw["Z_g_earth"].values.astype(np.float32)
print(f"IDW: {len(collar_raw)} collars  "
      f"(Z range {collar_z_idw.min():.0f}-{collar_z_idw.max():.0f} m asl)")


def idw_z(query_xy: np.ndarray, power: float = 2.0) -> np.ndarray:
    dx = query_xy[:, 0:1] - collar_xy_idw[np.newaxis, :, 0]
    dy = query_xy[:, 1:2] - collar_xy_idw[np.newaxis, :, 1]
    d2 = dx**2 + dy**2 + 1e-6
    w  = 1.0 / d2 ** (power / 2.0)
    return (w * collar_z_idw).sum(axis=1) / w.sum(axis=1)


# ── 3. Reconstruct model ──────────────────────────────────────────────────────
class IntervalEncoder(nn.Module):
    def __init__(self, in_dim, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, batch_first=True, bidirectional=True)

    def forward(self, x_pad, lengths):
        packed = pack_padded_sequence(x_pad, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return torch.cat([h[0], h[1]], dim=-1)


class DrillholeGNN(nn.Module):
    def __init__(self, hidden=HIDDEN):
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
        self.head  = nn.Linear(hidden, 2)

    def forward(self, seq_pad, seq_len, h_feats, ei):
        emb = self.enc(seq_pad, seq_len)
        x   = torch.cat([emb, h_feats], dim=-1)
        h1  = self.drop(self.norm1(self.conv1(x, ei)))
        h1  = F.relu(h1 + self.proj(x))
        h2  = self.drop(self.norm2(self.conv2(h1, ei)))
        return self.head(F.relu(h2))


model = DrillholeGNN()
model.load_state_dict(torch.load(os.path.join(OUT, "gnn_model.pt"), map_location="cpu"))
model.eval()
print("Model loaded.")

# ── 4. Collar-collar edges ────────────────────────────────────────────────────
collar_xy = collar_xyz[:, :2]
nbrs_cc   = NearestNeighbors(n_neighbors=K + 1, algorithm="ball_tree").fit(collar_xy)
_, idx_cc = nbrs_cc.kneighbors(collar_xy)
idx_cc    = idx_cc[:, 1:]
src_cc    = np.repeat(np.arange(N_COLLARS), K)
dst_cc    = idx_cc.ravel()
ei_cc     = np.stack([src_cc, dst_cc])

nbrs_q = NearestNeighbors(n_neighbors=K, algorithm="ball_tree").fit(collar_xy)

# ── 5. Load query points ──────────────────────────────────────────────────────
query_df = pd.read_csv(os.path.join(EDA, "Simulated_with_Predicted_XY_Enhanced.csv"))
query_xy = query_df[["Predicted_X", "Predicted_Y"]].values.astype(np.float32)
N_QUERY  = len(query_xy)
print(f"\nQuery points: {N_QUERY:,}  |  chunk size: {CHUNK}")

hw_out = np.full(N_QUERY, np.nan, dtype=np.float32)
fw_out = np.full(N_QUERY, np.nan, dtype=np.float32)

# ── 6. Chunked inference ──────────────────────────────────────────────────────
# hole_feats already contains unmasked Z_HW/Z_FW for all 122 SS collar nodes
# (features [10,11]).  Query nodes get zeros for those positions.

t0       = time.time()
n_chunks = (N_QUERY + CHUNK - 1) // CHUNK

for ci in range(n_chunks):
    start = ci * CHUNK
    end   = min(start + CHUNK, N_QUERY)
    chunk = query_xy[start:end]
    nc    = len(chunk)

    z_surf   = idw_z(chunk).astype(np.float32)
    q_xyz    = np.column_stack([chunk, z_surf])
    q_xyz_s  = (q_xyz - xyz_mean) / xyz_scale

    # Query node scalar features: [X_s, Y_s, Z_s, 0...0] — 12 dims
    # positions [10,11] (Z_HW, Z_FW) remain 0 → query nodes have no known Z
    q_scalar = np.zeros((nc, SCALAR_DIM), dtype=np.float32)
    q_scalar[:, :3] = q_xyz_s

    _, idx_qc = nbrs_q.kneighbors(chunk)
    src_qc = np.repeat(np.arange(N_COLLARS, N_COLLARS + nc), K)
    dst_qc = idx_qc.ravel()
    src_cq = dst_qc.copy(); dst_cq = src_qc.copy()

    ei_full = torch.tensor(
        np.hstack([ei_cc,
                   np.stack([src_qc, dst_qc]),
                   np.stack([src_cq, dst_cq])]),
        dtype=torch.long,
    )

    hf_aug = torch.cat([hole_feats,
                        torch.tensor(q_scalar, dtype=torch.float32)], dim=0)

    q_seq  = torch.zeros(nc, max_len, N_INTERVAL_FEAT)
    q_len  = torch.ones(nc, dtype=torch.long)
    sp_aug = torch.cat([seq_padded, q_seq], dim=0)
    sl_aug = torch.cat([seq_lengths, q_len], dim=0)

    with torch.no_grad():
        out = model(sp_aug, sl_aug, hf_aug, ei_full).numpy()

    pred = tgt_scaler.inverse_transform(out)
    hw_out[start:end] = pred[N_COLLARS:, 0]
    fw_out[start:end] = pred[N_COLLARS:, 1]

    if ci % 200 == 0 or ci == n_chunks - 1:
        elapsed = time.time() - t0
        eta     = elapsed / (ci + 1) * (n_chunks - ci - 1)
        print(f"  [{ci+1:>5}/{n_chunks}]  {end:>8,}/{N_QUERY:,} pts  "
              f"elapsed {elapsed/60:.1f}min  ETA {eta/60:.1f}min")

# ── 7. Save ───────────────────────────────────────────────────────────────────
out_df = query_df[["Predicted_X", "Predicted_Y"]].copy()
out_df["Z_HW_elev"] = hw_out
out_df["Z_FW_elev"] = fw_out
out_path = os.path.join(OUT, "predicted_grid.csv")
out_df.to_csv(out_path, index=False)

elapsed = time.time() - t0
print(f"\nDone in {elapsed/60:.1f} min  |  Saved: {out_path}")
print(f"\nPrediction summary (metres asl):")
print(f"  Z_HW_elev : mean={hw_out.mean():.1f}  std={hw_out.std():.1f}  "
      f"range=[{hw_out.min():.1f}, {hw_out.max():.1f}]")
print(f"  Z_FW_elev : mean={fw_out.mean():.1f}  std={fw_out.std():.1f}  "
      f"range=[{fw_out.min():.1f}, {fw_out.max():.1f}]")
print(f"  SS thickness: mean={(hw_out-fw_out).mean():.1f}m  "
      f"std={(hw_out-fw_out).std():.1f}m")
