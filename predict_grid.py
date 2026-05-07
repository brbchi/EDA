"""
predict_grid.py
Batch GNN inference for simulated (X, Y) query points.
Z is ignored — kNN uses 2-D Euclidean distance (X, Y only) and the
scaled Z node feature is fixed to 0 (= training mean in scaled space).
Loads gnn_model.pt + gnn_scalers.pkl saved by train_gnn.py.

Output: outputs/predicted_grid.csv
  Columns: Predicted_X, Predicted_Y, depth_at_HW, depth_at_fw
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

CHUNK  = 500
ROOT   = os.path.dirname(os.path.abspath(__file__))
OUT    = os.path.join(ROOT, "outputs")

# ── 1. Load saved scalers + graph tensors ─────────────────────────────────────
with open(os.path.join(OUT, "gnn_scalers.pkl"), "rb") as f:
    sc = pickle.load(f)

collar_xyz      = sc["collar_xyz"]
hole_feats      = torch.tensor(sc["hole_feats"],   dtype=torch.float32)
seq_padded      = torch.tensor(sc["seq_padded"],   dtype=torch.float32)
seq_lengths     = torch.tensor(sc["seq_lengths"],  dtype=torch.long)
K               = sc["K"]
max_len         = sc["max_len"]
N_INTERVAL_FEAT = sc["N_INTERVAL_FEAT"]
SCALAR_DIM      = sc["SCALAR_DIM"]
tgt_scaler      = sc["tgt_scaler"]

feat_scaler = joblib.load(os.path.join(OUT, "feature_scaler.joblib"))
xy_mean  = feat_scaler.mean_[:2].astype(np.float32)
xy_scale = feat_scaler.scale_[:2].astype(np.float32)

N_COLLARS = len(collar_xyz)
HIDDEN    = 128
NODE_DIM  = 64 + SCALAR_DIM

print(f"Collars: {N_COLLARS}  |  K={K}  |  max_seq_len={max_len}")

# ── 2. Reconstruct model ──────────────────────────────────────────────────────
class IntervalEncoder(nn.Module):
    def __init__(self, in_dim, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, batch_first=True, bidirectional=True)

    def forward(self, x_pad, lengths):
        packed = pack_padded_sequence(x_pad, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return torch.cat([h[0], h[1]], dim=-1)


class DrillholeGNN(nn.Module):
    def __init__(self, hidden=HIDDEN):
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

# ── 3. Collar->collar edges (2-D XY) ─────────────────────────────────────────
collar_xy = collar_xyz[:, :2]
nbrs_cc   = NearestNeighbors(n_neighbors=K + 1, algorithm="ball_tree").fit(collar_xy)
_, idx_cc = nbrs_cc.kneighbors(collar_xy)
idx_cc    = idx_cc[:, 1:]
src_cc    = np.repeat(np.arange(N_COLLARS), K)
dst_cc    = idx_cc.ravel()
ei_cc     = np.stack([src_cc, dst_cc])

nbrs_q = NearestNeighbors(n_neighbors=K, algorithm="ball_tree").fit(collar_xy)

# ── 4. Load query points ──────────────────────────────────────────────────────
query_df = pd.read_csv(os.path.join(ROOT, "Simulated_with_Predicted_XY_Enhanced.csv"))
query_xy = query_df[["Predicted_X", "Predicted_Y"]].values.astype(np.float32)
N_QUERY  = len(query_xy)
print(f"Query points: {N_QUERY:,}  |  chunk size: {CHUNK}")

hw_out = np.full(N_QUERY, np.nan, dtype=np.float32)
fw_out = np.full(N_QUERY, np.nan, dtype=np.float32)

# ── 5. Chunked inference ──────────────────────────────────────────────────────
t0       = time.time()
n_chunks = (N_QUERY + CHUNK - 1) // CHUNK

for ci in range(n_chunks):
    start = ci * CHUNK
    end   = min(start + CHUNK, N_QUERY)
    chunk = query_xy[start:end]
    nc    = len(chunk)

    _, idx_qc = nbrs_q.kneighbors(chunk)
    src_qc = np.repeat(np.arange(N_COLLARS, N_COLLARS + nc), K)
    dst_qc = idx_qc.ravel()
    src_cq = dst_qc;  dst_cq = src_qc

    ei_full = torch.tensor(
        np.hstack([ei_cc, np.stack([src_qc, dst_qc]), np.stack([src_cq, dst_cq])]),
        dtype=torch.long
    )

    q_xy_s   = (chunk - xy_mean) / xy_scale
    q_scalar = np.zeros((nc, SCALAR_DIM), dtype=np.float32)
    q_scalar[:, :2] = q_xy_s
    hf_aug   = torch.cat([hole_feats, torch.tensor(q_scalar, dtype=torch.float32)], dim=0)

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

# ── 6. Save ───────────────────────────────────────────────────────────────────
query_df = query_df[["Predicted_X", "Predicted_Y"]].copy()
query_df["depth_at_HW"] = hw_out
query_df["depth_at_fw"]  = fw_out
out_path = os.path.join(OUT, "predicted_grid.csv")
query_df.to_csv(out_path, index=False)

elapsed = time.time() - t0
print(f"\nDone in {elapsed/60:.1f} min  |  Saved: {out_path}")
print(f"\nPrediction summary:")
print(f"  depth_at_HW : mean={hw_out.mean():.1f}m  std={hw_out.std():.1f}m  "
      f"range=[{hw_out.min():.1f}, {hw_out.max():.1f}]")
print(f"  depth_at_fw : mean={fw_out.mean():.1f}m  std={fw_out.std():.1f}m  "
      f"range=[{fw_out.min():.1f}, {fw_out.max():.1f}]")
