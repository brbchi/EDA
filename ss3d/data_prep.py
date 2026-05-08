"""
data_prep.py  --  EDA/ss3d
Interval-level data preparation for 3-D SS indicator GNN.

Each composite interval becomes a graph node.  The target is is_SS (0/1).
The GNN learns the 3-D spatial continuity of the SS layer exactly as
Chauke (2026) learned iron-grade continuity — just replace grade with the
SS presence indicator.

Z_HW = Z of the first SS composite interval  (no arithmetic, raw Z column)
Z_FW = Z of the last  SS composite interval

These are saved in hole_targets.csv for the visualisation comparison only.
The GNN never uses them directly — it predicts is_SS everywhere, and
Z_HW / Z_FW are derived from the resulting 3-D probability field.

Outputs (outputs/)
  df_model.csv         84 990 rows, one per composite interval
  hole_targets.csv     132 rows, one per drillhole
  feature_scaler.joblib
"""

import os, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

warnings.filterwarnings("ignore")
np.random.seed(42)

ROOT    = os.path.dirname(os.path.abspath(__file__))
EDA     = os.path.join(ROOT, "..")
OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Load ───────────────────────────────────────────────────────────────────
collar = pd.read_csv(os.path.join(EDA, "collar.csv"))
comp   = pd.read_csv(os.path.join(EDA, "Composites.csv"))

collar.rename(columns={"Nom": "Drillhole"}, inplace=True)
comp.rename(columns={
    "Sample Number": "Sample_Num",
    "Drillholes":    "Drillhole",
    "Depth From":    "Depth_From",
    "Depth To":      "Depth_To",
    "Lithologies":   "Lithology",
}, inplace=True)

comp["Z"]      = pd.to_numeric(comp["Z"],      errors="coerce")
comp["Length"] = pd.to_numeric(comp["Length"], errors="coerce")

print(f"Collar     : {collar.shape[0]} holes")
print(f"Composites : {comp.shape[0]} intervals")

# ── 2. Clean ──────────────────────────────────────────────────────────────────
n0 = len(comp)
comp.dropna(subset=["Lithology", "Z", "Length"], inplace=True)
print(f"Dropped {n0 - len(comp)} bad rows -> {len(comp)} remain")

# ── 3. True Z_HW / Z_FW (for visualisation only) ─────────────────────────────
ss = comp[comp["Lithology"] == "SS"].copy()

first_ss    = ss.loc[ss.groupby("Drillhole")["Depth_From"].idxmin()].set_index("Drillhole")
Z_HW_series = first_ss["Z"]   # raw Z of first SS interval

last_ss     = ss.loc[ss.groupby("Drillhole")["Depth_To"].idxmax()].set_index("Drillhole")
Z_FW_series = last_ss["Z"]    # raw Z of last SS interval

ss_thick = Z_HW_series - Z_FW_series

hw_fw = pd.DataFrame({
    "Drillhole"     : Z_HW_series.index,
    "Z_HW"          : Z_HW_series.values,
    "Z_FW"          : Z_FW_series.values,
    "ss_thickness"  : ss_thick.values,
    "ss_encountered": 1,
}).reset_index(drop=True)

no_ss_ids = collar[~collar["Drillhole"].isin(hw_fw["Drillhole"])]["Drillhole"].tolist()
no_ss = pd.DataFrame({
    "Drillhole":      no_ss_ids,
    "Z_HW":           np.nan,
    "Z_FW":           np.nan,
    "ss_thickness":   np.nan,
    "ss_encountered": 0,
})
hole_targets = pd.concat([hw_fw, no_ss], ignore_index=True)

enc1 = hole_targets[hole_targets["ss_encountered"] == 1]
print(f"\nHoles with SS    : {len(enc1)}")
print(f"Holes without SS : {len(no_ss)}")
print(f"Z_HW range : [{enc1['Z_HW'].min():.1f}, {enc1['Z_HW'].max():.1f}] m asl")
print(f"Z_FW range : [{enc1['Z_FW'].min():.1f}, {enc1['Z_FW'].max():.1f}] m asl")

# ── 4. Merge collar info into composites ──────────────────────────────────────
comp = comp.merge(collar[["Drillhole", "Z_g_earth", "Longueur"]],
                  on="Drillhole", how="left")

# ── 5. Interval features ──────────────────────────────────────────────────────
comp["MID_DEPTH"]       = (comp["Depth_From"] + comp["Depth_To"]) / 2.0
comp["Z_mid"]           = comp["Z"].astype(float)          # elevation (m asl) at interval midpoint
comp["interval_length"] = comp["Length"].astype(float)
comp["depth_ratio"]     = comp["MID_DEPTH"] / comp["Longueur"].clip(lower=1e-6)
comp["is_SS"]           = (comp["Lithology"] == "SS").astype(int)

# Preserve original (unscaled) spatial coords for graph construction
comp["X_orig"]         = comp["X"]
comp["Y_orig"]         = comp["Y"]
comp["Z_orig"]         = comp["Z"]    # = Z_mid (elevation at midpoint)
comp["Z_g_earth_orig"] = comp["Z_g_earth"]

ss_frac = comp["is_SS"].mean()
print(f"\nis_SS fraction : {ss_frac:.3f}  ({comp['is_SS'].sum():,} SS / {len(comp):,} total)")

# ── 6. Scale spatial + interval features ──────────────────────────────────────
SCALE_COLS = ["X", "Y", "Z_mid", "MID_DEPTH", "interval_length", "depth_ratio"]
scaler = StandardScaler()
comp[SCALE_COLS] = scaler.fit_transform(comp[SCALE_COLS])
joblib.dump(scaler, os.path.join(OUT_DIR, "feature_scaler.joblib"))
print(f"Feature scaler saved ({len(SCALE_COLS)} columns).")

# ── 7. Assemble df_model ──────────────────────────────────────────────────────
KEEP = [
    "Sample_Num", "Drillhole",
    "Depth_From", "Depth_To",
    "X", "Y", "Z_mid", "MID_DEPTH", "interval_length", "depth_ratio",
    "is_SS", "Z_g_earth",
    "X_orig", "Y_orig", "Z_orig", "Z_g_earth_orig",
]
df_model = comp[KEEP].reset_index(drop=True)

# ── 8. Save ───────────────────────────────────────────────────────────────────
df_model.to_csv(os.path.join(OUT_DIR, "df_model.csv"), index=False)

hole_summary = hole_targets.merge(
    collar[["Drillhole", "X", "Y", "Z_g_earth", "Longueur"]], on="Drillhole", how="left")
hole_summary.to_csv(os.path.join(OUT_DIR, "hole_targets.csv"), index=False)

print(f"\ndf_model     : {df_model.shape}  -> outputs/df_model.csv")
print(f"hole_targets : {hole_summary.shape} -> outputs/hole_targets.csv")
print("\n=== DATA PREP COMPLETE ===")
