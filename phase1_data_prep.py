"""
PHASE 1 -- Data Preprocessing
Builds df_model.csv: one row per composited 1m interval with all
features, targets, and scaling ready for GNN training.

No torch dependency -- pure pandas / sklearn / numpy.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

SEED     = 42
np.random.seed(SEED)

ROOT     = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Load raw data ──────────────────────────────────────────────────────────
collar = pd.read_csv(os.path.join(ROOT, "collar.csv"))
comp   = pd.read_csv(os.path.join(ROOT, "Composites.csv"))

# Normalise column names
comp.rename(columns={
    "Sample Number": "Sample_Num",
    "Drillholes":    "Drillhole",
    "Depth From":    "Depth_From",
    "Depth To":      "Depth_To",
    "Lithologies":   "Lithology",
}, inplace=True)
collar.rename(columns={"Nom": "Drillhole"}, inplace=True)

print(f"Collar : {collar.shape[0]} rows")
print(f"Composites: {comp.shape[0]} rows")

# ── 2. Drop NaN lithology rows (<0.05 %) ─────────────────────────────────────
n_before = len(comp)
comp.dropna(subset=["Lithology"], inplace=True)
print(f"Dropped {n_before - len(comp)} NaN-lithology rows -> {len(comp)} remain")

# ── 3. Per-hole SS targets ────────────────────────────────────────────────────
ss = comp[comp["Lithology"] == "SS"]
hw_fw = (
    ss.groupby("Drillhole")
    .agg(depth_at_HW=("Depth_From", "min"),
         depth_at_fw =("Depth_To",   "max"))
    .reset_index()
)
hw_fw["ss_thickness"]   = hw_fw["depth_at_fw"] - hw_fw["depth_at_HW"]
hw_fw["ss_encountered"] = 1

no_ss_ids = collar[~collar["Drillhole"].isin(hw_fw["Drillhole"])]["Drillhole"].tolist()
no_ss = pd.DataFrame({
    "Drillhole":      no_ss_ids,
    "depth_at_HW":    np.nan,
    "depth_at_fw":    np.nan,
    "ss_thickness":   np.nan,
    "ss_encountered": 0,
})
hole_targets = pd.concat([hw_fw, no_ss], ignore_index=True)

print(f"\nHoles with SS   : {(hole_targets['ss_encountered']==1).sum()}")
print(f"Holes without SS: {(hole_targets['ss_encountered']==0).sum()}")
print(hole_targets[["depth_at_HW","depth_at_fw","ss_thickness"]].describe().round(2))

# ── 4. Merge collar and targets into composites ───────────────────────────────
comp = comp.merge(collar[["Drillhole","Z_g_earth","Longueur"]], on="Drillhole", how="left")
comp = comp.merge(hole_targets, on="Drillhole", how="left")

# ── 5. Derived features ───────────────────────────────────────────────────────
comp["MID_DEPTH"]       = (comp["Depth_From"] + comp["Depth_To"]) / 2.0
comp["Z_mid"]           = comp["Z_g_earth"]   - comp["MID_DEPTH"]
comp["interval_length"] = comp["Depth_To"]    - comp["Depth_From"]
comp["depth_ratio"]     = comp["MID_DEPTH"]   / comp["Longueur"].clip(lower=1e-6)
comp["is_SS"]           = (comp["Lithology"] == "SS").astype(int)

# ── 6. Preserve original (unscaled) spatial coords for mapping ────────────────
comp["X_orig"]         = comp["X"]
comp["Y_orig"]         = comp["Y"]
comp["Z_g_earth_orig"] = comp["Z_g_earth"]

# ── 7. Standardise continuous features ───────────────────────────────────────
SCALE_COLS = ["X", "Y", "Z_g_earth", "MID_DEPTH", "Z_mid",
              "interval_length", "depth_ratio"]

scaler = StandardScaler()
comp[SCALE_COLS] = scaler.fit_transform(comp[SCALE_COLS])
joblib.dump(scaler, os.path.join(OUT_DIR, "feature_scaler.joblib"))
print(f"\nScaler fitted on {len(SCALE_COLS)} features and saved.")

# ── 8. Assemble df_model ──────────────────────────────────────────────────────
KEEP = [
    "Sample_Num", "Drillhole", "Depth_From", "Depth_To",
    # scaled features (node inputs)
    "X", "Y", "Z_g_earth", "MID_DEPTH", "Z_mid",
    "interval_length", "depth_ratio",
    # binary flags (unscaled)
    "is_SS", "ss_encountered",
    # targets (original depth scale)
    "depth_at_HW", "depth_at_fw", "ss_thickness",
    # original coords for spatial mapping
    "X_orig", "Y_orig", "Z_g_earth_orig",
]
df_model = comp[KEEP].reset_index(drop=True)

# ── 9. Integrity checks ───────────────────────────────────────────────────────
enc1 = df_model[df_model["ss_encountered"] == 1]
enc0 = df_model[df_model["ss_encountered"] == 0]

assert enc1[["depth_at_HW","depth_at_fw"]].notna().all().all(), \
    "ss_encountered==1 rows missing targets!"
assert enc0["depth_at_HW"].isna().all(), \
    "ss_encountered==0 rows should have NaN targets!"

raw_il = df_model["Depth_To"] - df_model["Depth_From"]
n_odd  = ((raw_il < 0.1) | (raw_il > 5.0)).sum()
assert df_model.duplicated(subset=["Drillhole","Depth_From"]).sum() == 0, \
    "Duplicate intervals found!"

print("\n=== INTEGRITY CHECKS PASSED ===")
print(f"  [OK] All {enc1['Drillhole'].nunique()} ss_encountered=1 holes have HW/FW targets")
print(f"  [OK] All {enc0['Drillhole'].nunique()} ss_encountered=0 holes have NaN targets")
print(f"  [OK] Interval length range: {raw_il.min():.2f} - {raw_il.max():.2f} m  "
      f"({n_odd} outliers outside 0.1-5m)")
print(f"  [OK] No duplicate (Drillhole, Depth_From) entries")

# ── 10. Save ──────────────────────────────────────────────────────────────────
df_path = os.path.join(OUT_DIR, "df_model.csv")
ht_path = os.path.join(OUT_DIR, "hole_targets.csv")

df_model.to_csv(df_path, index=False)

hole_summary = hole_targets.merge(
    collar[["Drillhole","X","Y","Z_g_earth","Longueur"]], on="Drillhole", how="left")
hole_summary.to_csv(ht_path, index=False)

print(f"\ndf_model   : {df_model.shape}  -> {df_path}")
print(f"hole_targets: {hole_summary.shape} -> {ht_path}")
print(f"\nNode feature columns  : {['X','Y','Z_g_earth','MID_DEPTH','Z_mid','interval_length','depth_ratio','is_SS','ss_encountered']}")
print(f"Target columns        : ['depth_at_HW', 'depth_at_fw']")
print(f"ss_encountered==1 rows: {(df_model['ss_encountered']==1).sum()}")
print(f"ss_encountered==0 rows: {(df_model['ss_encountered']==0).sum()}")
print(f"is_SS==1 rows         : {(df_model['is_SS']==1).sum()}")

print("\n=== PHASE 1 COMPLETE ===")
print("df_model.csv is ready for Phase 2 (graph construction + k-sweep).")
