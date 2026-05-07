"""
data_prep.py  --  EDA/zelev
Reads Composites.csv + collar.csv and extracts Z-elevation SS contacts.

Z_HW = elevation (m asl) at the TOP  of the first SS interval
     = Z_mid  +  interval_length / 2   (Z column = mid-interval elevation)
Z_FW = elevation (m asl) at the BASE of the last  SS interval
     = Z_mid  -  interval_length / 2

Outputs  (in outputs/)
  df_model.csv         one row per composite interval
  hole_targets.csv     one row per drillhole  (Z_HW, Z_FW, etc.)
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

# ── 3. Extract Z_HW / Z_FW from the Z column of Composites.csv ───────────────
ss = comp[comp["Lithology"] == "SS"].copy()

first_ss    = ss.loc[ss.groupby("Drillhole")["Depth_From"].idxmin()].set_index("Drillhole")
Z_HW_series = first_ss["Z"]   # Z column = mid-interval elevation; first SS row IS the HW contact

last_ss     = ss.loc[ss.groupby("Drillhole")["Depth_To"].idxmax()].set_index("Drillhole")
Z_FW_series = last_ss["Z"]    # last SS row IS the FW contact

ss_thick = Z_HW_series - Z_FW_series   # positive = HW above FW (expected)

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
print(f"\nTarget stats (SS holes, metres asl):")
print(f"  Z_HW : mean={enc1['Z_HW'].mean():.1f}  std={enc1['Z_HW'].std():.1f}  "
      f"range=[{enc1['Z_HW'].min():.1f}, {enc1['Z_HW'].max():.1f}]")
print(f"  Z_FW : mean={enc1['Z_FW'].mean():.1f}  std={enc1['Z_FW'].std():.1f}  "
      f"range=[{enc1['Z_FW'].min():.1f}, {enc1['Z_FW'].max():.1f}]")
print(f"  Thickness: mean={enc1['ss_thickness'].mean():.1f}m  "
      f"range=[{enc1['ss_thickness'].min():.1f}, {enc1['ss_thickness'].max():.1f}]")

# ── 4. Merge collar info into composites ──────────────────────────────────────
# Composites already has X, Y — only take Z_g_earth and Longueur from collar
comp = comp.merge(collar[["Drillhole", "Z_g_earth", "Longueur"]],
                  on="Drillhole", how="left")
comp = comp.merge(hole_targets[["Drillhole", "Z_HW", "Z_FW",
                                 "ss_thickness", "ss_encountered"]],
                  on="Drillhole", how="left")

# ── 5. Derived interval features ──────────────────────────────────────────────
comp["MID_DEPTH"]       = (comp["Depth_From"] + comp["Depth_To"]) / 2.0
comp["Z_mid"]           = comp["Z"].astype(float)
comp["interval_length"] = comp["Length"].astype(float)
comp["depth_ratio"]     = comp["MID_DEPTH"] / comp["Longueur"].clip(lower=1e-6)
comp["is_SS"]           = (comp["Lithology"] == "SS").astype(int)

comp["X_orig"]         = comp["X"]
comp["Y_orig"]         = comp["Y"]
comp["Z_g_earth_orig"] = comp["Z_g_earth"]

# ── 6. Standardise spatial + interval features ────────────────────────────────
SCALE_COLS = ["X", "Y", "Z_g_earth", "MID_DEPTH", "Z_mid",
              "interval_length", "depth_ratio"]
scaler = StandardScaler()
comp[SCALE_COLS] = scaler.fit_transform(comp[SCALE_COLS])
joblib.dump(scaler, os.path.join(OUT_DIR, "feature_scaler.joblib"))
print(f"\nFeature scaler fitted on {len(SCALE_COLS)} columns and saved.")

# ── 7. Assemble df_model ──────────────────────────────────────────────────────
KEEP = [
    "Sample_Num", "Drillhole", "Depth_From", "Depth_To",
    "X", "Y", "Z_g_earth", "MID_DEPTH", "Z_mid",
    "interval_length", "depth_ratio",
    "is_SS", "ss_encountered",
    "Z_HW", "Z_FW", "ss_thickness",
    "X_orig", "Y_orig", "Z_g_earth_orig",
]
df_model = comp[KEEP].reset_index(drop=True)

# ── 8. Integrity checks ───────────────────────────────────────────────────────
enc1_rows = df_model[df_model["ss_encountered"] == 1]
enc0_rows = df_model[df_model["ss_encountered"] == 0]
assert enc1_rows[["Z_HW", "Z_FW"]].notna().all().all(), "SS holes missing Z targets!"
assert enc0_rows["Z_HW"].isna().all(), "Non-SS holes should have NaN Z targets!"
assert df_model.duplicated(subset=["Drillhole", "Depth_From"]).sum() == 0, "Duplicate intervals!"
print("\n=== INTEGRITY CHECKS PASSED ===")

# ── 9. Save ───────────────────────────────────────────────────────────────────
df_model.to_csv(os.path.join(OUT_DIR, "df_model.csv"), index=False)

hole_summary = hole_targets.merge(
    collar[["Drillhole", "X", "Y", "Z_g_earth", "Longueur"]], on="Drillhole", how="left")
hole_summary.to_csv(os.path.join(OUT_DIR, "hole_targets.csv"), index=False)

print(f"\ndf_model     : {df_model.shape}  -> outputs/df_model.csv")
print(f"hole_targets : {hole_summary.shape} -> outputs/hole_targets.csv")
print("\n=== DATA PREP COMPLETE ===")
