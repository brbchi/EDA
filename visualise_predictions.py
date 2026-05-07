"""
visualise_predictions.py
Visualise GNN HW / FW predictions across the simulated basin grid.

Inputs:
  outputs/predicted_grid.csv   -- Predicted_X, Predicted_Y,
                                   depth_at_HW, depth_at_fw
  outputs/hole_targets.csv     -- ground-truth collar targets (ss_encountered==1)

Outputs (all in outputs/):
  viz_planview.png             -- plan-view depth maps + thickness
  viz_crosssections.png        -- E-W and N-S vertical slices (depth scale)
  viz_diagnostics.png          -- residuals, histograms, predicted vs actual
"""

import os, warnings
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import BoundaryNorm

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__  ))
OUT  = os.path.join(ROOT, "outputs")

GRID_RES = 300          # cells per axis for regular interpolation grid
CS_TOL   = 2000         # metres half-width for cross-section slices

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading data...")
grd = pd.read_csv(os.path.join(OUT, "predicted_grid.csv"))
trg = pd.read_csv(os.path.join(OUT, "hole_targets.csv"))
trg = trg[trg["ss_encountered"] == 1].copy()

# Derived quantities
grd["ss_thickness"] = grd["depth_at_fw"] - grd["depth_at_HW"]

print(f"  Grid points : {len(grd):,}")
print(f"  Known holes : {len(trg)}")

# ── 2. Build regular interpolation grid (for contourf) ────────────────────────
xi = np.linspace(grd["Predicted_X"].min(), grd["Predicted_X"].max(), GRID_RES)
yi = np.linspace(grd["Predicted_Y"].min(), grd["Predicted_Y"].max(), GRID_RES)
XI, YI = np.meshgrid(xi, yi)
pts = grd[["Predicted_X", "Predicted_Y"]].values

def regrid(col):
    return griddata(pts, grd[col].values, (XI, YI), method="linear")

print("Gridding HW depth...")
ZI_hw  = regrid("depth_at_HW")
print("Gridding FW depth...")
ZI_fw  = regrid("depth_at_fw")
print("Gridding thickness...")
ZI_thk = regrid("ss_thickness")


def sci_fmt(ax):
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v/1e3:.0f}k"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v/1e3:.0f}k"))
    ax.set_xlabel("X (km)"); ax.set_ylabel("Y (km)")


# ── 3. Figure 1 — Plan-view depth maps ───────────────────────────────────────
print("Plotting plan-view maps...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("GNN Predictions — Plan View", fontsize=13, fontweight="bold")

specs = [
    (ZI_hw,  "depth_at_HW (m)",    "Reds",     "HW Depth (m from collar)"),
    (ZI_fw,  "depth_at_fw (m)",    "Blues",    "FW Depth (m from collar)"),
    (ZI_thk, "SS Thickness (m)",   "Greens",   "SS Thickness (m)"),
]
for ax, (data, title, cmap, cbar_lbl) in zip(axes, specs):
    cf = ax.contourf(XI, YI, data, levels=20, cmap=cmap)
    cb = fig.colorbar(cf, ax=ax, shrink=0.85)
    cb.set_label(cbar_lbl, fontsize=9)
    # Overlay drillhole collars
    ax.scatter(trg["X"], trg["Y"], c="black", s=25, marker="^",
               zorder=5, label="Known holes", edgecolors="white", linewidths=0.4)
    sci_fmt(ax)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7)

plt.tight_layout()
fig.savefig(os.path.join(OUT, "viz_planview.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved viz_planview.png")


# ── 4. Figure 2 — Cross-sections ─────────────────────────────────────────────
print("Plotting cross-sections...")

mid_x = grd["Predicted_X"].median()
mid_y = grd["Predicted_Y"].median()

ew_slice = grd[np.abs(grd["Predicted_Y"] - mid_y) <= CS_TOL].copy()
ns_slice = grd[np.abs(grd["Predicted_X"] - mid_x) <= CS_TOL].copy()
ew_slice = ew_slice.sort_values("Predicted_X")
ns_slice = ns_slice.sort_values("Predicted_Y")

fig, axes = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle("Cross-Sections: Predicted SS Depth from Collar", fontsize=12, fontweight="bold")

# E-W cross-section (X axis) — depth increases downward so invert y-axis
ax = axes[0]
ax.fill_between(ew_slice["Predicted_X"], ew_slice["depth_at_HW"], ew_slice["depth_at_fw"],
                alpha=0.35, color="#f59e0b", label="SS layer")
ax.plot(ew_slice["Predicted_X"], ew_slice["depth_at_HW"], color="#b45309", lw=1.5, label="HW depth")
ax.plot(ew_slice["Predicted_X"], ew_slice["depth_at_fw"], color="#1d4ed8", lw=1.5, label="FW depth")
ew_holes = trg[np.abs(trg["Y"] - mid_y) <= CS_TOL * 2]
ax.scatter(ew_holes["X"], ew_holes["depth_at_HW"], marker="v", s=60, c="#b45309",
           zorder=6, label="True HW (holes)")
ax.scatter(ew_holes["X"], ew_holes["depth_at_fw"], marker="^", s=60, c="#1d4ed8",
           zorder=6, label="True FW (holes)")
ax.invert_yaxis()
ax.set_xlabel("X (m)"); ax.set_ylabel("Depth from collar (m)")
ax.set_title(f"E-W Cross-Section  (Y ≈ {mid_y:.0f} m  ±  {CS_TOL} m)", fontsize=10)
ax.legend(fontsize=8, ncol=3); ax.grid(alpha=0.3)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1e3:.1f}k"))

# N-S cross-section (Y axis)
ax = axes[1]
ax.fill_between(ns_slice["Predicted_Y"], ns_slice["depth_at_HW"], ns_slice["depth_at_fw"],
                alpha=0.35, color="#f59e0b", label="SS layer")
ax.plot(ns_slice["Predicted_Y"], ns_slice["depth_at_HW"], color="#b45309", lw=1.5, label="HW depth")
ax.plot(ns_slice["Predicted_Y"], ns_slice["depth_at_fw"], color="#1d4ed8", lw=1.5, label="FW depth")
ns_holes = trg[np.abs(trg["X"] - mid_x) <= CS_TOL * 2]
ax.scatter(ns_holes["Y"], ns_holes["depth_at_HW"], marker="v", s=60, c="#b45309",
           zorder=6, label="True HW (holes)")
ax.scatter(ns_holes["Y"], ns_holes["depth_at_fw"], marker="^", s=60, c="#1d4ed8",
           zorder=6, label="True FW (holes)")
ax.invert_yaxis()
ax.set_xlabel("Y (m)"); ax.set_ylabel("Depth from collar (m)")
ax.set_title(f"N-S Cross-Section  (X ≈ {mid_x:.0f} m  ±  {CS_TOL} m)", fontsize=10)
ax.legend(fontsize=8, ncol=3); ax.grid(alpha=0.3)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1e3:.1f}k"))

plt.tight_layout()
fig.savefig(os.path.join(OUT, "viz_crosssections.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved viz_crosssections.png")


# ── 5. Figure 3 — Diagnostics ────────────────────────────────────────────────
print("Plotting diagnostics...")

# Re-run model predictions at collar locations to get residuals
# We use the known holes and compare against grd nearest-neighbour prediction
from sklearn.neighbors import NearestNeighbors as NN
nn1 = NN(n_neighbors=1).fit(grd[["Predicted_X", "Predicted_Y"]].values)
_, idx = nn1.kneighbors(trg[["X", "Y"]].values)
idx = idx.ravel()

p_hw = grd["depth_at_HW"].values[idx]
p_fw = grd["depth_at_fw"].values[idx]
t_hw = trg["depth_at_HW"].values
t_fw = trg["depth_at_fw"].values
res_hw = p_hw - t_hw
res_fw = p_fw - t_fw

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Prediction Diagnostics", fontsize=13, fontweight="bold")

# Row 1: HW
ax = axes[0, 0]
ax.scatter(t_hw, p_hw, alpha=0.7, s=25, c="#2563eb", edgecolors="none")
lims = [min(t_hw.min(), p_hw.min()), max(t_hw.max(), p_hw.max())]
ax.plot(lims, lims, "k--", lw=1)
ax.set_xlabel("True HW depth (m)"); ax.set_ylabel("Predicted HW depth (m)")
ax.set_title("HW — Predicted vs True"); ax.grid(alpha=0.3)

ax = axes[0, 1]
ax.hist(grd["depth_at_HW"].dropna(), bins=50, color="#2563eb", alpha=0.8, edgecolor="none")
ax.axvline(t_hw.mean(), color="#dc2626", ls="--", lw=1.5, label=f"Known mean={t_hw.mean():.0f}m")
ax.set_xlabel("depth_at_HW (m)"); ax.set_ylabel("Count"); ax.set_title("HW Depth Distribution (grid)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[0, 2]
sc = ax.scatter(trg["X"], trg["Y"], c=res_hw, cmap="RdBu_r",
                vmin=-np.percentile(np.abs(res_hw), 95),
                vmax= np.percentile(np.abs(res_hw), 95),
                s=60, edgecolors="k", linewidths=0.4)
plt.colorbar(sc, ax=ax).set_label("Residual (m)")
ax.set_title(f"HW Residual Map  (RMSE={np.sqrt((res_hw**2).mean()):.1f}m)")
sci_fmt(ax); ax.grid(alpha=0.2)

# Row 2: FW
ax = axes[1, 0]
ax.scatter(t_fw, p_fw, alpha=0.7, s=25, c="#16a34a", edgecolors="none")
lims = [min(t_fw.min(), p_fw.min()), max(t_fw.max(), p_fw.max())]
ax.plot(lims, lims, "k--", lw=1)
ax.set_xlabel("True FW depth (m)"); ax.set_ylabel("Predicted FW depth (m)")
ax.set_title("FW — Predicted vs True"); ax.grid(alpha=0.3)

ax = axes[1, 1]
ax.hist(grd["depth_at_fw"].dropna(), bins=50, color="#16a34a", alpha=0.8, edgecolor="none")
ax.axvline(t_fw.mean(), color="#dc2626", ls="--", lw=1.5, label=f"Known mean={t_fw.mean():.0f}m")
ax.set_xlabel("depth_at_fw (m)"); ax.set_ylabel("Count"); ax.set_title("FW Depth Distribution (grid)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1, 2]
sc = ax.scatter(trg["X"], trg["Y"], c=res_fw, cmap="RdBu_r",
                vmin=-np.percentile(np.abs(res_fw), 95),
                vmax= np.percentile(np.abs(res_fw), 95),
                s=60, edgecolors="k", linewidths=0.4)
plt.colorbar(sc, ax=ax).set_label("Residual (m)")
ax.set_title(f"FW Residual Map  (RMSE={np.sqrt((res_fw**2).mean()):.1f}m)")
sci_fmt(ax); ax.grid(alpha=0.2)

plt.tight_layout()
fig.savefig(os.path.join(OUT, "viz_diagnostics.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved viz_diagnostics.png")

print("\n=== Visualisation complete ===")
print("Outputs:")
for f in ["viz_planview.png", "viz_crosssections.png", "viz_diagnostics.png"]:
    print(f"  outputs/{f}")
