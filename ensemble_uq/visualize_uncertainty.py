"""
visualize_uncertainty.py  --  EDA/ensemble_uq

Interactive 3-D visualization of the HW contact uncertainty envelope.

Surfaces:
  Central   : HW contact coloured by Z_HW_std  (yellow = certain, red = uncertain)
  Envelope  : Z_HW ± 2·σ  (translucent steel-blue — the 95 % confidence band)
  Drillholes: coloured by is_SS  (green = SS, grey = not-SS)

Each point on the central surface answers: "at this XY location, how much do the
100 ensemble models disagree about WHERE the HW contact sits?"
  - Low σ (yellow) = models all place the contact at nearly the same elevation
  - High σ (red)   = models disagree — usually far from any drillhole

Output: outputs/uncertainty_envelope.html  (open in any browser)
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go

HERE     = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(HERE, "outputs")
SS3D_OUT = os.path.join(HERE, "..", "ss3d", "outputs")

GRID_X, GRID_Y = 200, 160

# ── 1. Load contacts ──────────────────────────────────────────────────────────
contacts = pd.read_csv(os.path.join(OUT_DIR, "certainty_grid_contacts.csv"))
print(f"Contacts loaded : {len(contacts):,} grid points")

# Recover the original linspace axes
xi = contacts["X"].values[:GRID_X]       # (200,)
yi = contacts["Y"].values[::GRID_X]      # (160,)

Z_HW     = contacts["Z_HW"].values.reshape(GRID_Y, GRID_X)
Z_HW_std = contacts["Z_HW_std"].values.reshape(GRID_Y, GRID_X)
Z_FW     = contacts["Z_FW"].values.reshape(GRID_Y, GRID_X)
Z_FW_std = contacts["Z_FW_std"].values.reshape(GRID_Y, GRID_X)

valid = ~np.isnan(Z_HW)
print(f"Grid points with SS detected : {valid.sum():,} / {GRID_X*GRID_Y:,}")
print(f"Z_HW_std  mean={np.nanmean(Z_HW_std):.1f} m  "
      f"max={np.nanmax(Z_HW_std):.1f} m  min={np.nanmin(Z_HW_std):.1f} m")

# ── 2. Load drillhole intervals ───────────────────────────────────────────────
df = pd.read_csv(os.path.join(SS3D_OUT, "df_model.csv"))
print(f"Drillhole intervals : {len(df):,}")

# ── 3. Build figure ───────────────────────────────────────────────────────────
fig = go.Figure()

std_min = float(np.nanmin(Z_HW_std))
std_max = float(np.nanmax(Z_HW_std))

# ── Central HW surface — coloured by elevation σ ─────────────────────────────
fig.add_trace(go.Surface(
    x=xi, y=yi, z=Z_HW,
    surfacecolor=Z_HW_std,
    colorscale="YlOrRd",          # yellow=low σ (certain), red=high σ (uncertain)
    cmin=std_min, cmax=std_max,
    colorbar=dict(
        title=dict(text="Z_HW σ (m)", side="right"),
        x=1.02, len=0.55, thickness=15,
    ),
    opacity=1.0,
    name="HW contact",
    showlegend=True,
    hovertemplate=(
        "X: %{x:.0f} m<br>"
        "Y: %{y:.0f} m<br>"
        "Z_HW: %{z:.1f} m<extra>HW contact</extra>"
    ),
))

# ── Upper envelope: Z_HW + 2·σ ───────────────────────────────────────────────
fig.add_trace(go.Surface(
    x=xi, y=yi,
    z=Z_HW + 2 * Z_HW_std,
    colorscale=[[0, "steelblue"], [1, "steelblue"]],
    showscale=False,
    opacity=0.20,
    name="HW + 2σ  (upper bound)",
    showlegend=True,
    hovertemplate="Z_HW + 2σ: %{z:.1f} m<extra>+2σ</extra>",
))

# ── Lower envelope: Z_HW − 2·σ ───────────────────────────────────────────────
fig.add_trace(go.Surface(
    x=xi, y=yi,
    z=Z_HW - 2 * Z_HW_std,
    colorscale=[[0, "steelblue"], [1, "steelblue"]],
    showscale=False,
    opacity=0.20,
    name="HW − 2σ  (lower bound)",
    showlegend=True,
    hovertemplate="Z_HW − 2σ: %{z:.1f} m<extra>−2σ</extra>",
))

# ── Drillhole intervals ───────────────────────────────────────────────────────
for is_ss, color, label, size in [
    (1, "#2ca02c", "SS intervals",     2.0),
    (0, "#aaaaaa", "Non-SS intervals", 1.2),
]:
    sub = df[df["is_SS"] == is_ss]
    fig.add_trace(go.Scatter3d(
        x=sub["X_orig"], y=sub["Y_orig"], z=sub["Z_orig"],
        mode="markers",
        marker=dict(size=size, color=color, opacity=0.55),
        name=label,
    ))

# ── Layout ────────────────────────────────────────────────────────────────────
fig.update_layout(
    title=dict(
        text=(
            "SS Layer — HW Contact Uncertainty Envelope  "
            "(Deep Ensemble, M = 100)<br>"
            "<sup>Surface colour = elevation σ across 100 realizations  |  "
            "Translucent band = ±2σ  (≈95 % confidence)</sup>"
        ),
        x=0.5, xanchor="center", font=dict(size=14),
    ),
    scene=dict(
        xaxis_title="X (m)",
        yaxis_title="Y (m)",
        zaxis_title="Elevation (m asl)",
        camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9)),
        aspectmode="manual",
        aspectratio=dict(x=2.0, y=1.6, z=0.7),
    ),
    legend=dict(
        x=0.01, y=0.98,
        bgcolor="rgba(255,255,255,0.75)",
        bordercolor="lightgrey", borderwidth=1,
    ),
    margin=dict(l=0, r=20, t=90, b=0),
    width=1300, height=750,
)

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, "uncertainty_envelope.html")
fig.write_html(out_path)
print(f"\nSaved: {out_path}")
print("Open in your browser — fully interactive (rotate, zoom, hover).")
