"""
viz_3d_interactive.py
Interactive 3-D Plotly visualisation of the predicted SS layer.
Opens automatically in your default browser.

Output: outputs/viz_3d_interactive.html
"""

import os, warnings
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import plotly.graph_objects as go
import plotly.io as pio

warnings.filterwarnings("ignore")

ROOT     = os.path.dirname(os.path.abspath(__file__))
OUT      = os.path.join(ROOT, "outputs")
GRID_RES = 150   # surface resolution (150x150 = 22k cells, smooth but fast)

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading data...")
grd = pd.read_csv(os.path.join(OUT, "predicted_grid.csv"))
trg = pd.read_csv(os.path.join(OUT, "hole_targets.csv"))
trg = trg[trg["ss_encountered"] == 1].copy()

# ── 2. Regular XY grid ────────────────────────────────────────────────────────
xi = np.linspace(grd["Predicted_X"].min(), grd["Predicted_X"].max(), GRID_RES)
yi = np.linspace(grd["Predicted_Y"].min(), grd["Predicted_Y"].max(), GRID_RES)
XI, YI = np.meshgrid(xi, yi)
pts = grd[["Predicted_X", "Predicted_Y"]].values

def regrid(values):
    z = griddata(pts, values, (XI, YI), method="linear")
    z_nn = griddata(pts, values, (XI, YI), method="nearest")
    return np.where(np.isnan(z), z_nn, z)

# ── 3. Interpolate surface Z from collar data ─────────────────────────────────
print("Interpolating surface Z and gridding depths...")
surf_z = griddata(
    trg[["X", "Y"]].values,
    trg["Z_g_earth"].values.astype(float),
    (XI, YI), method="linear",
    fill_value=float(trg["Z_g_earth"].mean()),
)

hw_depth = regrid(grd["depth_at_HW"].values)
fw_depth = regrid(grd["depth_at_fw"].values)

hw_elev   = surf_z - hw_depth
fw_elev   = surf_z - fw_depth
thickness = fw_depth - hw_depth

# ── 4. Drillhole geometry ─────────────────────────────────────────────────────
collar_z  = trg["Z_g_earth"].values
hw_z_true = collar_z - trg["depth_at_HW"].values
fw_z_true = collar_z - trg["depth_at_fw"].values
tx, ty    = trg["X"].values, trg["Y"].values

# Build drillhole sticks as line segments (NaN separates each stick)
stick_x, stick_y, stick_z = [], [], []
for i in range(len(trg)):
    stick_x += [tx[i], tx[i], None]
    stick_y += [ty[i], ty[i], None]
    stick_z += [collar_z[i], fw_z_true[i], None]

# ── 5. Build Plotly figure ────────────────────────────────────────────────────
print("Building interactive 3-D figure...")
fig = go.Figure()

# FW surface — blue, semi-transparent
fig.add_trace(go.Surface(
    x=xi, y=yi, z=fw_elev,
    surfacecolor=fw_depth,
    colorscale="Blues",
    opacity=0.5,
    showscale=False,
    name="FW Contact",
    hovertemplate="X: %{x:.0f}<br>Y: %{y:.0f}<br>Elev: %{z:.0f} m<br>FW depth: %{surfacecolor:.0f} m<extra>FW Contact</extra>",
))

# HW surface — coloured by SS thickness
fig.add_trace(go.Surface(
    x=xi, y=yi, z=hw_elev,
    surfacecolor=thickness,
    colorscale="YlOrRd",
    opacity=0.90,
    colorbar=dict(title="SS Thickness (m)", x=1.02, thickness=18, len=0.6),
    name="HW Contact",
    hovertemplate="X: %{x:.0f}<br>Y: %{y:.0f}<br>Elev: %{z:.0f} m<br>Thickness: %{surfacecolor:.0f} m<extra>HW Contact</extra>",
))

# Drillhole sticks
fig.add_trace(go.Scatter3d(
    x=stick_x, y=stick_y, z=stick_z,
    mode="lines",
    line=dict(color="#374151", width=3),
    name="Drillholes",
    hoverinfo="skip",
))

# Collar points
fig.add_trace(go.Scatter3d(
    x=tx, y=ty, z=collar_z,
    mode="markers",
    marker=dict(size=5, color="#1f2937", symbol="circle"),
    name="Collars",
    hovertemplate="Hole: %{text}<br>X: %{x:.0f}<br>Y: %{y:.0f}<br>Z: %{z:.0f} m<extra>Collar</extra>",
    text=trg["Drillhole"].values,
))

# True HW intercepts
fig.add_trace(go.Scatter3d(
    x=tx, y=ty, z=hw_z_true,
    mode="markers",
    marker=dict(size=6, color="#b45309", symbol="diamond"),
    name="True HW",
    hovertemplate="Hole: %{text}<br>HW elev: %{z:.0f} m<extra>True HW</extra>",
    text=trg["Drillhole"].values,
))

# True FW intercepts
fig.add_trace(go.Scatter3d(
    x=tx, y=ty, z=fw_z_true,
    mode="markers",
    marker=dict(size=6, color="#1d4ed8", symbol="diamond"),
    name="True FW",
    hovertemplate="Hole: %{text}<br>FW elev: %{z:.0f} m<extra>True FW</extra>",
    text=trg["Drillhole"].values,
))

# ── 6. Layout ─────────────────────────────────────────────────────────────────
fig.update_layout(
    title=dict(
        text="GNN Predicted SS Layer — HW & FW Contact Surfaces",
        font=dict(size=16), x=0.5,
    ),
    scene=dict(
        xaxis=dict(title="X (m)", tickformat=".0f"),
        yaxis=dict(title="Y (m)", tickformat=".0f"),
        zaxis=dict(title="Elevation (m asl)"),
        aspectmode="manual",
        aspectratio=dict(x=2.0, y=1.5, z=0.4),
        camera=dict(
            eye=dict(x=-1.5, y=-1.8, z=1.0),
            up=dict(x=0, y=0, z=1),
        ),
    ),
    legend=dict(x=0.01, y=0.95, bgcolor="rgba(255,255,255,0.7)"),
    margin=dict(l=0, r=0, t=50, b=0),
    width=1400, height=800,
)

# ── 7. Save and open ──────────────────────────────────────────────────────────
html_path = os.path.join(OUT, "viz_3d_interactive.html")
fig.write_html(html_path, include_plotlyjs="cdn", full_html=True)
print(f"Saved: {html_path}")

# Open in default browser
import webbrowser
webbrowser.open(f"file:///{html_path.replace(os.sep, '/')}")
print("Opened in browser.")
