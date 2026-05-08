"""
viz_3d_interactive.py  --  EDA/ss3d
Interactive 3-D Plotly visualization of the GNN-predicted SS layer contacts.

Output: outputs/viz_3d_interactive.html
"""

import os, warnings
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

ROOT     = os.path.dirname(os.path.abspath(__file__))
EDA      = os.path.join(ROOT, "..")
OUT      = os.path.join(ROOT, "outputs")
GRID_RES = 150

print("Loading data...")
grd = pd.read_csv(os.path.join(OUT, "predicted_grid.csv")).dropna()
trg = pd.read_csv(os.path.join(OUT, "hole_targets.csv"))
trg = trg[trg["ss_encountered"] == 1].copy()

trg["Z_HW_true"] = trg["Z_HW"].astype(float)
trg["Z_FW_true"] = trg["Z_FW"].astype(float)

xi = np.linspace(grd["Predicted_X"].min(), grd["Predicted_X"].max(), GRID_RES)
yi = np.linspace(grd["Predicted_Y"].min(), grd["Predicted_Y"].max(), GRID_RES)
XI, YI = np.meshgrid(xi, yi)
pts    = grd[["Predicted_X", "Predicted_Y"]].values


def regrid(values):
    z    = griddata(pts, values, (XI, YI), method="linear")
    z_nn = griddata(pts, values, (XI, YI), method="nearest")
    return np.where(np.isnan(z), z_nn, z)


print("Gridding predicted Z-elevation surfaces...")
hw_elev   = regrid(grd["Z_HW_elev"].values)
fw_elev   = regrid(grd["Z_FW_elev"].values)
thickness = hw_elev - fw_elev

collar_z  = trg["Z_g_earth"].values.astype(float)
hw_z_true = trg["Z_HW_true"].values
fw_z_true = trg["Z_FW_true"].values
tx, ty    = trg["X"].values, trg["Y"].values

stick_x, stick_y, stick_z = [], [], []
for i in range(len(trg)):
    stick_x += [tx[i], tx[i], None]
    stick_y += [ty[i], ty[i], None]
    stick_z += [collar_z[i], fw_z_true[i], None]

print("Building interactive 3-D figure...")
fig = go.Figure()

fig.add_trace(go.Surface(
    x=xi, y=yi, z=fw_elev,
    surfacecolor=fw_elev,
    colorscale="Blues",
    opacity=0.5,
    showscale=False,
    name="FW Contact",
    hovertemplate="X: %{x:.0f}<br>Y: %{y:.0f}<br>FW elev: %{z:.0f} m asl<extra>FW Contact</extra>",
))

fig.add_trace(go.Surface(
    x=xi, y=yi, z=hw_elev,
    surfacecolor=thickness,
    colorscale="YlOrRd",
    opacity=0.90,
    colorbar=dict(title="SS Thickness (m)", x=1.02, thickness=18, len=0.6),
    name="HW Contact",
    hovertemplate=(
        "X: %{x:.0f}<br>Y: %{y:.0f}<br>"
        "HW elev: %{z:.0f} m asl<br>"
        "Thickness: %{surfacecolor:.0f} m<extra>HW Contact</extra>"
    ),
))

fig.add_trace(go.Scatter3d(
    x=stick_x, y=stick_y, z=stick_z,
    mode="lines",
    line=dict(color="#374151", width=3),
    name="Drillholes",
    hoverinfo="skip",
))

fig.add_trace(go.Scatter3d(
    x=tx, y=ty, z=collar_z,
    mode="markers",
    marker=dict(size=5, color="#1f2937", symbol="circle"),
    name="Collars",
    hovertemplate=(
        "Hole: %{text}<br>X: %{x:.0f}<br>Y: %{y:.0f}<br>"
        "Collar Z: %{z:.0f} m asl<extra>Collar</extra>"
    ),
    text=trg["Drillhole"].values,
))

fig.add_trace(go.Scatter3d(
    x=tx, y=ty, z=hw_z_true,
    mode="markers",
    marker=dict(size=7, color="#b45309", symbol="diamond"),
    name="True HW",
    hovertemplate="Hole: %{text}<br>HW elev: %{z:.0f} m asl<extra>True HW</extra>",
    text=trg["Drillhole"].values,
))

fig.add_trace(go.Scatter3d(
    x=tx, y=ty, z=fw_z_true,
    mode="markers",
    marker=dict(size=7, color="#1d4ed8", symbol="diamond"),
    name="True FW",
    hovertemplate="Hole: %{text}<br>FW elev: %{z:.0f} m asl<extra>True FW</extra>",
    text=trg["Drillhole"].values,
))

fig.update_layout(
    title=dict(
        text="3-D SS Indicator GNN — Predicted HW / FW Contacts (m asl)",
        font=dict(size=16), x=0.5,
    ),
    scene=dict(
        xaxis=dict(title="X (m)", tickformat=".0f"),
        yaxis=dict(title="Y (m)", tickformat=".0f"),
        zaxis=dict(title="Elevation (m asl)"),
        aspectmode="manual",
        aspectratio=dict(x=2.0, y=1.5, z=0.6),
        camera=dict(eye=dict(x=-1.5, y=-1.8, z=1.0), up=dict(x=0, y=0, z=1)),
    ),
    legend=dict(x=0.01, y=0.95, bgcolor="rgba(255,255,255,0.7)"),
    margin=dict(l=0, r=0, t=50, b=0),
    width=1400, height=800,
)

html_path = os.path.join(OUT, "viz_3d_interactive.html")
fig.write_html(html_path, include_plotlyjs="cdn", full_html=True)
print(f"Saved: {html_path}")

import webbrowser
webbrowser.open(f"file:///{html_path.replace(os.sep, '/')}")
print("Opened in browser.")
