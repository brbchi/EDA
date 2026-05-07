"""
viz_3d_layer.py
3-D PyVista visualisation of the predicted SS hanging-wall and foot-wall
contact surfaces across the basin.

Steps:
  1. Interpolate surface elevation (Z) from 132 collar locations to the grid
  2. Compute absolute HW / FW elevations  (Z_surface - depth)
  3. Build two structured-grid surfaces in PyVista
  4. Render and save:
       outputs/viz_3d_layer.png   -- high-res screenshot (off-screen)
       outputs/viz_3d_layer.vtp   -- VTK file (open in ParaView for full
                                     interactivity)

Inputs:
  outputs/predicted_grid.csv   -- Predicted_X, Predicted_Y, depth_at_HW, depth_at_fw
  outputs/hole_targets.csv     -- collar X, Y, Z_g_earth, depth_at_HW, depth_at_fw
"""

import os, warnings
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import pyvista as pv

warnings.filterwarnings("ignore")

ROOT    = os.path.dirname(os.path.abspath(__file__))
OUT     = os.path.join(ROOT, "outputs")
GRID_RES = 250       # resolution of the structured surface grid

pv.global_theme.background = "white"
pv.global_theme.font.color  = "black"

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading data...")
grd = pd.read_csv(os.path.join(OUT, "predicted_grid.csv"))
trg = pd.read_csv(os.path.join(OUT, "hole_targets.csv"))
trg = trg[trg["ss_encountered"] == 1].copy()

print(f"  Grid pts : {len(grd):,}   |   Known holes : {len(trg)}")

# ── 2. Regular XY grid ────────────────────────────────────────────────────────
x_min, x_max = grd["Predicted_X"].min(), grd["Predicted_X"].max()
y_min, y_max = grd["Predicted_Y"].min(), grd["Predicted_Y"].max()
xi = np.linspace(x_min, x_max, GRID_RES)
yi = np.linspace(y_min, y_max, GRID_RES)
XI, YI = np.meshgrid(xi, yi)          # both shape (GRID_RES, GRID_RES)

scatter_pts = grd[["Predicted_X", "Predicted_Y"]].values

def regrid(values, method="linear"):
    return griddata(scatter_pts, values, (XI, YI),
                    method=method, fill_value=np.nan)

# ── 3. Interpolate surface Z from collar data onto the grid ───────────────────
print("Interpolating surface elevation from collars...")
surf_z = griddata(
    trg[["X", "Y"]].values,
    trg["Z_g_earth"].values.astype(float),
    (XI, YI),
    method="linear",
    fill_value=float(trg["Z_g_earth"].mean()),
)

# ── 4. Grid HW / FW depths, compute absolute elevations ──────────────────────
print("Gridding HW / FW depths...")
hw_depth = regrid(grd["depth_at_HW"].values)
fw_depth = regrid(grd["depth_at_fw"].values)

# Fill remaining NaN edges with nearest-neighbour
hw_depth_nn = regrid(grd["depth_at_HW"].values, method="nearest")
fw_depth_nn = regrid(grd["depth_at_fw"].values, method="nearest")
hw_depth = np.where(np.isnan(hw_depth), hw_depth_nn, hw_depth)
fw_depth = np.where(np.isnan(fw_depth), fw_depth_nn, fw_depth)

hw_elev   = surf_z - hw_depth          # absolute elevation of HW contact
fw_elev   = surf_z - fw_depth          # absolute elevation of FW contact
thickness = fw_depth - hw_depth        # SS thickness at each grid cell

# ── 5. Build PyVista structured surfaces ─────────────────────────────────────
def make_surface(Z_elev, scalar, scalar_name):
    """Create a pv.StructuredGrid surface at elevation Z_elev."""
    pts = np.column_stack([
        XI.ravel().astype(np.float64),
        YI.ravel().astype(np.float64),
        Z_elev.ravel().astype(np.float64),
    ])
    mesh = pv.StructuredGrid()
    mesh.points     = pts
    mesh.dimensions = [GRID_RES, GRID_RES, 1]
    mesh.point_data[scalar_name] = scalar.ravel().astype(np.float32)
    return mesh

print("Building PyVista surfaces...")
hw_surf  = make_surface(hw_elev,   hw_depth,  "HW_depth_m")
fw_surf  = make_surface(fw_elev,   fw_depth,  "FW_depth_m")
thk_surf = make_surface(hw_elev,   thickness, "SS_thickness_m")   # colour HW by thickness

# ── 6. Collar and HW/FW points ────────────────────────────────────────────────
collar_pts = np.column_stack([
    trg["X"].values, trg["Y"].values, trg["Z_g_earth"].values
]).astype(np.float64)

hw_pts = np.column_stack([
    trg["X"].values, trg["Y"].values,
    (trg["Z_g_earth"] - trg["depth_at_HW"]).values
]).astype(np.float64)

fw_pts = np.column_stack([
    trg["X"].values, trg["Y"].values,
    (trg["Z_g_earth"] - trg["depth_at_fw"]).values
]).astype(np.float64)

collar_cloud = pv.PolyData(collar_pts)
hw_cloud     = pv.PolyData(hw_pts)
fw_cloud     = pv.PolyData(fw_pts)

# Drillhole sticks (line from collar to FW depth for ss_encountered holes)
lines = []
for i in range(len(trg)):
    top = collar_pts[i]
    bot = fw_pts[i]
    lines.append(pv.Line(top, bot))
drill_lines = pv.MultiBlock(lines).combine()

# ── 7. Render ─────────────────────────────────────────────────────────────────
print("Rendering 3-D scene...")
pl = pv.Plotter(off_screen=True, window_size=[2000, 1200])

# FW surface (deeper, shown semi-transparent in blue)
pl.add_mesh(fw_surf, scalars="FW_depth_m", cmap="Blues",
            opacity=0.55, show_scalar_bar=False, label="FW Contact")

# HW surface (top of SS, coloured by thickness)
pl.add_mesh(thk_surf, scalars="SS_thickness_m", cmap="YlOrRd",
            opacity=0.92, scalar_bar_args={"title": "SS Thickness (m)",
                                           "color": "black", "vertical": True},
            label="HW Contact (colour = SS thickness)")

# Drillhole sticks
pl.add_mesh(drill_lines, color="#374151", line_width=2, label="Drillholes")

# Collar spheres
pl.add_points(collar_cloud, color="#1f2937", point_size=10,
              render_points_as_spheres=True, label="Collars")

# True HW / FW intercept points
pl.add_points(hw_cloud, color="#b45309", point_size=12,
              render_points_as_spheres=True, label="True HW intercept")
pl.add_points(fw_cloud, color="#1d4ed8", point_size=12,
              render_points_as_spheres=True, label="True FW intercept")

pl.add_legend(face=None, size=(0.25, 0.20), loc="lower right",
              background_opacity=0.6)
pl.show_axes()
pl.add_title("GNN Predicted SS Layer — HW & FW Contact Surfaces", font_size=14,
             color="black")

# Camera: oblique view from NW
pl.camera_position = "iso"
pl.camera.azimuth  = -45
pl.camera.elevation = 25
pl.reset_camera()

png_path = os.path.join(OUT, "viz_3d_layer.png")
pl.screenshot(png_path, transparent_background=False)
print(f"  Saved {png_path}")

# ── 8. Save VTK for interactive ParaView viewing ──────────────────────────────
vtp_hw = os.path.join(OUT, "viz_3d_hw_surface.vts")
vtp_fw = os.path.join(OUT, "viz_3d_fw_surface.vts")
hw_surf.save(vtp_hw)
fw_surf.save(vtp_fw)
print(f"  Saved {vtp_hw}")
print(f"  Saved {vtp_fw}")

print("\n=== 3-D visualisation complete ===")
print(f"  Screenshot : outputs/viz_3d_layer.png")
print(f"  VTK files  : outputs/viz_3d_hw_surface.vts  /  viz_3d_fw_surface.vts")
print("  Open .vts files in ParaView for fully interactive 3-D exploration.")
