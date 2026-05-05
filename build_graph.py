import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

K      = 10
ROOT   = os.path.dirname(os.path.abspath(__file__))
collar = pd.read_csv(os.path.join(ROOT, "collar.csv"))

coords  = collar[["X", "Y"]].values
nbrs    = NearestNeighbors(n_neighbors=K + 1, algorithm="ball_tree").fit(coords)
_, indices = nbrs.kneighbors(coords)
indices = indices[:, 1:]   # drop self

fig, ax = plt.subplots(figsize=(10, 9))

for i in range(len(collar)):
    x0, y0 = collar.loc[i, "X"], collar.loc[i, "Y"]
    for j in indices[i]:
        x1, y1 = collar.loc[j, "X"], collar.loc[j, "Y"]
        ax.plot([x0, x1], [y0, y1], color="black", lw=0.6, alpha=0.5, zorder=1)

sc = ax.scatter(collar["X"], collar["Y"], c=collar["Z_g_earth"],
                cmap="viridis", s=40, zorder=2,
                edgecolors="black", linewidths=0.4)

cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Z Depth", fontsize=10)

ax.set_xlabel("XCOLLAR", fontsize=11)
ax.set_ylabel("YCOLLAR", fontsize=11)
ax.set_title(f"Two-dimensional graph visualisation of the sampled drillhole connections  (k={K})",
             fontsize=11)
ax.ticklabel_format(style="sci", axis="both", scilimits=(5, 5))

plt.tight_layout()
out = os.path.join(ROOT, f"graph_2d_collars_k{K}.png")
plt.savefig(out, dpi=160, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")
