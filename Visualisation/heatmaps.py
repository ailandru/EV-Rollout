# heatmaps.py
# Heat maps with custom scales per dataset.

import os
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as cx
from matplotlib.colors import ListedColormap

warnings.filterwarnings("ignore", category=UserWarning)

# ---- Config ----
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "..", "Output_Weighted")
OUT_DIR = os.path.join(PROJECT_ROOT, "..", "Visualisation_Output")
os.makedirs(OUT_DIR, exist_ok=True)

MAP_TITLE_FONT = dict(fontsize=14, weight="bold")
FIGSIZE = (10, 10)
POINT_SIZE = 10
POINT_ALPHA = 0.9
BASEMAP = cx.providers.OpenStreetMap.Mapnik
WEB_MERCATOR = 3857


# Utility: make bins and labels
def make_bins(max_val, step):
    edges = np.round(np.arange(0, max_val + step, step), 4).tolist()
    labels = []
    for i in range(len(edges) - 1):
        lo = edges[i]
        hi = edges[i + 1]
        if i == 0:
            label = f"{lo:.3f}–{hi:.3f}"
        else:
            label = f"{(lo + step/100):.3f}–{hi:.3f}"
        labels.append(label)
    return edges, labels


# Colour ramp
def blue_to_red_cmap(n):
    blues = np.array([0/255, 92/255, 230/255])
    reds  = np.array([220/255, 33/255, 39/255])
    cols = [blues + (reds - blues) * (i / (n - 1)) for i in range(n)]
    return ListedColormap(cols)


def plot_ranked_points(gpkg_path, value_col, title, out_png, max_val, step):
    # Bins + labels
    bin_edges, labels = make_bins(max_val, step)
    cmap = blue_to_red_cmap(len(labels))

    full_path = os.path.join(DATA_DIR, gpkg_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Could not find: {full_path}")

    gdf = gpd.read_file(full_path)
    if gdf.empty:
        raise ValueError(f"No data in {full_path}")
    if value_col not in gdf.columns:
        raise KeyError(f"Column '{value_col}' not found. Available: {list(gdf.columns)}")

    # Clean + clip
    vals = pd.to_numeric(gdf[value_col], errors="coerce").clip(lower=0, upper=max_val)
    gdf = gdf.assign(_val=vals).dropna(subset=["_val"]).copy()

    gdf["_bin"] = pd.cut(gdf["_val"], bins=bin_edges, labels=labels,
                         include_lowest=True, right=True)

    # CRS
    if gdf.crs is None:
        warnings.warn("Input has no CRS. Assuming EPSG:4326")
        gdf = gdf.set_crs(4326)
    gdf_3857 = gdf.to_crs(WEB_MERCATOR)

    # Plot
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for i, lab in enumerate(labels):
        sel = gdf_3857[gdf_3857["_bin"] == lab]
        if sel.empty:
            continue
        ax.scatter(
            sel.geometry.x, sel.geometry.y,
            s=POINT_SIZE, c=[cmap(i)], alpha=POINT_ALPHA,
            label=str(lab), linewidths=0
        )

    try:
        cx.add_basemap(ax, source=BASEMAP, crs=f"EPSG:{WEB_MERCATOR}")
    except Exception as e:
        warnings.warn(f"Basemap failed to load ({e}).")

    ax.set_title(f"{title} (0–{max_val})", **MAP_TITLE_FONT)
    ax.set_axis_off()
    ax.set_aspect("equal")
    if not gdf_3857.empty:
        x0, y0, x1, y1 = gdf_3857.total_bounds
        ax.set_xlim(x0 - 50, x1 + 50)
        ax.set_ylim(y0 - 50, y1 + 50)

    ax.legend(
        title=f"Value rank (0–{max_val})",
        loc="lower left", bbox_to_anchor=(0.01, 0.01),
        ncol=2, frameon=True, fontsize=8, title_fontsize=9, markerscale=1.2
    )

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, out_png)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---- Run directly ----
# 1) combined_weighted_ev_locations.gpkg (max=0.065, step=0.005)
plot_ranked_points(
    gpkg_path="combined_weighted_ev_locations.gpkg",
    value_col="combined_weight",
    title="Heat Map: Combined Weighted EV Locations",
    out_png="heatmap_combined_weighted.png",
    max_val=0.065,
    step=0.005
)

# 2) ev_combined_weighted_ev_locations.gpkg (max=0.05, step=0.005)
plot_ranked_points(
    gpkg_path="ev_combined_weighted_ev_locations.gpkg",
    value_col="ev_combined_weight",
    title="Heat Map: EV-Combined Weighted EV Locations",
    out_png="heatmap_ev_combined_weighted.png",
    max_val=0.05,
    step=0.005
)
