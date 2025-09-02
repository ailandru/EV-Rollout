# heatmaps.py
# Heat maps with 12-bin scales (custom max per dataset) — legends always show all 12.

import os
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as cx
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

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


# Make 12 bins and readable labels
def make_bins(max_val, n_groups=12, decimals=3, label_offset=0.001):
    """
    Create 'n_groups' bins from 0 to max_val inclusive.
    Labels are 'lo–hi'; except the first bin, add a small offset to 'lo'
    so labels read like 0.041–0.060 instead of 0.040–0.060.
    """
    edges = np.linspace(0.0, float(max_val), n_groups + 1)  # precise edges
    labels = []
    for i in range(n_groups):
        lo, hi = edges[i], edges[i + 1]
        lo_lab = 0.0 if i == 0 else lo + label_offset
        labels.append(f"{lo_lab:.{decimals}f}–{hi:.{decimals}f}")
    return edges, labels


# --- UPDATED: Blue → Red with lighter low-end blues + subtle extra separation ---
def blue_to_red_cmap(n):
    """
    Blue → Red with a lighter (but still visible) starting blue and a gentle
    ease so the first few blue bins differ more clearly.
    """
    start_blue = np.array([135, 180, 255]) / 255.0  # ~ #87B4FF (light, not white)
    end_red    = np.array([220,  33,  39]) / 255.0  # ~ #DC2127

    t = np.linspace(0.0, 1.0, n)
    t = t ** 0.85  # lower -> stronger separation among early (blue) bins

    cols = start_blue + (end_red - start_blue) * t[:, None]
    return ListedColormap(cols)


def plot_ranked_points(gpkg_path, value_col, title, out_png, max_val, n_groups=12):
    # Bins + labels (always 12)
    bin_edges, labels = make_bins(max_val=max_val, n_groups=n_groups)
    cmap = blue_to_red_cmap(len(labels))

    full_path = os.path.join(DATA_DIR, gpkg_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Could not find: {full_path}")

    gdf = gpd.read_file(full_path)
    if gdf.empty:
        raise ValueError(f"No data in {full_path}")
    if value_col not in gdf.columns:
        raise KeyError(f"Column '{value_col}' not found. Available: {list(gdf.columns)}")

    # Clean + clip and bin
    vals = pd.to_numeric(gdf[value_col], errors="coerce").clip(lower=0, upper=max_val)
    gdf = gdf.assign(_val=vals).dropna(subset=["_val"]).copy()
    gdf["_bin"] = pd.cut(gdf["_val"], bins=bin_edges, labels=labels,
                         include_lowest=True, right=True)

    # CRS
    if gdf.crs is None:
        warnings.warn("Input has no CRS. Assuming EPSG:4326")
        gdf = gdf.set_crs(4326)
    gdf_3857 = gdf.to_crs(WEB_MERCATOR)

    # Plot points (group by bin)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for i, lab in enumerate(labels):
        sel = gdf_3857[gdf_3857["_bin"] == lab]
        if not sel.empty:
            ax.scatter(
                sel.geometry.x, sel.geometry.y,
                s=POINT_SIZE, c=[cmap(i)], alpha=POINT_ALPHA,
                linewidths=0
            )

    # Basemap
    try:
        cx.add_basemap(ax, source=BASEMAP, crs=f"EPSG:{WEB_MERCATOR}")
    except Exception as e:
        warnings.warn(f"Basemap failed to load ({e}).")

    ax.set_title(title, **MAP_TITLE_FONT)
    ax.set_axis_off()
    ax.set_aspect("equal")
    if not gdf_3857.empty:
        x0, y0, x1, y1 = gdf_3857.total_bounds
        ax.set_xlim(x0 - 50, x1 + 50)
        ax.set_ylim(y0 - 50, y1 + 50)

    # --- ALWAYS show 12 legend entries ---
    handles = [
        Line2D([0], [0], marker='o', linestyle='',
               markerfacecolor=cmap(i), markeredgecolor=cmap(i),
               markersize=6, label=lab)
        for i, lab in enumerate(labels)
    ]
    ax.legend(
        handles=handles,
        title=f"Value rank (12 groups, 0–{max_val})",
        loc="lower left", bbox_to_anchor=(0.01, 0.01),
        ncol=2, frameon=True, fontsize=8, title_fontsize=9, markerscale=1.2
    )

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, out_png)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---- Run directly ----
# A) Combined (max=0.32) — 12 groups
plot_ranked_points(
    gpkg_path="combined_weighted_ev_locations.gpkg",
    value_col="combined_weight",
    title="Spatial Distribution: Suitable EV Charging Locations based on Cars",
    out_png="heatmap_combined_weighted.png",
    max_val=0.32,
    n_groups=12
)

# B) EV-Combined (max=0.32) — 12 groups
plot_ranked_points(
    gpkg_path="ev_combined_weighted_ev_locations.gpkg",
    value_col="ev_combined_weight",
    title="Spatial Distribution: Suitable EV Charging Locations based on EVs",
    out_png="heatmap_ev_combined_weighted.png",
    max_val=0.32,
    n_groups=12
)

# C) S2 All Vehicles × Income (max=0.20) — 12 groups
plot_ranked_points(
    gpkg_path="s2_household_income_combined_all_vehicles_core.gpkg",
    value_col="s2_all_vehicles_income_combined",
    title="Spatial Distribution: Suitable EV Charging Locations - Income & Cars",
    out_png="C4_s2_heatmap_all.png",
    max_val=0.20,
    n_groups=12
)

# D) S2 EV Vehicles × Income (max=0.18) — 12 groups
plot_ranked_points(
    gpkg_path="s2_household_income_combined_ev_vehicles_core.gpkg",
    value_col="s2_ev_vehicles_income_combined",
    title="Spatial Distribution: Suitable EV Charging Locations - Income & EVs",
    out_png="C4_s2_heatmap_ev.png",
    max_val=0.18,
    n_groups=12
)
