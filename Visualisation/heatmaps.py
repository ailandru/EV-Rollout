# heatmaps.py
# Heat maps with 8-bin scales (custom max per dataset) — legends always show all 8.

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


# Make 8 bins and readable labels
def make_bins(max_val, n_groups=8, decimals=3, label_offset=0.001):
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


# Red colormap - lighter shades for smaller values, darker for bigger values
def red_cmap(n):
    """
    Red colormap with lighter shades for smaller values and darker shades for larger values.
    """
    # Light red to dark red
    start_red = np.array([255, 200, 200]) / 255.0  # Light red
    end_red = np.array([139, 0, 0]) / 255.0        # Dark red
    
    t = np.linspace(0.0, 1.0, n)
    cols = start_red + (end_red - start_red) * t[:, None]
    return ListedColormap(cols)


def plot_ranked_points(gpkg_path, value_col, title, out_png, max_val, n_groups=8):
    """
    Plot ranked points from a GPKG file with comprehensive error handling.
    Returns True if successful, False if failed.
    """
    try:
        # Bins + labels (always 8)
        bin_edges, labels = make_bins(max_val=max_val, n_groups=n_groups)
        cmap = red_cmap(len(labels))

        full_path = os.path.join(DATA_DIR, gpkg_path)
        if not os.path.exists(full_path):
            print(f"❌ SKIPPED: File not found - {gpkg_path}")
            print(f"   Looking for: {full_path}")
            return False

        print(f"📊 Processing: {gpkg_path}")
        gdf = gpd.read_file(full_path)
        
        if gdf.empty:
            print(f"❌ SKIPPED: No data in {gpkg_path}")
            return False
            
        if value_col not in gdf.columns:
            print(f"❌ SKIPPED: Column '{value_col}' not found in {gpkg_path}")
            print(f"   Available columns: {list(gdf.columns)}")
            return False

        # Clean + clip and bin
        vals = pd.to_numeric(gdf[value_col], errors="coerce").clip(lower=0, upper=max_val)
        gdf = gdf.assign(_val=vals).dropna(subset=["_val"]).copy()
        
        if gdf.empty:
            print(f"❌ SKIPPED: No valid data after cleaning in {gpkg_path}")
            return False
            
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

        # --- ALWAYS show 8 legend entries ---
        handles = [
            Line2D([0], [0], marker='o', linestyle='',
                   markerfacecolor=cmap(i), markeredgecolor=cmap(i),
                   markersize=6, label=lab)
            for i, lab in enumerate(labels)
        ]
        
        # Enhanced legend with basemap feature descriptions
        legend_title = f"Value rank (8 groups, 0–{max_val})\n\nBasemap features:\n• Yellow triangles: Peaks/High Points\n• Lines: Roads and Highways"
        
        ax.legend(
            handles=handles,
            title=legend_title,
            loc="lower left", bbox_to_anchor=(0.01, 0.01),
            ncol=2, frameon=True, fontsize=8, title_fontsize=9, markerscale=1.2
        )

        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, out_png)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"✅ Saved: {out_path}")
        return True
        
    except Exception as e:
        print(f"❌ ERROR processing {gpkg_path}: {str(e)}")
        return False


def validate_files(datasets):
    """
    Validate all required files exist before processing.
    Returns lists of existing and missing files.
    """
    existing_files = []
    missing_files = []
    
    for dataset in datasets:
        gpkg_path = dataset["gpkg_path"]
        full_path = os.path.join(DATA_DIR, gpkg_path)
        if os.path.exists(full_path):
            existing_files.append(dataset)
        else:
            missing_files.append(gpkg_path)
    
    return existing_files, missing_files


# ---- Dataset Configuration ----
DATASETS = [
    {
        "gpkg_path": "s3_1_primary_combined_all_vehicles.gpkg",
        "value_col": "combined_weight",
        "title": "Spatial Distribution: Suitable EV Charging - Primary Substation & Cars",
        "out_png": "C4_s3_1_all_heatmap.png",
        "max_val": 0.21
    },
    {
        "gpkg_path": "s3_1_primary_combined_ev_vehicles.gpkg",
        "value_col": "ev_combined_weight",
        "title": "Spatial Distribution: Suitable EV Charging - Primary Substation & EVs",
        "out_png": "C4_s3_1_ev_heatmap.png",
        "max_val": 0.20
    },
    {
        "gpkg_path": "s3_2_secondary_combined_all_vehicles.gpkg",  # Fixed: changed from "primary" to "secondary"
        "value_col": "combined_weight",
        "title": "Spatial Distribution: Suitable EV Charging - Secondary Substation & Cars",  # Updated title
        "out_png": "C4_s3_2_all_heatmap.png",
        "max_val": 0.12
    },
    {
        "gpkg_path": "s3_2_secondary_combined_ev_vehicles.gpkg",  # Fixed: changed from "primary" to "secondary"
        "value_col": "ev_combined_weight",
        "title": "Spatial Distribution: Suitable EV Charging - Secondary Substation & EVs",  # Updated title
        "out_png": "C4_s3_2_ev_heatmap.png",
        "max_val": 0.07
    },
    {
        "gpkg_path": "s3_3_primary&secondary_combined_all_vehicles.gpkg",
        "value_col": "combined_weight",
        "title": "Spatial Distribution: Suitable EV Charging - Combined Substation & Cars",
        "out_png": "C4_s3_3_all_heatmap.png",
        "max_val": 0.21
    },
    {
        "gpkg_path": "s3_3_primary&secondary_combined_ev_vehicles.gpkg",
        "value_col": "ev_combined_weight",
        "title": "Spatial Distribution: Suitable EV Charging - Combined Substation & EVs",
        "out_png": "C4_s3_3_ev_heatmap.png",
        "max_val": 0.20
    }
]


# ---- Main Execution ----
if __name__ == "__main__":
    print(f"🔍 Checking data directory: {DATA_DIR}")
    print(f"📂 Output directory: {OUT_DIR}")
    print("=" * 60)
    
    # Validate files first
    existing_files, missing_files = validate_files(DATASETS)
    
    if missing_files:
        print("⚠️  MISSING FILES:")
        for missing_file in missing_files:
            print(f"   • {missing_file}")
        print()
    
    if existing_files:
        print(f"📋 Processing {len(existing_files)} available datasets...")
        print()
        
        successful = 0
        failed = 0
        
        for dataset in existing_files:
            success = plot_ranked_points(
                gpkg_path=dataset["gpkg_path"],
                value_col=dataset["value_col"],
                title=dataset["title"],
                out_png=dataset["out_png"],
                max_val=dataset["max_val"],
                n_groups=8
            )
            
            if success:
                successful += 1
            else:
                failed += 1
        
        print("=" * 60)
        print(f"📊 SUMMARY: {successful} successful, {failed} failed, {len(missing_files)} missing")
        
    else:
        print("❌ No valid datasets found to process!")
        
    print("🏁 Process completed.")