# heatmaps.py
# Heat maps with 10-bin equal count distribution using red color scheme and grey canvas basemap

import os
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as cx
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

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
# Changed to ESRI canvas basemap
BASEMAP = cx.providers.Esri.WorldTopoMap
WEB_MERCATOR = 3857

# Define which datasets should exclude zero values
S3_DATASETS = [
    "s3_1_primary_combined_all_vehicles.gpkg",
    "s3_1_primary_combined_ev_vehicles.gpkg"
]


def add_north_arrow(ax):
    """
    Add a simple north arrow to the upper right corner of the map.
    """
    # Get the current axis limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Position in upper right corner (as percentages of the axis range)
    x_pos = xlim[1] - 0.08 * (xlim[1] - xlim[0])  # 8% from right
    y_pos = ylim[1] - 0.08 * (ylim[1] - ylim[0])  # 8% from top

    # Arrow length (5% of the y-axis range -> bigger than before)
    arrow_length = 0.05 * (ylim[1] - ylim[0])

    # Create arrow pointing north
    arrow = FancyArrowPatch(
        (x_pos, y_pos - arrow_length),
        (x_pos, y_pos),
        connectionstyle="arc3",
        arrowstyle='-|>',  # more prominent arrowhead
        mutation_scale=40,  # bigger arrowhead
        color='black',
        linewidth=3
    )
    ax.add_patch(arrow)

    # Add "N" label above the arrow (bigger font size)
    ax.text(x_pos, y_pos + 0.01 * (ylim[1] - ylim[0]), 'N',
            ha='center', va='bottom', fontsize=18, fontweight='bold', color='black')


def create_equal_count_bins(values, n_groups=10, decimals=4):
    """
    Create equal count bins where each bin contains exactly the same number of points.
    This uses percentiles to ensure exactly 10% of data in each bin.
    
    Arguments:
        values (array): The data values to bin
        n_groups (int): Number of bins to create (default 10)
        decimals (int): Number of decimal places for labels
    
    Returns:
        tuple: (bin_labels, percentile_values, bin_assignments)
    """
    # Sort values to understand the distribution
    sorted_values = np.sort(values)
    n_points = len(sorted_values)
    points_per_bin = n_points // n_groups
    
    print(f"   Creating {n_groups} equal-count bins with ~{points_per_bin} points each")
    
    # Calculate percentiles for exactly equal-sized bins
    percentiles = np.linspace(0, 100, n_groups + 1)
    bin_edges = np.percentile(values, percentiles)
    
    # Create labels showing the actual value ranges for each bin
    bin_labels = []
    for i in range(n_groups):
        low_val = bin_edges[i]
        high_val = bin_edges[i + 1]
        bin_labels.append(f"{low_val:.{decimals}f}–{high_val:.{decimals}f}")
    
    # Assign each point to a bin using percentile ranks
    # This ensures exactly equal counts per bin
    percentile_ranks = np.searchsorted(bin_edges[1:-1], values, side='left')
    
    return bin_labels, bin_edges, percentile_ranks

def red_cmap(n):
    """
    Red colormap with lightest red for lowest values (bottom 10%) 
    and darkest red for highest values (top 10%).
    """
    colors = []
    for i in range(n):
        # Intensity increases from 0 to 1 as we go from lowest to highest values
        intensity = i / (n - 1) if n > 1 else 0
        
        # Light red (255, 200, 200) to dark red (139, 0, 0)
        r = 1.0 - 0.455 * intensity  # 1.0 to 0.545 (255 to 139 normalized)
        g = 0.784 - 0.784 * intensity  # 0.784 to 0 (200 to 0 normalized)
        b = 0.784 - 0.784 * intensity  # 0.784 to 0 (200 to 0 normalized)
        
        colors.append([r, g, b])
    
    return ListedColormap(colors)

def plot_ranked_points(gpkg_path, value_col, title, out_png, n_groups=10):
    """
    Plot ranked points using equal count distribution with red color scheme.
    Each bin contains exactly the same number of points (10% each).
    For S3 datasets, excludes points with values of exactly 0.0000.
    """
    try:
        full_path = os.path.join(DATA_DIR, gpkg_path)
        if not os.path.exists(full_path):
            print(f"❌ SKIPPED: File not found - {gpkg_path}")
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

        # Clean values - only remove NaN
        vals = pd.to_numeric(gdf[value_col], errors="coerce")
        gdf = gdf.assign(_val=vals).dropna(subset=["_val"]).copy()
        
        if gdf.empty:
            print(f"❌ SKIPPED: No valid data after cleaning in {gpkg_path}")
            return False
        
        # For S3 datasets, exclude values that are exactly 0.0000
        if gpkg_path in S3_DATASETS:
            initial_count = len(gdf)
            gdf = gdf[gdf["_val"] > 0.0000].copy()
            excluded_count = initial_count - len(gdf)
            print(f"   Excluded {excluded_count} points with values = 0.0000 for S3 dataset")
            
            if gdf.empty:
                print(f"❌ SKIPPED: No data remaining after excluding zero values in {gpkg_path}")
                return False
        
        # Print data statistics
        data_min, data_max = gdf["_val"].min(), gdf["_val"].max()
        print(f"   Data range: {data_min:.6f} to {data_max:.6f}")
        print(f"   Total points for mapping: {len(gdf)}")
        
        # Create equal count bins
        bin_labels, bin_edges, bin_assignments = create_equal_count_bins(
            gdf["_val"].values, n_groups=n_groups
        )
        cmap = red_cmap(len(bin_labels))
        
        # Assign bin labels to each point
        gdf["_bin_index"] = bin_assignments
        gdf["_bin_label"] = [bin_labels[i] if i < len(bin_labels) else bin_labels[-1] 
                            for i in bin_assignments]
        
        # Verify equal counts and print bin information
        print(f"   Bin distribution:")
        for i, label in enumerate(bin_labels):
            count = (gdf["_bin_index"] == i).sum()
            actual_min = gdf[gdf["_bin_index"] == i]["_val"].min() if count > 0 else 0
            actual_max = gdf[gdf["_bin_index"] == i]["_val"].max() if count > 0 else 0
            print(f"     Bin {i+1:2d}: {label} ({count:4d} points) [actual: {actual_min:.6f}-{actual_max:.6f}]")

        # CRS handling
        if gdf.crs is None:
            warnings.warn("Input has no CRS. Assuming EPSG:4326")
            gdf = gdf.set_crs(4326)
        gdf_3857 = gdf.to_crs(WEB_MERCATOR)

        # Create plot
        fig, ax = plt.subplots(figsize=FIGSIZE)
        
        # Plot points by bin (darkest red = highest values = top 10%)
        for i, label in enumerate(bin_labels):
            sel = gdf_3857[gdf_3857["_bin_index"] == i]
            if not sel.empty:
                ax.scatter(
                    sel.geometry.x, sel.geometry.y,
                    s=POINT_SIZE, c=[cmap(i)], alpha=POINT_ALPHA,
                    linewidths=0
                )

        # Add grey canvas basemap
        try:
            cx.add_basemap(ax, source=BASEMAP, crs=f"EPSG:{WEB_MERCATOR}")
        except Exception as e:
            warnings.warn(f"Basemap failed to load ({e}).")

        # Styling
        ax.set_title(title, **MAP_TITLE_FONT)
        ax.set_axis_off()
        ax.set_aspect("equal")
        if not gdf_3857.empty:
            x0, y0, x1, y1 = gdf_3857.total_bounds
            ax.set_xlim(x0 - 50, x1 + 50)
            ax.set_ylim(y0 - 50, y1 + 50)

        # Add north arrow
        add_north_arrow(ax)

        # Create legend showing value ranges for each equal-count bin
        handles = [
            Line2D([0], [0], marker='o', linestyle='',
                   markerfacecolor=cmap(i), markeredgecolor=cmap(i),
                   markersize=6, label=label)
            for i, label in enumerate(bin_labels)
        ]
        
        # Legend title - emphasize equal count distribution
        legend_title = f"Equal Count Distribution\n(~{len(gdf)//n_groups} points per group)"
        if gpkg_path in S3_DATASETS:
            legend_title += "\n(Excludes 0.0000 values)"
        
        ax.legend(
            handles=handles,
            title=legend_title,
            loc="lower left", bbox_to_anchor=(0.01, 0.01),
            ncol=2, frameon=True, fontsize=7, title_fontsize=9, markerscale=1.2
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
        "gpkg_path": "combined_weighted_ev_locations.gpkg",
        "value_col": "combined_weight",
        "title": "S1: Suitable EV Charging Locations Based on All Cars",
        "out_png": "C4_s1_all_heatmap.png"
    },
    {
        "gpkg_path": "ev_combined_weighted_ev_locations.gpkg",
        "value_col": "ev_combined_weight", 
        "title": " S1: Suitable EV Charging Locations Based on EVs Only",
        "out_png": "C4_s1_ev_heatmap.png"
    },
    {
        "gpkg_path": "s2_household_income_combined_all_vehicles_core.gpkg",
        "value_col": "s2_all_vehicles_income_combined",
        "title": "S2: Suitable EV Charging Locations Based on All Cars and Income",
        "out_png": "C4_s2_all_income_heatmap.png"
    },
    {
        "gpkg_path": "s2_household_income_combined_ev_vehicles_core.gpkg",
        "value_col": "s2_ev_vehicles_income_combined",
        "title": "S2: Suitable EV Charging Locations Based on EVs and Income",
        "out_png": "C4_s2_ev_income_heatmap.png"
    },
    {
        "gpkg_path": "s3_1_primary_combined_all_vehicles.gpkg",
        "value_col": "s3_1_primary_combined_all_vehicles_weight",
        "title": "S3: Suitable EV Charging Locations Based on All Cars and Substation Capacity",
        "out_png": "C4_s3_1_all_heatmap.png"
    },
    {
        "gpkg_path": "s3_1_primary_combined_ev_vehicles.gpkg",
        "value_col": "s3_1_primary_combined_ev_vehicles_weight",
        "title": "S3: Suitable EV Charging Locations Based on EVs and Substation Capacity",
        "out_png": "C4_s3_1_ev_heatmap.png"
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
                n_groups=10  # Always 10 equal-count categories
            )
            
            if success:
                successful += 1
            else:
                failed += 1
        
        print("=" * 60)
        print(f"📊 SUMMARY: {successful} successful, {failed} failed, {len(missing_files)} missing")
        print(f"🎨 All heatmaps use:")
        print(f"   • Red color scheme (light→dark = low→high values)")
        print(f"   • Grey canvas basemap") 
        print(f"   • Equal count distribution (exactly same number of points per bin)")
        print(f"   • Value ranges determined by actual data distribution")
        print(f"   • Top 10% of points = darkest red")
        print(f"   • Bottom 10% of points = lightest red")
        print(f"   • S3 datasets exclude points with values = 0.0000")
        print(f"   • North arrow in upper left corner")
        
    else:
        print("❌ No valid datasets found to process!")
        
    print("🏁 Process completed.")